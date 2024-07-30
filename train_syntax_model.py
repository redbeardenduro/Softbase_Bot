import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pennylane as qml
from transformers import T5TokenizerFast, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from pytorch_lightning.loggers import CSVLogger

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['rewritten_intent'].astype(str).tolist()
    targets = examples['snippet'].astype(str).tolist()
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Dataset class
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        return item

# DataModule to handle data loading
class CodeDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_json(self.data_file)
        self.dataset = CodeDataset(preprocess_function(df))
        print("DataModule setup complete.")

    def train_dataloader(self):
        print("Creating DataLoader...")
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, prefetch_factor=2, persistent_workers=True)

# Define a quantum device
dev = qml.device("default.qubit", wires=4)

# Define a quantum node
@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4))
    qml.BasicEntanglerLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Define a quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.weight_shapes = {"weights": (num_layers, num_qubits)}
        self.weights = nn.Parameter(torch.randn(*self.weight_shapes["weights"]))

    def forward(self, inputs):
        return quantum_circuit(inputs, self.weights)

# Lightning Module to handle the model
class T5FineTuner(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=5e-5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.quantum_layer = QuantumLayer(num_qubits=4, num_layers=6)
        self.dense = nn.Linear(512, 4)  # Dimensionality reduction layer

    def forward(self, input_ids, attention_mask, labels=None):
        reduced_input = self.dense(input_ids.float())
        q_out = self.quantum_layer(reduced_input)
        q_out = torch.stack(q_out, dim=1)  # Ensure correct stacking and dimension
        q_out = q_out.view(input_ids.size(0), -1)  # Reshape to match batch size
        q_out_padded = torch.nn.functional.pad(q_out, (0, 512 - q_out.size(1)))
        combined_input = torch.cat((input_ids.float(), q_out_padded), dim=1)[:, :512]  # Truncate to the maximum length of 512
        return self.model(input_ids=combined_input.int(), attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        print(f"Training step {batch_idx} complete. Loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_dataloader()) * self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

# Setup
print("Loading tokenizer and model...")
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
print("Tokenizer and model loaded.")

data_module = CodeDataModule(tokenizer, 'datasets/conala-corpus/conala-train.json')
t5_finetuner = T5FineTuner(model, tokenizer)
print("DataModule and model initialized.")

# Checkpointing and logging
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath='./results',
    filename='t5-finetuned-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min',
)
csv_logger = CSVLogger(save_dir='./logs/')

# Early stopping
early_stopping_callback = EarlyStopping(
    monitor='train_loss',
    patience=3,
    verbose=True,
    mode='min'
)

# Training
print("Starting training...")
trainer = pl.Trainer(
    max_epochs=3,
    precision='bf16-mixed',  # Use bfloat16 mixed precision for CPU
    callbacks=[checkpoint_callback, early_stopping_callback],
    logger=csv_logger,
    log_every_n_steps=10,  # Log every 10 steps
    val_check_interval=0.25,  # Validation check every 25% of the training epoch
)

trainer.fit(t5_finetuner, datamodule=data_module)
print("Training complete.")

# Save the model
print("Saving the model...")
model.save_pretrained('syntax_correction_model')
tokenizer.save_pretrained('syntax_correction_tokenizer')
print("Model saved.")
