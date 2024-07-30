from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from email_notifier import EmailNotifier
from report_generator import ReportGenerator
from config.settings import Config
import logging
from starlette.responses import JSONResponse

app = FastAPI()
email_notifier = EmailNotifier(Config.FROM_EMAIL, Config.FROM_PASSWORD, 'mail.shaw.ca', 587)
report_generator = ReportGenerator()

# Setup logging
logging.basicConfig(level=logging.INFO)

class PartsData(BaseModel):
    parts_df: list  # Update to list to match the incoming data structure

class EmailData(BaseModel):
    to_email: str
    subject: str
    body: str

@app.post("/send_email")
def send_email(email_data: EmailData):
    try:
        email_notifier.send_email(email_data.to_email, email_data.subject, email_data.body)
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/generate_report")
def generate_report(data: PartsData):
    try:
        logging.info(f"Received data for report generation: {data.parts_df}")
        report_generator.generate_visual_reports(data.parts_df)
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/shutdown")
async def shutdown(request: Request):
    async def _shutdown():
        await request.app.state.loop.stop()
    request.app.state.loop.create_task(_shutdown())
    return JSONResponse(content={"message": "Server is shutting down"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
