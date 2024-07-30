from fastapi import FastAPI, Request
from data_loader import DataLoader
import logging
from starlette.responses import JSONResponse

app = FastAPI()
data_loader = DataLoader()

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.get("/load_parts_data")
def load_parts_data():
    parts_data = data_loader.load_parts_data()
    logging.info(f"Returning parts data: {parts_data}")
    return {"parts_data": parts_data}

@app.get("/shutdown")
async def shutdown(request: Request):
    async def _shutdown():
        await request.app.state.loop.stop()
    request.app.state.loop.create_task(_shutdown())
    return JSONResponse(content={"message": "Server is shutting down"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
