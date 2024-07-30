from fastapi import FastAPI, Request
from chatgpt_integration import ChatGPTIntegration
import logging
from starlette.responses import JSONResponse

app = FastAPI()
chatgpt = ChatGPTIntegration()

@app.post("/generate_report")
def generate_report(parts_df: dict):
    logging.info(f"Received parts_df for report generation: {parts_df}")
    report = chatgpt.generate_report(parts_df["parts_df"])
    return {"report": report}

@app.post("/find_duplicates")
def find_duplicates(parts_df: dict):
    logging.info(f"Received parts_df for finding duplicates: {parts_df}")
    duplicates = chatgpt.find_duplicates(parts_df["parts_df"])
    return {"duplicates": duplicates}

@app.get("/shutdown")
async def shutdown(request: Request):
    async def _shutdown():
        await request.app.state.loop.stop()
    request.app.state.loop.create_task(_shutdown())
    return JSONResponse(content={"message": "Server is shutting down"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
