from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from data_cleaner import DataCleaner
from starlette.responses import JSONResponse

app = FastAPI()
data_cleaner = DataCleaner()

class DuplicatesData(BaseModel):
    duplicates: list
    engine: str

class PartsData(BaseModel):
    parts_df: list
    engine: str

@app.post("/merge_duplicates")
def merge_duplicates(data: DuplicatesData):
    try:
        data_cleaner.merge_duplicates(data.duplicates, data.engine)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/create_new_part_numbers")
def create_new_part_numbers(data: PartsData):
    try:
        data_cleaner.create_new_part_numbers(data.parts_df, data.engine)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/update_database")
def update_database(data: PartsData):
    try:
        data_cleaner.update_database_with_cleaned_data(data.parts_df, data.engine)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/shutdown")
async def shutdown(request: Request):
    async def _shutdown():
        await request.app.state.loop.stop()
    request.app.state.loop.create_task(_shutdown())
    return JSONResponse(content={"message": "Server is shutting down"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
