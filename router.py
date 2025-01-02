from fastapi import FastAPI, HTTPException, Query
from utils.sqlOperator import Fetcher

app = FastAPI()

@app.get("/rfis")
async def get_rfis(project_id: str = Query(..., description="Project ID to fetch RFIs")):
    try:
        fetch = Fetcher()
        result = fetch.list_rfi(project_id)
        fetch.close_connection()
        return {"status": "success", "data": result}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/projects")
async def get_projects(company_id: str = Query(..., description="Company ID to fetch projects")):
    try:
        fetch = Fetcher()
        result = fetch.list_projects(company_id)
        fetch.close_connection()
        return {"status": "success", "data": result}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)