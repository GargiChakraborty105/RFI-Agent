from fastapi import FastAPI, HTTPException, Query
from utils.sqlOperator import Fetcher, Uploader
from utils.procore_data_fetcher import Procore


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

@app.get("/initialise_rfi_table")
async def initialise_database():
    try:
        upload = Uploader()
        procore_fetch = Procore()
        companies = procore_fetch.fetch_companies()
        print("Companies:", companies)
        company_ids = [companies[x]["id"] for x in range(len(companies))]
        for company_id in company_ids:
            projects = procore_fetch.fetch_projects(company_id)
            print("Projects:", projects)
            print(len(projects))
            for x in projects:
                x['company_id'] = x['company']['id']
                x['company_name'] = x['company']['name']
            project_ids = [projects[i]["id"] for i in range(len(projects))]
            for project_id in project_ids:
                rfis = procore_fetch.fetch_rfis(company_id, project_id)
                if len(rfis)<1:
                    continue
                if type(rfis) != list:
                    continue
                print(f"rfis : {rfis}")
                print(type(rfis))
                for x in rfis:
                    print(type(x))
                    x['project_id'] = project_id
                    x['assignees_name'] = [y['name'] for y in x['assignees']]
                    x['assignees_id'] = [y['id'] for y in x['assignees']]
                    x['priority_name'] = x['priority']['name']
                    x['priority_value'] = x['priority']['value']
                    x['questions_body'] = [y['body'] for y in x['questions']]
                print(rfis)
                
                upload.rfi_uploader(rfis)
            upload.projects_uploader(projects)
        return {'status' : 200, 'message': "RFI Table Updated Successfully"}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)