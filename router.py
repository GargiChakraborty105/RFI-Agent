from fastapi import FastAPI, HTTPException, Query
from utils.sqlOperator import Fetcher, Uploader
from utils.procore_data_fetcher import Procore
from Analysis import RfiAnalysis, AssignAssistance
from datetime import datetime, timedelta

app = FastAPI()

# @app.get("/rfis")
# async def get_rfis(project_id: str = Query(..., description="Project ID to fetch RFIs")):
#     try:
#         fetch = Fetcher()
#         results = fetch.list_rfi(project_id)
#         rfis = []
#         for result in results:

#         fetch.close_connection()
#         return {"status": "success", "data": result}
#     except Exception as e:
#         print("Error processing request:", e)
#         raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/dashboard/company/{company_id}/project/{project_id}")
async def get_projects(company_id: int, project_id: int):
    try:
        fetch = Fetcher()
        results = fetch.list_rfi(project_id)
        fetch.close_connection()
        rfi_counts = len(results['rfis'])
        results['rfi_counts'] = rfi_counts
        users = fetch.list_user(company_id)
        rfis = []
        for rfi in results['rfis']:
            analytics = fetch.list_analytics(rfi['id'])
            assigner = AssignAssistance(users, [rfi])
            analytics
            rfis.append(analytics)
        results['rfis'] = rfis

        return {"status": "success", "data": results}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/initialise_analytics_table")
async def initialise_analytics_table():
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
                i = 0
                while i < len(rfis):
                    x = rfis[i]
                    if x['status'] == 'closed':
                        print(f'removed: {x}')
                        rfis.remove(x)
                        continue
                    print(type(x))
                    x['project_id'] = project_id
                    x['assignees_name'] = [y['name'] for y in x['assignees']]
                    x['assignees_id'] = [y['id'] for y in x['assignees']]
                    x['priority_name'] = x['priority']['name']
                    x['priority_value'] = x['priority']['value'] if x['priority']['value'] is not None else 0
                    x['questions_body'] = [y['body'] for y in x['questions']]
                    print("converting dates")
                    x['updated_at'] = datetime.strptime(x['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
                    x['due_date'] = datetime.strptime(x['due_date'], '%Y-%m-%d')
                    i+=1
                print(rfis)
                
                analyser = RfiAnalysis(rfis)
                analytics = analyser.run_analysis()

                upload.analytics_uploader(analytics)
                
                # upload.rfi_uploader(rfis)
            # upload.projects_uploader(projects)
        upload.close_connection()
        return {'status' : 200, 'message': "RFI Table Updated Successfully"}
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
        upload.close_connection()
        return {'status' : 200, 'message': "RFI Table Updated Successfully"}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get('/initialize_user_table')
async def initialize_user_table():
    try:
        upload = Uploader()
        fetcher = Fetcher()
        procore_fetch = Procore()
        print('\n\nFetching Company\n')
        companies = procore_fetch.fetch_companies()
        print("Companies:", companies)
        company_ids = [companies[x]["id"] for x in range(len(companies))]
        for company_id in company_ids:
            print('\n\nFetching Users\n')
            users = procore_fetch.fetch_company_users(company_id)
            print(users)
            if len(users)<1:
                continue
            if type(users) != list:
                continue
            print(f"rfis : {users}")
            print(type(users))
            print("Users:", users)
            print(len(users))
            for user in users:
                user['current_workload'] = fetcher.current_workload_calculator(user['id'])
                user['historical_performance_score'] = fetcher.historical_performance_calculator(user['id'])
                user['company_id'] = company_id
            print('\n\nUploaading Users\n')
            upload.user_uploader(users)
        upload.close_connection()
        fetcher.close_connection()
        return {'status' : 200, 'message': "RFI Table Updated Successfully"}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)