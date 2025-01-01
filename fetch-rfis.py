import requests
from dotenv import load_dotenv
import os
from utils.sqlOperator import Uploader

# Fetch the list of companies to get the company_id
def fetch_companies(access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(f"{BASE_URL}/companies", headers=headers)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error while fetching companies: {err}")
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        raise
    except ValueError as err:
        print(f"JSON Parsing Error while fetching companies: {err}")
        print("Response Text:", response.text)
        raise
def fetch_company_users(access_token, company_id, project_id):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Procore-Company-Id" : company_id
    }
    response = requests.get(f"https://sandbox.procore.com/rest/v1.0/projects/{project_id}/rfis/potential_rfi_managers", headers=headers)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error while fetching projects: {err}")
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        raise
    except ValueError as err:
        print(f"JSON Parsing Error while fetching projects: {err}")
        print("Response Text:", response.text)
        raise
# Fetch projects using the company_id
def fetch_projects(access_token, company_id):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(f"https://sandbox.procore.com/rest/v1.0/projects?company_id={company_id}", headers=headers)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error while fetching projects: {err}")
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        raise
    except ValueError as err:
        print(f"JSON Parsing Error while fetching projects: {err}")
        print("Response Text:", response.text)
        raise
def fetch_rfis(access_token, company_id, project_id):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Procore-Company-Id": f"{company_id}"
    }
    # try:
    response = requests.get(f"http://sandbox.procore.com/rest/v1.0/projects/{project_id}/rfis",headers=headers)
    return response.json()
    # except:
        # return "Error Fetching RFIs"
if __name__ == "__main__":
    load_dotenv(dotenv_path='.env', override=True)

    BASE_URL = "https://sandbox.procore.com/rest/v2.0"
    access_token = os.environ.get('ACCESS_TOKEN')
    try:
        # Step 1: Get the company ID
        companies = fetch_companies(access_token)
        print("Companies:", companies)
        company_id = companies["data"][0]["id"]  # Replace with the desired company
        print("Selected Company ID:", company_id)
        print(type(company_id))

        # Step 2: Fetch projects for the company
        projects = fetch_projects(access_token, company_id)
        print("Projects:", projects)
        print(len(projects))
        for x in projects:
            x['company_id'] = x['company']['id']
            x['company_name'] = x['company']['name']
        project_id = projects[0]["id"]
        print(project_id)
        users = fetch_company_users(access_token, company_id, project_id)
        print(f'users : {users}')
        rfis = fetch_rfis(access_token, company_id, project_id)
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
        upload = Uploader()
        upload.rfi_uploader(rfis)
        upload.projects_uploader(projects)
    except Exception as e:
        print("An error occurred:", e)
