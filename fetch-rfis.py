import requests
from dotenv import load_dotenv
import os

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
    load_dotenv()

    BASE_URL = "https://sandbox.procore.com/rest/v2.0/"
    access_token = os.getenv('ACCESS_TOKEN')

    try:
        # Step 1: Get the company ID
        companies = fetch_companies(access_token)
        print("Companies:", companies)
        company_id = companies["data"][1]["id"]  # Replace with the desired company
        print("Selected Company ID:", company_id)
        print(type(company_id))
        # Step 2: Fetch projects for the company
        projects = fetch_projects(access_token, company_id)
        print("Projects:", projects)
        print(len(projects))
        project_id = projects[0]["id"]
        print(project_id)
        rfis = fetch_rfis(access_token, company_id, project_id)
        print(f"rfis : {rfis}")
        print(len(rfis))
    except Exception as e:
        print("An error occurred:", e)
