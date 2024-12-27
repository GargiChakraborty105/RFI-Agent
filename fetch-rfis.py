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

if __name__ == "__main__":
    load_dotenv()

    BASE_URL = "https://sandbox.procore.com/rest/v2.0/"
    access_token = os.getenv('ACCESS_TOKEN')

    try:
        # Step 1: Get the company ID
        companies = fetch_companies(access_token)
        print("Companies:", companies)
        company_id = companies["data"][0]["id"]  # Replace with the desired company
        print("Selected Company ID:", company_id)

        # Step 2: Fetch projects for the company
        projects = fetch_projects(access_token, company_id)
        print("Projects:", projects)
    except Exception as e:
        print("An error occurred:", e)
