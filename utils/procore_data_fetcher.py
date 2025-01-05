import requests
from dotenv import load_dotenv
import os
from utils.sqlOperator import Uploader

class Procore:
    def __init__(self):
        load_dotenv()

        self.BASE_URL = "https://sandbox.procore.com/rest/v2.0/"
        self.access_token = os.getenv('ACCESS_TOKEN')

    def fetch_projects(self, company_id):
        headers = {
            "Authorization": f"Bearer {self.access_token}"
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
    def fetch_rfis(self, company_id, project_id):
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Procore-Company-Id": f"{company_id}"
        }
        # try:
        response = requests.get(f"https://sandbox.procore.com/rest/v1.0/projects/{project_id}/rfis",headers=headers)
        return response.json()
    
    def fetch_companies(self):
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        response = requests.get(f"https://sandbox.procore.com/rest/v1.0/companies", headers=headers)
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

    def fetch_company_users(self, company_id):
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Procore-Company-Id" : f'{company_id}'
        }
        response = requests.get(f"https://sandbox.procore.com//rest/v1.3/companies/{company_id}/users", headers=headers)
        
        return response.json()