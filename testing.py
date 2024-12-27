import requests

# Replace with your access token
access_token = 'eyJhbGciOiJFUzUxMiJ9.eyJhbXIiOltdLCJhaWQiOiJtSHhscWp3NVFzeUhtSWI0WFdLYU9WdUJaNjBJdk9qcFFMQVpQaXJ5eXNNIiwiYW91aWQiOjEyMjA5ODI3LCJhb3V1aWQiOiJhM2ZmZDZmNy01YjNmLTRmMGEtOTJjNi0zZDY0ZmIwODc1ZjIiLCJleHAiOjE3MzUyOTQzMDEsInNpYXQiOjE3MzUyODg4ODEsInVpZCI6MTIyMDk4MjcsInV1aWQiOiJhM2ZmZDZmNy01YjNmLTRmMGEtOTJjNi0zZDY0ZmIwODc1ZjIiLCJsYXN0X21mYV9jaGVjayI6MTczNTI4ODkwMX0.ACrs5ZctMMCfMRjZ76yL_0HREEw9Ho9eagGUddkHMShbGyMr7fQVnJ4YJTHPVqhD3hwOa_2AU6oybvrDR_YbFzqhAd_h7kU_6tJEHsH-aGZ3wDaUiXAE4dB-NiAcY5xi1y35rHfY0g4BhylDNM5EvVpneknJet8kr44sHnnyb57I_pxl'

# Set the API endpoint for companies
url = "https://api.procore.com/rest/v1.0/companies"

headers = {
    "Authorization": f"Bearer {access_token}",
}

response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    companies = response.json()
    print(companies)
else:
    print(f"Error: {response.status_code}, {response.text}")