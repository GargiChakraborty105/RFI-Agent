import requests
import dotenv
import os
#token generation url 
dotenv.load_dotenv()
#https://login.procore.com/oauth/authorize?client_id=atpQwDiOJKEvGA35HP_dlcavpqa56b6v1gZbIWvrcRA&redirect_uri=http://localhost/callback&response_type=code

# Replace these with your app credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")  # Same as what you provided during app creation
AUTH_CODE = os.getenv("AUTH_CODE")

TOKEN_URL = "https://sandbox.procore.com/oauth/token"

# Exchange the authorization code for an access token
payload = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
    "code": AUTH_CODE,
}

response = requests.post(TOKEN_URL, data=payload)
response.raise_for_status()

# Extract the access token
access_token = response.json().get("access_token")
dotenv.set_key(".env","ACCESS_TOKEN", access_token)
print("Access Token:", access_token)