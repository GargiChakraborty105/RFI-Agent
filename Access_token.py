import requests

#token generation url 

#https://login.procore.com/oauth/authorize?client_id=atpQwDiOJKEvGA35HP_dlcavpqa56b6v1gZbIWvrcRA&redirect_uri=http://localhost/callback&response_type=code

# Replace these with your app credentials
CLIENT_ID = "atpQwDiOJKEvGA35HP_dlcavpqa56b6v1gZbIWvrcRA"
CLIENT_SECRET = "GjVONicEuD1_CUDrh5qU5XxPo7VV-VpD2R8u-_GTgvk"
REDIRECT_URI = "http://localhost/callback"  # Same as what you provided during app creation
AUTH_CODE = "USO-R2gD5DuBK8VEHvggGGsKAKFkBRN5IGR7Sms1MHY"

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
print("Access Token:", access_token)