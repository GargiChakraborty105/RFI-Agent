import requests

# Replace these with your app credentials
CLIENT_ID = "mHxlqjw5QsyHmIb4XWKaOVuBZ60IvOjpQLAZPiryysM"
CLIENT_SECRET = "XGNaagkLBFdr4KLHwb1cc8yOz60BHa44DVGPxC3tStA"
REDIRECT_URI = "http://localhost/callback"  # Same as what you provided during app creation
AUTH_CODE = "5NUjMn7HTElnsFqxAhncDO9ikxugXOvUqoDdrzaYhuY"

TOKEN_URL = "https://login.procore.com/oauth/token"

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