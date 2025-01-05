import os
import requests
# Replace with your Procore credentials
PROCORE_CLIENT_ID = "atpQwDiOJKEvGA35HP_dlcavpqa56b6v1gZbIWvrcRA"
PROCORE_CLIENT_SECRET = "GjVONicEuD1_CUDrh5qU5XxPo7VV-VpD2R8u-_GTgvk"
PROCORE_API_URL = "https://sandbox.procore.com"
PROCORE_WEBHOOK_URL = "http://localhost/procore/webhook"  # Replace with your actual public URL

def get_procore_access_token():
    """Get an access token from Procore."""
    token_url = f"{PROCORE_API_URL}/oauth/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": PROCORE_CLIENT_ID,
        "client_secret": PROCORE_CLIENT_SECRET,
    }
    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get access token: {response.text}")
    return response.json().get("access_token")

def create_procore_webhook():
    """Register a webhook with Procore."""
    access_token = get_procore_access_token()
    webhook_url = f"{PROCORE_API_URL}/vapid/webhooks"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "resource_name": "projects",  # Change to "users" for user events
        "event_type": "project.create",  # Or "project.update", "user.create", etc.
        "endpoint_url": PROCORE_WEBHOOK_URL,
    }
    response = requests.post(webhook_url, json=payload, headers=headers)
    if response.status_code == 201:
        print("Webhook created successfully!")
        print(response.json())
    else:
        print(f"Failed to create webhook: {response.text}")

# Call the function to create a webhook
if __name__ == "__main__":
    create_procore_webhook()
