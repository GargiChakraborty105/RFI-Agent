import webbrowser

# Replace these with your app details
client_id = "atpQwDiOJKEvGA35HP_dlcavpqa56b6v1gZbIWvrcRA"
redirect_uri = "http://localhost/callback"
scope = "read:rfi"  # Example scope

auth_url = f"https://sandbox.procore.com/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code"

# Open the authorization URL in a browser
webbrowser.open(auth_url)

print(f"Visit this URL to log in: {auth_url}")
