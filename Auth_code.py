import dotenv
import os

dotenv.load_dotenv()



# Replace these with your app details
client_id = os.getenv("CLIENT_ID")
redirect_uri = os.getenv("REDIRECT_URI")

auth_url = f"https://sandbox.procore.com/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code"


print(f"Visit this URL to log in: {auth_url}")
