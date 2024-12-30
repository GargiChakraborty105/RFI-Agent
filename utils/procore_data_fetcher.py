import requests
from dotenv import load_dotenv
import os
from utils.sqlUploader import Uploader

class Procore:
    def __init__(self):
        load_dotenv()

        self.BASE_URL = "https://sandbox.procore.com/rest/v2.0/"
        self.access_token = os.getenv('ACCESS_TOKEN')