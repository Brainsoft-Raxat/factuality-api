import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import settings
from typing import Any

class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])  # Assuming GET and POST are the methods you want to retry
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.URL = settings.FACTUALITY_MODEL_URL
        self.TOKEN = "Bearer " + settings.FACTUALITY_MODEL_TOKEN

    def score_articles(self, payload: dict[str, Any]) -> dict:
        headers = {"Authorization": self.TOKEN}
        try:
            response = self.session.post(self.URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed with error: {e}")

