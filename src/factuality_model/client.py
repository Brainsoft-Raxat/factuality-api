import requests
import json 

from typing import Any

from .config import settings


class Client:
    URL: str = settings.FACTUALITY_MODEL_URL
    TOKEN: str = "Bearer " + settings.FACTUALITY_MODEL_TOKEN

    @property
    def client(self):
        return requests.Session()

    def score_articles(self, payload: dict[str, Any]) -> dict:
        with self.client as client:
            headers = {"Authorization": self.TOKEN}
            return client.post(self.URL, headers=headers, json=payload)
