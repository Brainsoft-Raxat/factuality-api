import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from .config import settings


class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.URL = settings.BASE_MODEL_URL
        self.TOKEN = "Bearer " + settings.BASE_MODEL_TOKEN
        self.FACTUALITY = settings.FACTUALITY_MODEL_PATH
        self.FREEDOM = settings.FREEDOM_MODEL_PATH
        self.BIAS = settings.BIAS_MODEL_PATH
        self.GENRE = settings.GENRE_MODEL_PATH

    def score_factuality(self, payload: dict[str, Any]) -> dict:
        headers = {"Authorization": self.TOKEN}
        try:
            response = self.session.post(
                self.URL + self.FACTUALITY, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed with error: {e}")

    def score_freedom(self, payload: dict[str, Any]) -> dict:
        headers = {"Authorization": self.TOKEN}
        try:
            response = self.session.post(
                self.URL + self.FREEDOM, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed with error: {e}")

    def score_bias(self, payload: dict[str, Any]) -> dict:
        headers = {"Authorization": self.TOKEN}
        try:
            response = self.session.post(
                self.URL + self.BIAS, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed with error: {e}")

    def score_genre(self, payload: dict[str, Any]) -> dict:
        headers = {"Authorization": self.TOKEN}
        try:
            response = self.session.post(
                self.URL + self.GENRE, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed with error: {e}")

    def score_articles_sequentially(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        resulting_scores = []

        def score_article(input_text):
            payload = {"inputs": input_text}
            scores = {"factuality": {}, "freedom": {}, "bias": {}, "genre": {}}

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_model = {
                    executor.submit(self.score_factuality, payload): "factuality",
                    executor.submit(self.score_freedom, payload): "freedom",
                    executor.submit(self.score_bias, payload): "bias",
                    executor.submit(self.score_genre, payload): "genre"
                }
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        response = future.result()
                        if model_name == "factuality":
                            for item in response[0]:
                                if item['label'] == 'neutral':
                                    scores[model_name]['MIXED'] = item['score']
                                elif item['label'] == 'entailment':
                                    scores[model_name]['HIGH'] = item['score']
                                elif item['label'] == 'contradiction':
                                    scores[model_name]['LOW'] = item['score']
                        elif model_name == "freedom":
                            for item in response[0]:
                                label = item['label'].replace('LABEL_', '')
                                label_map = {
                                    "0": "MOSTLY_FREE",
                                    "1": "EXCELLENT",
                                    "2": "LIMITED_FREEDOM",
                                    "3": "TOTAL_OPPRESSION",
                                    "4": "MODERATE_FREEDOM"
                                }
                                scores[model_name][label_map[label]
                                                   ] = item['score']
                        elif model_name == "bias":
                            for item in response[0]:
                                label = item['label'].replace('LABEL_', '')
                                label_map = {
                                    "0": "LEAST_BIASED",
                                    "1": "FAR_RIGHT",
                                    "2": "RIGHT",
                                    "3": "RIGHT_CENTER",
                                    "4": "LEFT",
                                    "5": "LEFT_CENTER",
                                    "6": "FAR_LEFT"
                                }
                                scores[model_name][label_map[label]
                                                   ] = item['score']
                        elif model_name == "genre":
                            for item in response[0]:
                                if item['label'] == 'opinion':
                                    scores[model_name]['OPINION'] = item['score']
                                elif item['label'] == 'satire':
                                    scores[model_name]['SATIRE'] = item['score']
                                elif item['label'] == 'reporting':
                                    scores[model_name]['REPORTING'] = item['score']
                    except Exception as exc:
                        print(f"{model_name} model generated an exception: {exc}")
            return scores

        # Process each article sequentially
        for input_text in input_texts:
            article_scores = score_article(input_text)
            resulting_scores.append(article_scores)

        return resulting_scores

    # implementation by creating max of
    def score_articles_concurrently(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        resulting_scores = []

        def score_article(input_text):
            payload = {"inputs": input_text}
            scores = {"factuality": {}, "freedom": {}, "bias": {}, "genre": {}}

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_model = {
                    executor.submit(self.score_factuality, payload): "factuality",
                    executor.submit(self.score_freedom, payload): "freedom",
                    executor.submit(self.score_bias, payload): "bias",
                    executor.submit(self.score_genre, payload): "genre"
                }
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        response = future.result()
                        if model_name == "factuality":
                            for item in response[0]:
                                if item['label'] == 'neutral':
                                    scores[model_name]['MIXED'] = item['score']
                                elif item['label'] == 'entailment':
                                    scores[model_name]['HIGH'] = item['score']
                                elif item['label'] == 'contradiction':
                                    scores[model_name]['LOW'] = item['score']
                        elif model_name == "freedom":
                            for item in response[0]:
                                label = item['label'].replace('LABEL_', '')
                                label_map = {
                                    "0": "MOSTLY_FREE",
                                    "1": "EXCELLENT",
                                    "2": "LIMITED_FREEDOM",
                                    "3": "TOTAL_OPPRESSION",
                                    "4": "MODERATE_FREEDOM"
                                }
                                scores[model_name][label_map[label]
                                                   ] = item['score']
                        elif model_name == "bias":
                            for item in response[0]:
                                label = item['label'].replace('LABEL_', '')
                                label_map = {
                                    "0": "LEAST_BIASED",
                                    "1": "FAR_RIGHT",
                                    "2": "RIGHT",
                                    "3": "RIGHT_CENTER",
                                    "4": "LEFT",
                                    "5": "LEFT_CENTER",
                                    "6": "FAR_LEFT"
                                }
                                scores[model_name][label_map[label]
                                                   ] = item['score']
                        elif model_name == "genre":
                            for item in response[0]:
                                if item['label'] == 'opinion':
                                    scores[model_name]['OPINION'] = item['score']
                                elif item['label'] == 'satire':
                                    scores[model_name]['SATIRE'] = item['score']
                                elif item['label'] == 'reporting':
                                    scores[model_name]['REPORTING'] = item['score']
                    except Exception as exc:
                        print(f"{model_name} model generated an exception: {exc}")
            return scores

        with ThreadPoolExecutor(max_workers=len(input_texts)) as executor:
            future_to_article = {executor.submit(
                score_article, text): text for text in input_texts}
            for future in as_completed(future_to_article):
                try:
                    article_scores = future.result()
                    resulting_scores.append(article_scores)
                except Exception as exc:
                    print(f"Article processing generated an exception: {exc}")

        return resulting_scores
