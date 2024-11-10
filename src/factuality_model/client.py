import httpx
import asyncio
import time
from typing import Any, Dict, List
from src.factuality_model.config import settings


class AsyncClient:
    def __init__(self, max_retries: int = 10, backoff_factor: float = 0.5):
        self.TOKEN = "Bearer " + settings.HF_API_KEY
        self.factuality_url = settings.FACTUALITY_MODEL_URL
        self.bias_url = settings.BIAS_MODEL_URL
        self.genre_url = settings.GENRE_MODEL_URL
        self.persuasion_url = settings.PERSUASION_MODEL_URL
        self.framing_url = settings.FRAMING_MODEL_URL
        self.headers = {"Authorization": self.TOKEN}
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def _make_request(
        self, url: str, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(url, headers=self.headers, json=data)
                    response.raise_for_status()
                    response_data = response.json()

                    # Ensure consistent output format
                    if isinstance(response_data, list):
                        return [
                            {
                                "label": item.get("label", "unknown"),
                                "score": item.get("score", 0.0),
                            }
                            for item in response_data
                        ]
                    elif "label" in response_data and "score" in response_data:
                        return [
                            {
                                "label": response_data["label"],
                                "score": response_data["score"],
                            }
                        ]
                    else:
                        return [
                            {
                                "label": "unknown",
                                "score": 0.0,
                                "error": "Unexpected response format",
                            }
                        ]

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == self.max_retries:
                        return [
                            {
                                "error": f"Request failed after {self.max_retries} retries: {str(e)}",
                                "url": url,
                            }
                        ]
                    else:
                        wait_time = self.backoff_factor * (2 ** (attempt - 1))
                        print(
                            f"Retrying {url} in {wait_time:.2f} seconds (attempt {attempt}/{self.max_retries})..."
                        )
                        await asyncio.sleep(wait_time)

    async def get_all_scores(self, text: str) -> Dict[str, Any]:
        requests_data = {
            "factuality": (
                self.factuality_url,
                {
                    "inputs": text,
                    "parameters": {"top_k": "10", "function_to_apply": "softmax"},
                },
            ),
            "bias": (
                self.bias_url,
                {
                    "inputs": text,
                    "parameters": {"top_k": "10", "function_to_apply": "softmax"},
                },
            ),
            "genre": (
                self.genre_url,
                {
                    "inputs": text,
                    "parameters": {"top_k": "10", "function_to_apply": "softmax"},
                },
            ),
            "persuasion": (
                self.persuasion_url,
                {
                    "inputs": text,
                    "parameters": {"top_k": "10", "function_to_apply": "softmax"},
                },
            ),
            "framing": (
                self.framing_url,
                {
                    "inputs": text,
                    "parameters": {"top_k": 5, "function_to_apply": "softmax"},
                },
            ),
        }

        tasks = [self._make_request(url, data) for url, data in requests_data.values()]
        start_time = time.time()

        results = await asyncio.gather(*tasks)
        results_dict = {key: res for key, res in zip(requests_data.keys(), results)}

        # Check if any category contains valid scores; otherwise, provide defaults
        for key, value in results_dict.items():
            # Ensure each entry is a list of dictionaries with "label" and "score"
            if not any("label" in entry and "score" in entry for entry in value):
                results_dict[key] = [{"label": "unknown", "score": 0.0}]

        results_dict["execution_time"] = time.time() - start_time
        return results_dict


if __name__ == "__main__":
    client = AsyncClient()
    text = "As she sweeps up broken glass outside her shop, ..."
    results = asyncio.run(client.get_all_scores(text))
    print(results)
