import json
import urllib.error
import urllib.request

from apps.agent_service.domain.interfaces.data_api_client import DataAPIClient
from shared.contracts.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)


class DataAPIError(RuntimeError):
    """Raised when communication with the Data API fails."""


class HTTPDataAPIClient(DataAPIClient):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        url = f"{self.base_url}/api/v1/retrieval"

        payload = {
            "query": request.query,
            "source": request.source,
            "top_k": request.top_k,
            "filters": request.filters,
            "retrieval_mode": request.retrieval_mode.value,
        }

        http_request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise DataAPIError(
                f"Data API returned HTTP {error.code}: {body}"
            ) from error

        except urllib.error.URLError as error:
            raise DataAPIError(
                f"Unable to connect to Data API: {error.reason}"
            ) from error

        except TimeoutError as error:
            raise DataAPIError(
                f"Data API request timed out after "
                f"{self.timeout_seconds} seconds."
            ) from error

        return RetrievalResponse(
            query=response_data["query"],
            source=response_data["source"],
            result_count=response_data["result_count"],
            results=[
                RetrievalResult(
                    content=result["content"],
                    score=result["score"],
                    source=result["source"],
                    metadata=result.get("metadata", {}),
                )
                for result in response_data["results"]
            ],
        )