import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def _build_request(self, messages: list[Message], max_tokens: int, stream: bool) -> tuple[str, dict, dict]:
        """
        Build URL, headers and JSON body shared by the sync and streaming calls.

        Gemini differs from Anthropic here: the model is part of the URL, the API key goes into
        'x-goog-api-key', the assistant role is called 'model', the system prompt is a separate
        'system_instruction' object and the token limit lives in 'generationConfig'.
        """
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = f"{self._endpoint}/{self._model_name}:{method}"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        body = {
            "system_instruction": {"parts": [{"text": self._system_prompt}]},
            "contents": [
                {
                    "role": "user" if m.role == Role.USER else "model",
                    "parts": [{"text": m.content}],
                }
                for m in messages
                if m.role in (Role.USER, Role.ASSISTANT)
            ],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        return url, headers, body

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Concatenate text parts of the first candidate (thought parts are skipped)."""
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        limit = kwargs.pop("max_tokens", 1024)
        url, headers, body = self._build_request(messages, limit, stream=False)

        try:
            response = requests.post(url=url, json=body, headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Could not reach Gemini API: {e}") from e

        if response.status_code != 200:
            raise Exception(f"Gemini API returned {response.status_code}: {response.text}")

        data = response.json()
        text = self._extract_text(data)
        if not text:
            raise ValueError(f"Gemini API response contains no text candidates: {data}")

        print(text)
        return Message(role=Role.ASSISTANT, content=text)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        limit = kwargs.pop("max_tokens", 1024)
        url, headers, body = self._build_request(messages, limit, stream=True)

        timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
        parts: list[str] = []

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"Gemini API returned {response.status}: {await response.text()}")

                    async for raw in response.content:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        chunk_text = self._extract_text(json.loads(line.removeprefix("data: ")))
                        if chunk_text:
                            print(chunk_text, end="", flush=True)
                            parts.append(chunk_text)
        except aiohttp.ClientError as e:
            raise Exception(f"Could not reach Gemini API: {e}") from e

        print()
        if not parts:
            raise ValueError("Gemini API stream contained no text")

        return Message(role=Role.ASSISTANT, content="".join(parts))
