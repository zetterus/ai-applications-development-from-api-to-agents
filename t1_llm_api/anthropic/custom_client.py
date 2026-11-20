import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        limit = kwargs.pop("max_tokens", 1024)
        converted = [m.to_dict() for m in messages if m.role in (Role.ASSISTANT, Role.USER)]
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2026-10-21",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model_name,
            "max_tokens": limit,
            "system": self._system_prompt,
            "messages": converted,
        }
        try:
            response = requests.post(url=self._endpoint, json=body, headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Could not reach Claude's API: {e}") from e

        if response.status_code != 200:
            raise Exception(f"An error occurred while connecting to Claude's API: {response.status_code}. Response: {response.text}")

        data = response.json()

        collector = []
        for part in data.get("content") or []:
            if part["type"] == "text":
                collector.append(part["text"])
        if not collector:
            raise ValueError(f"Claude's API response contains no text content blocks: {data}")
        text = "".join(collector)
        print(text)
        return Message(role=Role.ASSISTANT, content=text)


    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        #TODO:
        # https://docs.anthropic.com/en/docs/build-with-claude/streaming
        # - Prepare headers with api key, anthropic version and content type
        # - Add System prompt
        # - Execute post request to AI API (use `aihttp`)
        # - Handle stream with chunks
        # - Parse response
        # - Print chunks to console
        # - Return ASSISTANT message
        raise NotImplementedError

