import json
from typing import Any

from anthropic import AsyncAnthropic, omit

from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient

# Anthropic requires `max_tokens` in every request (OpenAI does not)
MAX_TOKENS = 4096


class AgentMCPFundamentals:
    """Handles AI model interactions and integrates with MCP client"""

    def __init__(self, api_key: str, model: str, tools: list[dict[str, Any]], mcp_client: MCPClient):
        self.model = model
        self.tools = self._to_anthropic_tools(tools)
        self.mcp_client = mcp_client
        self.client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert tools provided by `MCPClient.get_tools()` (OpenAI function-calling style) to Anthropic format:
        {"name": ..., "description": ..., "input_schema": <JSON Schema>}.
        Both the flat form and the nested one ({"type": "function", "function": {...}}) are supported.
        """
        converted = []
        for tool in tools:
            spec = tool.get("function", tool)
            converted.append({
                "name": spec["name"],
                "description": spec.get("description") or "",
                "input_schema": spec.get("parameters") or {"type": "object", "properties": {}},
            })
        return converted

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """
        Convert the conversation history to Anthropic format. Returns (system_prompt, messages):
        - `system` messages are removed from the list and joined into a separate `system` parameter
        - assistant tool calls become `tool_use` blocks (arguments are parsed from JSON string into a dict)
        - `tool` messages become `tool_result` blocks; consecutive results are merged into ONE user message
        """
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content)

            elif msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    blocks: list[dict[str, Any]] = []
                    if msg.content:  # Anthropic rejects empty text blocks
                        blocks.append({"type": "text", "text": msg.content})
                    for tool_call in msg.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"] or "{}"),
                        })
                    converted.append({"role": "assistant", "content": blocks})
                elif msg.content:
                    converted.append({"role": "assistant", "content": msg.content})

            elif msg.role == Role.TOOL:
                block = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}
                last = converted[-1] if converted else None
                if last and last["role"] == "user" and isinstance(last["content"], list):
                    last["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})

        return "\n\n".join(system_parts), converted

    async def _stream_response(self, messages: list[Message]) -> Message:
        """Stream Anthropic response and handle tool calls"""
        system, converted = self._to_anthropic_messages(messages)

        print("🤖: ", end="", flush=True)

        async with self.client.messages.stream(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system or omit,
                messages=converted,
                tools=self.tools or omit,
                temperature=0.0,  # supported by Claude Haiku 4.5 (Sonnet 5 rejects non-default values)
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            final_message = await stream.get_final_message()

        print()

        content = "".join(block.text for block in final_message.content if block.type == "text")
        tool_calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            }
            for block in final_message.content if block.type == "tool_use"
        ]

        return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    async def get_response(self, messages: list[Message]) -> Message:
        """Process user query with streaming and tool calling"""
        ai_message: Message = await self._stream_response(messages)

        if ai_message.tool_calls:
            messages.append(ai_message)
            await self._call_tools(ai_message, messages)
            return await self.get_response(messages)

        return ai_message

    async def _call_tools(self, ai_message: Message, messages: list[Message]):
        """Execute tool calls using MCP client"""
        #TODO:
        # 1. Iterate through `ai_message.tool_calls` (each item looks like
        #    {"id": ..., "type": "function", "function": {"name": ..., "arguments": "<JSON string>"}})
        # 2. Get tool name (`function.name`) and tool arguments (`function.arguments` is a JSON string, so parse it
        #    with `json.loads` into a dict before passing to the MCP client)
        # 3. Wrap into try/except block and call mcp_client tool call. If succeed then append a tool message
        #    `Message(role=Role.TOOL, ...)` with the result (don't forget `tool_call_id` = tool call id and `name`),
        #    otherwise append a tool message with the error text (it kind of fallback strategy).
        #    IMPORTANT for Anthropic: every `tool_use` MUST get a `tool_result`, otherwise the next request fails with
        #    400. So add a tool message for EVERY tool call, even a failed one. The history is converted to
        #    `tool_result` blocks in `_to_anthropic_messages`.
        raise NotImplementedError()
