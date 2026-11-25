import json
from collections.abc import Mapping
from typing import Any

from anthropic import AsyncAnthropic, omit

from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient

# Anthropic requires `max_tokens` in every request (OpenAI does not)
MAX_TOKENS = 4096


class AgentMCPFundamentals:
    """Handles AI model interactions and integrates with one or more MCP clients"""

    def __init__(
        self,
        api_key: str,
        model: str,
        tools: list[dict[str, Any]],
        mcp_client: MCPClient | Mapping[str, MCPClient],
    ):
        self.model = model
        self.tools = self._to_anthropic_tools(tools)
        if isinstance(mcp_client, Mapping):
            self.tool_clients: dict[str, MCPClient] = dict(mcp_client)
            self.default_client: MCPClient | None = None
        else:
            self.tool_clients = {}
            self.default_client = mcp_client
        self.client = AsyncAnthropic(api_key=api_key)

    def _get_client_for_tool(self, tool_name: str) -> MCPClient:
        """Return the MCP client responsible for executing `tool_name`."""
        client = self.tool_clients.get(tool_name) or self.default_client
        if client is None:
            raise ValueError(f"No MCP client registered for tool '{tool_name}'.")
        return client

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
        """Execute tool calls using the matching MCP client for each tool"""
        for tool_call in ai_message.tool_calls or []:
            tool_call_id = tool_call["id"]
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"].get("arguments") or "{}")
                mcp_client = self._get_client_for_tool(tool_name)
                result = await mcp_client.call_tool(tool_name, tool_args)
                content = result if isinstance(result, str) else str(result)
            except Exception as e:
                content = f"Error calling tool '{tool_name}': {e}"

            messages.append(
                Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
