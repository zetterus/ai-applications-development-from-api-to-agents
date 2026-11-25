import os
import sys
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Resource
from mcp.types import Prompt

from commons.constants import ANTHROPIC_API_KEY, ANTHROPIC_HAIKU_MODEL
from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.agent import AgentMCPFundamentals
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient
from t9_mcp_fundamentals.agent.mcp_clients.http import HttpMCPClient
from t9_mcp_fundamentals.agent.mcp_clients.stdio import StdioMCPClient
from t9_mcp_fundamentals.agent.prompts import SYSTEM_PROMPT


async def main():
    clients: list[MCPClient] = [
        HttpMCPClient(mcp_server_url="http://localhost:8005/mcp"),
        HttpMCPClient(mcp_server_url="http://localhost:8010/mcp"),
    ]

    async with AsyncExitStack() as stack:
        connected_clients = [await stack.enter_async_context(client) for client in clients]

        all_resources: list[Resource] = []
        all_tools: list[dict] = []
        all_prompts: list[Prompt] = []
        tool_clients: dict[str, MCPClient] = {}

        for mcp_client in connected_clients:
            all_resources.extend(await mcp_client.get_resources())
            all_prompts.extend(await mcp_client.get_prompts())

            client_tools = await mcp_client.get_tools()
            for tool in client_tools:
                tool_name = tool["name"]
                if tool_name in tool_clients:
                    raise ValueError(f"Duplicate tool name '{tool_name}' across MCP servers.")
                tool_clients[tool_name] = mcp_client
            all_tools.extend(client_tools)

        print("\nAvailable Resources:")
        for resource in all_resources:
            print(f"  - {resource.uri} ({resource.mimeType}): {resource.description}")

        print("\nAvailable Tools:")
        for tool in all_tools:
            print(f"  - {tool['name']}: {tool.get('description', '')}")

        agent = AgentMCPFundamentals(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_HAIKU_MODEL,
            tools=all_tools,
            mcp_client=tool_clients,
        )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ]

        print("\nAvailable Prompts:")
        for prompt in all_prompts:
            print(f"  - {prompt.name}: {prompt.description}")

        while True:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            messages.append(Message(role=Role.USER, content=user_input))
            ai_message = await agent.get_response(messages)
            messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())
