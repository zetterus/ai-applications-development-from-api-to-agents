from abc import abstractmethod, ABC
from typing import Optional, Any

from mcp import Client
from mcp.types import CallToolResult, TextContent, GetPromptResult, ReadResourceResult, Resource, TextResourceContents, \
    BlobResourceContents, Prompt
from pydantic import AnyUrl


class MCPClient(ABC):

    def __init__(self) -> None:
        self.client: Optional[Client] = None

    @abstractmethod
    async def __aenter__(self):
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        response = await self.client.list_tools()
        tools = response.tools
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        tool_result: CallToolResult = await self.client.call_tool(tool_name, tool_args)
        content = tool_result.content[0]
        print(f"    ⚙️: {content}\n")
        if isinstance(content, TextContent):
            return content.text
        else:
            return content

    async def get_resources(self) -> list[Resource]:
        """Get available resources from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        try:
            response = await self.client.list_resources()
            return response.resources
        except Exception as e:
            print(F"No resources available. Error message: {e}")
            return []

    async def get_resource(self, uri: AnyUrl) -> str:
        """Get specific resource content"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        result: ReadResourceResult = await self.client.read_resource(str(uri))
        item = result.contents[0]

        if isinstance(item, TextResourceContents):
            return item.text
        if isinstance(item, BlobResourceContents):
            return item.blob
        raise NotImplementedError("Invalid resource type.")

    async def get_prompts(self) -> list[Prompt]:
        """Get available prompts from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        try:
            response = await self.client.list_prompts()
            return response.prompts
        except Exception as e:
            print(F"No prompts available. Error message: {e}")
            return []

    async def get_prompt(self, name: str) -> str:
        """Get specific prompt content"""
        if not self.client:
            raise RuntimeError("MCP client not connected.")
        combined_content = ""
        response: GetPromptResult = await self.client.get_prompt(name)
        for message in response.messages:
            if message.content:
                if isinstance(message.content, TextContent):
                    combined_content += message.content.text + "\n"
                if isinstance(message.content, str):
                    combined_content += message.content + "\n"
        return combined_content
