import asyncio
import sys
from mcp import Client, StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable, args=["-m","app.mcp_server.server"])

async def main():
    async with Client(SERVER) as client:
        tools = await client.list_tools()
        for tool in tools.tools:
            print(f"TOOL {tool.name}: {tool.description} \n schema: {tool.input_schema}")

        for args in ({"name": "leave"},{"name":"foo"}):
            result = await client.call_tool("read_policy",args)
            print(f"\n CALL read_policy{args} -> is_error={result.is_error}")
            print(result.content[0].text[:200])

if __name__ == "__main__":
    asyncio.run(main())