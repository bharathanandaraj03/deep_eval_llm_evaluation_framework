import asyncio
import sys
import anthropic
from mcp import Client, StdioServerParameters
from app.chatbot import config
from dotenv import load_dotenv

SERVER = StdioServerParameters(command=sys.executable,args=["-m","app.mcp_server.server"])

SYSTEM_PROMPT = (
    "You are an HR assitant for Acme Corp. Use the tools to answer policy questions and manage "
    "employee leave. Ask the user to confirm before cancelling a leave request. If the tools "
    "cannot answer, say you don't know and refer the user to HR. "
)

DECLINE_REPLY = "I can't help with that request. Please contact HR"
MAX_TURNS = 10

def to_claude_tool(tool) -> dict:
    return {"name":tool.name, "description":tool.description or "", "input_schema": tool.input_schema }

async def run_mcp_agent(user_input: str, server= SERVER, claude=None) -> dict:
    claude = claude or anthropic.AsyncAnthropic()
    tools_called = []

    async with Client(server) as mcp_client:
        tools = [to_claude_tool(t) for t in (await mcp_client.list_tools()).tools]
        messages = [{"role":"user","content":user_input}]

        for _ in range(MAX_TURNS):
            response = await claude.beta.messages.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools,
                cache_control={"type": "ephemeral"},
                output_config={"effort": config.EFFORT},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = await mcp_client.call_tool(block.name, block.input)
                text = "\n".join(c.text for c in result.content if c.type == "text")
                tools_called.append({"name": block.name, "input": block.input, "output": text, "is_error": result.is_error,"result":result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": result.is_error})
            messages.append({"role": "user", "content": tool_results})

    if response.stop_reason == "refusal":
        output = DECLINE_REPLY
    else:
        output = "".join(b.text for b in response.content if b.type == "text")

    return {"input": user_input, "output": output, "tools_called": tools_called, "available_tools": tools}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    result = asyncio.run(run_mcp_agent(" ".join(sys.argv[1:]) or "How many annual leave days does E200 have left?"))
    for call in result["tools_called"]:
        print(f"TOOL {call['name']}({call['input']}) -> is_error={call['is_error']}\n  {call['output']}")
    print(f"\nANSWER: {result['output']}")

