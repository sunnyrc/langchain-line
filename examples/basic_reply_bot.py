"""
This example shows how to:
1. Parse incoming LINE webhook events into LangChain messages
2. Use a LangChain agent with LINE tools to reply

Requirements:
    pip install langchain-line langchain-openai fastapi uvicorn

Environment variables:
    LINE_CHANNEL_ACCESS_TOKEN - Your LINE channel access token
    LINE_CHANNEL_SECRET - Your LINE channel secret
    OPENAI_API_KEY - Your OpenAI API key

Usage:
    uvicorn examples.basic_reply_bot:app --port 8000
"""

import os

from fastapi import FastAPI, Header, HTTPException, Request
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_line import LineReplyMessage, LineWebhookParser

app = FastAPI()

parser = LineWebhookParser(channel_secret=os.environ["LINE_CHANNEL_SECRET"])
reply_tool = LineReplyMessage(
    channel_access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
)

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful LINE chatbot. When a user sends you a message, "
            "reply using the line_reply_message tool. Keep replies concise.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, [reply_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[reply_tool])


@app.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(...)):
    body = await request.body()

    try:
        events = parser.parse_with_metadata(body.decode("utf-8"), x_line_signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        user_message = event["message"].content
        reply_token = event["reply_token"]
        agent_executor.invoke(
            {
                "input": (
                    f"The user said: '{user_message}'. "
                    f"Reply using reply_token='{reply_token}'."
                )
            }
        )

    return "OK"
