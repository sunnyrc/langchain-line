"""RAG agent that replies via LINE with conversation memory.

This example shows how to:
1. Use LineChatMessageHistory for per-user conversation memory
2. Use LineCallbackHandler to auto-forward LLM responses to LINE
3. Build a RAG chain that answers questions from a document

Requirements:
    pip install langchain-line langchain-openai langchain-community
    pip install faiss-cpu fastapi uvicorn

Environment variables:
    LINE_CHANNEL_ACCESS_TOKEN - Your LINE channel access token
    LINE_CHANNEL_SECRET - Your LINE channel secret
    OPENAI_API_KEY - Your OpenAI API key

Usage:
    uvicorn examples.rag_line_bot:app --port 8000
"""

import os

from fastapi import FastAPI, Header, HTTPException, Request
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_line import (
    LineCallbackHandler,
    LineChatMessageHistory,
    LineWebhookParser,
)

app = FastAPI()

channel_access_token = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
channel_secret = os.environ["LINE_CHANNEL_SECRET"]

parser = LineWebhookParser(channel_secret=channel_secret)

embeddings = OpenAIEmbeddings()

sample_docs = [
    "Our store is open Monday to Friday, 9am to 6pm.",
    "We offer free shipping on orders over 1000 THB.",
    "Returns are accepted within 30 days of purchase.",
    "Contact support at support@example.com.",
]
vectorstore = FAISS.from_texts(sample_docs, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-5.3-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support bot. "
            "Answer questions using the provided context. "
            "If you don't know, say so.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
chain = prompt | llm

_histories: dict[str, LineChatMessageHistory] = {}


def get_history(user_id: str) -> LineChatMessageHistory:
    if user_id not in _histories:
        _histories[user_id] = LineChatMessageHistory(user_id=user_id)
    return _histories[user_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


@app.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(...)):
    body = await request.body()

    try:
        events = parser.parse_with_metadata(body.decode("utf-8"), x_line_signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        user_id = event["user_id"]
        user_message = event["message"].content
        reply_token = event["reply_token"]

        docs = retriever.invoke(user_message)
        context = "\n".join(doc.page_content for doc in docs)

        line_handler = LineCallbackHandler(
            channel_access_token=channel_access_token,
            user_id=user_id,
            reply_token=reply_token,
        )

        chain_with_history.invoke(
            {"input": user_message, "context": context},
            config={
                "configurable": {"session_id": user_id},
                "callbacks": [line_handler],
            },
        )

    return "OK"
