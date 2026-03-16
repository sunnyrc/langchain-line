import json
from typing import Any
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from linebot.v3.webhook import WebhookParser as LineWebhookParserSDK
from linebot.v3.webhooks import MessageEvent, TextMessageContent


class LineWebhookParser:
    """Parse LINE webhook payloads into LangChain messages.

    Converts incoming LINE MessageEvent objects into HumanMessage
    instances that can be fed into a LangChain agent or LangGraph workflow.

    Args:
        channel_secret: LINE channel secret for signature validation.
    """

    def __init__(self, channel_secret: str) -> None:
        self.parser = LineWebhookParserSDK(channel_secret)

    def parse(self, body: str, signature: str) -> list[HumanMessage]:
        """Parse webhook body into a list of HumanMessages."""
        parsed = self.parse_with_metadata(body, signature)
        return [item["message"] for item in parsed]

    def parse_with_metadata(self, body: str, signature: str) -> list[dict[str, Any]]:
        """Parse webhook body and include LINE metadata."""
        events = self.parser.parse(body, signature)
        results: list[dict[str, Any]] = []

        for event in events:
            if not isinstance(event, MessageEvent):
                continue
            if not isinstance(event.message, TextMessageContent):
                continue

            user_id = event.source.user_id
            reply_token = event.reply_token
            timestamp = event.timestamp
            source_type = event.source.type

            human_message = HumanMessage(
                content=event.message.text,
                additional_kwargs={
                    "user_id": user_id,
                    "reply_token": reply_token,
                    "timestamp": timestamp,
                    "source_type": source_type,
                },
            )

            results.append(
                {
                    "message": human_message,
                    "user_id": user_id,
                    "reply_token": reply_token,
                    "timestamp": timestamp,
                    "source_type": source_type,
                }
            )

        return results
