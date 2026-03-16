from langchain_core.tools import BaseTool

class LineReplyMessage(BaseTool):
    name: str = "line_reply_message"
    description: str = "Reply to a LINE user message. Requires reply_token and message text."

    channel_access_token: str

    def _run(self, reply_token: str, message: str) -> str:
        """Reply to a LINE user message."""
        return self.channel_access_token.reply_message(reply_token, message)

    def _arun(self, reply_token: str, message: str) -> str:
        """Reply to a LINE user message."""
        return self.channel_access_token.reply_message(reply_token, message)

class LinePushMessage(BaseTool):
    name: str = "line_push_message"
    description: str = "Push a message to a LINE user. Requires user_id and message text."

    channel_access_token: str

    def _run(self, user_id: str, message: str) -> str:
        """Push a message to a LINE user."""
        return self.channel_access_token.push_message(user_id, message)
