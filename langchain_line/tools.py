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

class LineGetProfile(BaseTool):
    name: str = "line_get_profile"
    description: str = "Get a LINE user's profile. Requires user_id."

    channel_access_token: str

    def _run(self, user_id: str) -> str:
        """Get a LINE user's profile."""
        return self.channel_access_token.get_profile(user_id)

class LineGetGroupMembers(BaseTool):
    name: str = "line_get_group_members"
    description: str = "Get a list of LINE group members. Requires group_id."

    channel_access_token: str

    def _run(self, group_id: str) -> str:
        """Get a list of LINE group members."""
        return self.channel_access_token.get_group_members(group_id)

class LineBroadcastMessage(BaseTool):
    name: str = "line_broadcast_message"
    description: str = "Send a message to all followers of the LINE bot. Requires message text."

    channel_access_token: str

    def _run(self, message: str) -> str:
        """Broadcast a message to all LINE users."""
        return self.channel_access_token.broadcast_message(message)