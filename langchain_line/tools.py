from langchain_core.tools import BaseTool
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)


def _format_line_api_error(e: Exception) -> str:
    """Format a LINE API exception into a readable error string."""
    status = getattr(e, "status", None)
    reason = getattr(e, "reason", None)
    body = getattr(e, "body", None)
    parts: list[str] = []
    if status is not None:
        parts.append(f"status={status}")
    if reason:
        parts.append(f"reason={reason}")
    if body:
        parts.append(f"body={body}")
    details = ("; ".join(parts)) if parts else str(e)
    return f"LINE API request failed ({details})"


def _safe_line_call(fn, *args, **kwargs) -> str:
    """Call a LINE API function and catch exceptions gracefully."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return _format_line_api_error(e)


class LineReplyMessage(BaseTool):
    """Reply to a LINE message using a reply token."""

    name: str = "line_reply_message"
    description: str = (
        "Reply to a LINE user message. Requires reply_token and message text."
    )
    channel_access_token: str

    def _run(self, reply_token: str, message: str) -> str:
        """Reply to a LINE user message."""
        configuration = Configuration(access_token=self.channel_access_token)

        def _call() -> str:
            with ApiClient(configuration) as api_client:
                api = MessagingApi(api_client)
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=[TextMessage(text=message)],
                    )
                )
            return "Successfully sent reply message."

        return _safe_line_call(_call)


class LinePushMessage(BaseTool):
    """Send a push message to a LINE user at any time."""

    name: str = "line_push_message"
    description: str = (
        "Send a message to a LINE user by user ID. "
        "Works anytime, not just as a reply."
    )
    channel_access_token: str

    def _run(self, user_id: str, message: str) -> str:
        """Push a message to a LINE user."""
        configuration = Configuration(access_token=self.channel_access_token)

        def _call() -> str:
            with ApiClient(configuration) as api_client:
                api = MessagingApi(api_client)
                api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=message)],
                    )
                )
            return "Successfully sent push message."

        return _safe_line_call(_call)


class LineGetProfile(BaseTool):
    """Get a LINE user's display name and profile picture URL."""

    name: str = "line_get_profile"
    description: str = (
        "Get a LINE user's profile (display name, picture URL, status message)."
    )
    channel_access_token: str

    def _run(self, user_id: str) -> str:
        """Get a LINE user's profile."""
        configuration = Configuration(access_token=self.channel_access_token)

        def _call() -> str:
            with ApiClient(configuration) as api_client:
                api = MessagingApi(api_client)
                profile = api.get_profile(user_id)
            return (
                f"Display name: {profile.display_name}, "
                f"Picture URL: {profile.picture_url}, "
                f"Status message: {profile.status_message}"
            )

        return _safe_line_call(_call)
