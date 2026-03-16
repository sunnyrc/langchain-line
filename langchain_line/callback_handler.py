from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)


class LineCallbackHandler(BaseCallbackHandler):
    """Callback that sends LLM output directly to a LINE user.

    Attach this to a chain or agent to automatically push
    responses to LINE without manual tool calling.

    If a reply_token is provided, the first message uses reply
    (free); subsequent messages fall back to push.

    Args:
        channel_access_token: LINE channel access token.
        user_id: Target LINE user ID for push messages.
        reply_token: Optional reply token (preferred over push for replies).

    Example:
        >>> handler = LineCallbackHandler(
        ...     channel_access_token="token",
        ...     user_id="U1234",
        ...     reply_token="reply-token",
        ... )
        >>> llm.invoke("Hello", config={"callbacks": [handler]})
    """

    def __init__(
        self,
        channel_access_token: str,
        user_id: str,
        reply_token: Optional[str] = None,
    ) -> None:
        self.channel_access_token = channel_access_token
        self.user_id = user_id
        self.reply_token = reply_token
        self._reply_used = False

    def _send_message(self, text: str) -> None:
        """Send a message via reply (if token available) or push.

        Args:
            text: The text message to send.
        """
        configuration = Configuration(access_token=self.channel_access_token)
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            if self.reply_token and not self._reply_used:
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=self.reply_token,
                        messages=[TextMessage(text=text)],
                    )
                )
                self._reply_used = True
            else:
                api.push_message(
                    PushMessageRequest(
                        to=self.user_id,
                        messages=[TextMessage(text=text)],
                    )
                )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Send LLM output to LINE when the LLM finishes.

        Args:
            response: The LLM result containing generated text.
        """
        for generation_list in response.generations:
            for generation in generation_list:
                text = generation.text.strip()
                if text:
                    self._send_message(text)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Send final chain output to LINE.

        Only sends if the output dict has an 'output' key (standard
        AgentExecutor output format).

        Args:
            outputs: The chain output dictionary.
        """
        if isinstance(outputs, dict) and "output" in outputs:
            text = str(outputs["output"]).strip()
            if text:
                self._send_message(text)
