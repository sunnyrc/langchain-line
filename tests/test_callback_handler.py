from unittest.mock import MagicMock, patch

from langchain_core.outputs import Generation, LLMResult

from langchain_line.callback_handler import LineCallbackHandler
from tests.conftest import FAKE_CHANNEL_ACCESS_TOKEN, FAKE_REPLY_TOKEN, FAKE_USER_ID


def _make_handler(reply_token=None):
    return LineCallbackHandler(
        channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN,
        user_id=FAKE_USER_ID,
        reply_token=reply_token,
    )


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_llm_end_sends_push_when_no_reply_token(mock_api_client_cls, mock_api_cls):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler()
    result = LLMResult(generations=[[Generation(text="Hello!")]])
    handler.on_llm_end(result)

    mock_api.push_message.assert_called_once()
    mock_api.reply_message.assert_not_called()


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_llm_end_uses_reply_token_first(mock_api_client_cls, mock_api_cls):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler(reply_token=FAKE_REPLY_TOKEN)
    result = LLMResult(generations=[[Generation(text="Hi!")]])
    handler.on_llm_end(result)

    mock_api.reply_message.assert_called_once()
    mock_api.push_message.assert_not_called()


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_llm_end_falls_back_to_push_after_reply_used(
    mock_api_client_cls, mock_api_cls
):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler(reply_token=FAKE_REPLY_TOKEN)

    result1 = LLMResult(generations=[[Generation(text="First")]])
    handler.on_llm_end(result1)
    assert mock_api.reply_message.call_count == 1

    result2 = LLMResult(generations=[[Generation(text="Second")]])
    handler.on_llm_end(result2)
    assert mock_api.push_message.call_count == 1


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_llm_end_skips_empty_text(mock_api_client_cls, mock_api_cls):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler()
    result = LLMResult(generations=[[Generation(text="  ")]])
    handler.on_llm_end(result)

    mock_api.push_message.assert_not_called()


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_chain_end_sends_output(mock_api_client_cls, mock_api_cls):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler()
    handler.on_chain_end({"output": "Final answer"})

    mock_api.push_message.assert_called_once()


@patch("langchain_line.callback_handler.MessagingApi")
@patch("langchain_line.callback_handler.ApiClient")
def test_on_chain_end_ignores_missing_output_key(mock_api_client_cls, mock_api_cls):
    mock_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    handler = _make_handler()
    handler.on_chain_end({"result": "something"})

    mock_api.push_message.assert_not_called()
