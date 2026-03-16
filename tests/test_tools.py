from unittest.mock import MagicMock, patch

from linebot.v3.messaging import (
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)

from langchain_line.tools import (
    LineGetProfile,
    LinePushMessage,
    LineReplyMessage,
)
from tests.conftest import FAKE_CHANNEL_ACCESS_TOKEN, FAKE_REPLY_TOKEN, FAKE_USER_ID


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_reply_message_success(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api

    tool = LineReplyMessage(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"reply_token": FAKE_REPLY_TOKEN, "message": "Hello!"})

    assert "success" in result.lower()
    mock_api.reply_message.assert_called_once()
    call_args = mock_api.reply_message.call_args[0][0]
    assert isinstance(call_args, ReplyMessageRequest)
    assert call_args.reply_token == FAKE_REPLY_TOKEN
    assert call_args.messages[0].text == "Hello!"


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_reply_message_api_error(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api

    err = Exception("forbidden")
    err.status = 403
    err.reason = "Forbidden"
    err.body = None
    mock_api.reply_message.side_effect = err

    tool = LineReplyMessage(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"reply_token": FAKE_REPLY_TOKEN, "message": "Hi"})

    assert "failed" in result.lower()
    assert "403" in result


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_push_message_success(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api

    tool = LinePushMessage(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"user_id": FAKE_USER_ID, "message": "Hello!"})

    assert "success" in result.lower()
    mock_api.push_message.assert_called_once()
    call_args = mock_api.push_message.call_args[0][0]
    assert isinstance(call_args, PushMessageRequest)
    assert call_args.to == FAKE_USER_ID
    assert call_args.messages[0].text == "Hello!"


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_push_message_api_error(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api
    mock_api.push_message.side_effect = Exception("network error")

    tool = LinePushMessage(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"user_id": FAKE_USER_ID, "message": "Hi"})

    assert "failed" in result.lower()


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_get_profile_success(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api

    mock_profile = MagicMock()
    mock_profile.display_name = "Test User"
    mock_profile.picture_url = "https://example.com/pic.jpg"
    mock_profile.status_message = "Hello world"
    mock_api.get_profile.return_value = mock_profile

    tool = LineGetProfile(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"user_id": FAKE_USER_ID})

    assert "Test User" in result
    assert "https://example.com/pic.jpg" in result
    assert "Hello world" in result
    mock_api.get_profile.assert_called_once_with(FAKE_USER_ID)


@patch("langchain_line.tools.MessagingApi")
@patch("langchain_line.tools.ApiClient")
def test_get_profile_api_error(mock_api_client_cls, mock_messaging_api_cls):
    mock_api_client = MagicMock()
    mock_api_client_cls.return_value.__enter__ = MagicMock(return_value=mock_api_client)
    mock_api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_api = MagicMock()
    mock_messaging_api_cls.return_value = mock_api
    mock_api.get_profile.side_effect = Exception("not found")

    tool = LineGetProfile(channel_access_token=FAKE_CHANNEL_ACCESS_TOKEN)
    result = tool.invoke({"user_id": FAKE_USER_ID})

    assert "failed" in result.lower()
