import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, message_to_dict

from langchain_line.chat_message_history import LineChatMessageHistory


def test_add_and_retrieve_messages():
    history = LineChatMessageHistory(user_id="U1234")
    history.add_message(HumanMessage(content="Hello"))
    history.add_message(AIMessage(content="Hi there!"))

    assert len(history.messages) == 2
    assert history.messages[0].content == "Hello"
    assert history.messages[1].content == "Hi there!"


def test_clear_messages():
    history = LineChatMessageHistory(user_id="U1234")
    history.add_message(HumanMessage(content="Hello"))
    history.clear()

    assert history.messages == []


def test_max_messages_caps_output():
    history = LineChatMessageHistory(user_id="U1234", max_messages=3)
    for i in range(5):
        history.add_message(HumanMessage(content=f"msg {i}"))

    assert len(history.messages) == 3
    assert history.messages[0].content == "msg 2"
    assert history.messages[2].content == "msg 4"


def test_max_messages_default():
    history = LineChatMessageHistory(user_id="U1234")
    assert history.max_messages == 100


def test_separate_users():
    h1 = LineChatMessageHistory(user_id="U1111")
    h2 = LineChatMessageHistory(user_id="U2222")
    h1.add_message(HumanMessage(content="from user 1"))
    h2.add_message(HumanMessage(content="from user 2"))

    assert len(h1.messages) == 1
    assert len(h2.messages) == 1
    assert h1.messages[0].content == "from user 1"
    assert h2.messages[0].content == "from user 2"


# --- Redis backend tests ---


def _make_mock_redis():
    """Create a mock Redis client backed by a real list."""
    store: dict[str, list[bytes]] = {}
    mock = MagicMock()

    def rpush(key, value):
        store.setdefault(key, []).append(value.encode() if isinstance(value, str) else value)

    def lrange(key, start, end):
        items = store.get(key, [])
        # Emulate Redis lrange with negative indices
        return items[start:] if end == -1 else items[start : end + 1]

    def delete(key):
        store.pop(key, None)

    def expire(key, ttl):
        pass  # no-op for tests

    mock.rpush = MagicMock(side_effect=rpush)
    mock.lrange = MagicMock(side_effect=lrange)
    mock.delete = MagicMock(side_effect=delete)
    mock.expire = MagicMock(side_effect=expire)
    mock._store = store
    return mock


def test_redis_add_and_retrieve():
    r = _make_mock_redis()
    history = LineChatMessageHistory(user_id="U1234", redis_client=r)
    history.add_message(HumanMessage(content="Hello"))
    history.add_message(AIMessage(content="Hi!"))

    messages = history.messages
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi!"


def test_redis_clear():
    r = _make_mock_redis()
    history = LineChatMessageHistory(user_id="U1234", redis_client=r)
    history.add_message(HumanMessage(content="Hello"))
    history.clear()

    assert history.messages == []
    r.delete.assert_called_with("line:history:U1234")


def test_redis_max_messages():
    r = _make_mock_redis()
    history = LineChatMessageHistory(user_id="U1234", redis_client=r, max_messages=2)
    for i in range(4):
        history.add_message(HumanMessage(content=f"msg {i}"))

    messages = history.messages
    assert len(messages) == 2
    assert messages[0].content == "msg 2"
    assert messages[1].content == "msg 3"


def test_redis_ttl_is_set():
    r = _make_mock_redis()
    history = LineChatMessageHistory(user_id="U1234", redis_client=r, ttl=3600)
    history.add_message(HumanMessage(content="Hello"))

    r.expire.assert_called_with("line:history:U1234", 3600)


def test_redis_ttl_not_set_when_none():
    r = _make_mock_redis()
    history = LineChatMessageHistory(user_id="U1234", redis_client=r)
    history.add_message(HumanMessage(content="Hello"))

    r.expire.assert_not_called()
