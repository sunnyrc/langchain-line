import json
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict


class LineChatMessageHistory(BaseChatMessageHistory):
    """Chat history keyed by LINE user ID, with optional Redis backend.

    By default stores messages in memory. Pass a Redis client to
    persist across restarts.

    Args:
        user_id: LINE user ID to store messages for.
        max_messages: Maximum number of messages to retain. Defaults to 100.
        redis_client: Optional redis.Redis instance. If provided, messages
            are stored in Redis under key ``line:history:{user_id}``.
        ttl: Optional TTL in seconds for the Redis key. Defaults to None (no expiry).

    Example:
        >>> # In-memory
        >>> history = LineChatMessageHistory(user_id="U1234")

        >>> # With Redis
        >>> import redis
        >>> r = redis.Redis(host="localhost", port=6379)
        >>> history = LineChatMessageHistory(user_id="U1234", redis_client=r)
    """

    def __init__(
        self,
        user_id: str,
        max_messages: int = 100,
        redis_client: Optional[object] = None,
        ttl: Optional[int] = None,
    ) -> None:
        self.user_id = user_id
        self.max_messages = max_messages
        self.redis_client = redis_client
        self.ttl = ttl
        self._key = f"line:history:{user_id}"
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        """Return stored messages, capped at max_messages."""
        if self.redis_client is not None:
            return self._read_from_redis()
        return self._messages[-self.max_messages :]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        if self.redis_client is not None:
            self._append_to_redis(message)
        else:
            self._messages.append(message)

    def clear(self) -> None:
        """Clear all stored messages for this user."""
        if self.redis_client is not None:
            self.redis_client.delete(self._key)
        else:
            self._messages = []

    def _read_from_redis(self) -> list[BaseMessage]:
        """Read messages from Redis, returning the last max_messages."""
        data = self.redis_client.lrange(self._key, -self.max_messages, -1)
        return messages_from_dict([json.loads(item) for item in data])

    def _append_to_redis(self, message: BaseMessage) -> None:
        """Append a message to the Redis list and apply TTL."""
        self.redis_client.rpush(self._key, json.dumps(message_to_dict(message)))
        if self.ttl is not None:
            self.redis_client.expire(self._key, self.ttl)
