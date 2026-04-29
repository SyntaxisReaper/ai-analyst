import uuid
import threading
import json
import os
import redis
from redis.exceptions import RedisError
from datetime import datetime, timedelta


class SessionManager:
    def __init__(self, max_sessions=50, expire_hours=2, persist: bool = True, redis_url: str = None):
        self._sessions = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.expire_hours = expire_hours

        # Persistence via Redis (optional). If persist is True we attempt to
        # connect to the provided redis_url (or localhost) and store sessions
        # as JSON strings under keys `session:{id}` with an expiry matching
        # `expire_hours` so sessions survive backend restarts.
        self._persist = persist
        self._redis = None
        if self._persist:
            try:
                url = redis_url or os.getenv("SESSION_REDIS_URL", "redis://localhost:6379/0")
                self._redis = redis.Redis.from_url(url, decode_responses=True)
                # quick ping to validate connection
                self._redis.ping()
                self._load_from_redis()
            except RedisError:
                # If Redis cannot be used, fall back to in-memory only
                self._redis = None
                self._persist = False

        self._start_cleanup()

    def _load_from_redis(self):
        try:
            for key in self._redis.scan_iter(match="session:*"):
                sid = key.split(":", 1)[1]
                try:
                    data = self._redis.get(key)
                    if data:
                        self._sessions[sid] = json.loads(data)
                except Exception:
                    continue
        except RedisError:
            pass

    def _save_session_redis(self, sid: str):
        if not self._persist or not self._redis:
            return
        try:
            data = json.dumps(self._sessions[sid])
            key = f"session:{sid}"
            self._redis.set(key, data)
            # set expiry to expire_hours
            self._redis.expire(key, int(self.expire_hours * 3600))
        except RedisError:
            pass

    def _delete_session_redis(self, sid: str):
        if not self._persist or not self._redis:
            return
        try:
            self._redis.delete(f"session:{sid}")
        except RedisError:
            pass

    def create(self) -> str:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                self._cleanup_expired()
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions, key=lambda k: self._sessions[k]["last_used"])
                    del self._sessions[oldest]
                    self._delete_session_redis(oldest)
            sid = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            self._sessions[sid] = {
                "file_name": None,
                "summary": None,
                "metadata": {},
                "history": [],
                "created_at": now,
                "last_used": now,
            }
            if self._persist:
                self._save_session_redis(sid)
            return sid

    def exists(self, sid: str) -> bool:
        with self._lock:
            return sid in self._sessions

    def get(self, sid: str) -> dict:
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s["last_used"] = datetime.utcnow().isoformat()
                if self._persist:
                    self._save_session_redis(sid)
            return s

    def set_file(self, sid: str, data: dict):
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s.update(data)
                s["history"] = []
                s["last_used"] = datetime.utcnow().isoformat()
                if self._persist:
                    self._save_session_redis(sid)

    def add_message(self, sid: str, role: str, content: str, max_turns: int = 40):
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s["history"].append({"role": role, "content": content})
                if len(s["history"]) > max_turns:
                    s["history"] = s["history"][-max_turns:]
                s["last_used"] = datetime.utcnow().isoformat()
                if self._persist:
                    self._save_session_redis(sid)

    def clear_history(self, sid: str):
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s["history"] = []
                if self._persist:
                    self._save_session_db(sid)

    def delete(self, sid: str):
        with self._lock:
            self._sessions.pop(sid, None)
            if self._persist:
                self._delete_session_redis(sid)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _cleanup_expired(self):
        now = datetime.utcnow()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - datetime.fromisoformat(s["last_used"])) > timedelta(hours=self.expire_hours)
        ]
        for sid in expired:
            del self._sessions[sid]

    def _start_cleanup(self):
        def loop():
            import time
            while True:
                time.sleep(3600)
                with self._lock:
                    self._cleanup_expired()
        threading.Thread(target=loop, daemon=True).start()
