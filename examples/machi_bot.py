#!/usr/bin/env python3
"""
Kik account that answers group questions with ChatGPT.

It only replies in group chats, and only when the message starts with @machi.
Direct messages and other group chatter are ignored.

The default model is gpt-5-nano, the cheapest OpenAI chat model. gpt-4o-mini
(the 2024 "mini" model) still works if you set OPENAI_MODEL=gpt-4o-mini, but
it costs more.

Setup:
  1. Copy .env.example to .env and set BOT_USERNAME, BOT_PASSWORD, and OPENAI_API_KEY.
     Optional: DEVICE_ID, ANDROID_ID, BOT_NODE_JID (same device ids keep Kik from
     treating every launch as a new phone).
  2. Or put Kik login details in examples/creds.yaml (see examples/echo_bot.py).
  3. From the repo root: python3 examples/machi_bot.py
"""

import os
import re
import threading
from typing import Optional

import requests
import yaml

import kik_unofficial.datatypes.xmpp.chatting as chatting
from kik_unofficial.callbacks import KikClientCallback
from kik_unofficial.client import KikClient
from kik_unofficial.configuration import env
from kik_unofficial.datatypes.xmpp.errors import LoginError, SignUpError
from kik_unofficial.datatypes.xmpp.login import ConnectionFailedResponse, LoginResponse

# gpt-5-nano: $0.05 / 1M input, $0.40 / 1M output.
# gpt-4o-mini (what "chatgpt-mini-4o-2024" refers to) is $0.15 / $0.60.
DEFAULT_MODEL = "gpt-5-nano"
MENTION_RE = re.compile(r"^\s*@machi\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MAX_SENTENCES = 5
MAX_QUESTION_CHARS = 2000
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are Machi, texting in a Kik group chat like a regular person. "
    "Reply only in Tanglish: Tamil spoken in chat, written with English letters. "
    "Examples of the voice to use: \"enna da?\", \"theriyala\", \"seri, naalu maniku varuven\", \"adhu epdi nu kekuriya?\". "
    "The question may be in English or Tanglish. The reply is still Tanglish. "
    "Do not write English sentences. Do not use Tamil script. "
    "Answer the question directly. Most replies should be 1 or 2 sentences. "
    "Never use more than 5 sentences. "
    "Sound casual and plain. Do not say you are an AI, a bot, or ChatGPT. "
    "Do not use markdown, bullet lists, headings, or hashtags. "
    "If you are not sure, say so in one short Tanglish sentence."
)


def question_after_mention(body: str) -> Optional[str]:
    """Return the question after a leading @machi, or None if the message is not a mention."""
    if not body or not MENTION_RE.match(body):
        return None
    question = MENTION_RE.sub("", body, count=1)
    return question.lstrip(" \t,:;-").strip()


def limit_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Keep at most max_sentences so a long model reply never hits the group."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    parts = [part.strip() for part in SENTENCE_RE.split(cleaned) if part.strip()]
    if len(parts) <= max_sentences:
        return cleaned
    return " ".join(parts[:max_sentences])


def _is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4")


def ask_chatgpt(question: str, api_key: str, model: str) -> str:
    """Call the ChatGPT API and return a short plain-text answer."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    # Reasoning models bill hidden reasoning tokens as output and reject temperature.
    # "minimal" keeps those extra tokens near zero so a short answer stays cheap.
    if _is_reasoning_model(model):
        payload["reasoning_effort"] = "minimal"
        payload["max_completion_tokens"] = 400
    else:
        payload["temperature"] = 0.7
        payload["max_tokens"] = 200

    response = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"ChatGPT API returned {response.status_code}: {detail}")

    message = response.json()["choices"][0]["message"]
    return limit_sentences(_message_text(message))


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(part.get("text") or "")
        return "".join(chunks).strip()
    return ""


# Values shipped in .env.example and examples/creds.yaml. They are not a real account.
_EXAMPLE_VALUES = {
    "username": {"my.username", "bot_username"},
    "password": {"mYpAsSw0rD", "bot_password"},
    "node": {"my.username_pqy", "bot_node_jid"},
    "device_id": {"dddddddddddddddddddddddddddddddd"},
    "android_id": {"aaaaaaaaaaaaaaaa"},
}


def _real_cred(key: str, value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in _EXAMPLE_VALUES.get(key, ()):
        return None
    return text


def _load_kik_creds() -> dict:
    """Prefer .env, then examples/creds.yaml. Ignore the sample placeholder values."""
    creds = {
        "username": _real_cred("username", env.get("BOT_USERNAME")),
        "password": _real_cred("password", env.get("BOT_PASSWORD")),
        "node": _real_cred("node", env.get("BOT_NODE_JID")),
        "device_id": _real_cred("device_id", env.get("DEVICE_ID")),
        "android_id": _real_cred("android_id", env.get("ANDROID_ID")),
    }
    for creds_file in ("creds.yaml", os.path.join("examples", "creds.yaml")):
        if not os.path.isfile(creds_file):
            continue
        with open(creds_file) as handle:
            file_creds = yaml.safe_load(handle) or {}
        for key in ("username", "password", "node", "device_id", "android_id"):
            if not creds.get(key):
                creds[key] = _real_cred(key, file_creds.get(key))
        break
    if not creds.get("username") or not creds.get("password"):
        raise SystemExit("Set BOT_USERNAME and BOT_PASSWORD in .env, or fill in examples/creds.yaml.")
    return creds


class MachiBot(KikClientCallback):
    def __init__(self, creds: dict, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._reply_lock = threading.Lock()
        username = creds["username"]
        password = str(creds["password"])
        self.client = KikClient(
            self,
            username,
            password,
            creds.get("node"),
            device_id=creds.get("device_id"),
            android_id=creds.get("android_id"),
            enable_console_logging=True,
        )
        if not creds.get("node"):
            self.client.log.info("No Kik node yet. This login uses your username and password.")

    def on_authenticated(self):
        self.client.log.info("Logged in. Listening for group messages that start with @machi.")
        self.client.request_roster()

    def on_login_ended(self, response: LoginResponse):
        self.client.log.info(f"Full name: {response.first_name} {response.last_name}")
        self.client.log.info(f"Kik node is {response.kik_node}. Save that as BOT_NODE_JID in .env.")

    def on_chat_message_received(self, chat_message: chatting.IncomingChatMessage):
        self.client.log.info(f"Ignoring direct message from {chat_message.from_jid}")

    def on_group_message_received(self, chat_message: chatting.IncomingGroupChatMessage):
        body = chat_message.body or ""
        if self._is_from_self(chat_message.from_jid):
            return

        question = question_after_mention(body)
        if question is None:
            return

        self.client.log.info(f"Mention in {chat_message.group_jid} from {chat_message.from_jid}: {body}")
        self.client.send_read_receipt(chat_message.from_jid, chat_message.message_id, chat_message.group_jid)

        if not question:
            self.client.send_chat_message(chat_message.group_jid, "enna da?")
            return
        if len(question) > MAX_QUESTION_CHARS:
            self.client.send_chat_message(chat_message.group_jid, "romba periya message da. konjam short ah kekkala?")
            return

        with self._reply_lock:
            self.client.send_is_typing(chat_message.group_jid, True)
            try:
                answer = ask_chatgpt(question, self.api_key, self.model)
            except Exception as exc:
                self.client.log.error(f"ChatGPT request failed: {exc}")
                answer = ""
            finally:
                self.client.send_is_typing(chat_message.group_jid, False)

            if not answer:
                answer = "sorry da, puriyala. innoru thadava kekkala?"
            self.client.send_chat_message(chat_message.group_jid, answer)

    def on_connection_failed(self, response: ConnectionFailedResponse):
        self.client.log.error(f"Connection failed: {response.message}")

    def on_login_error(self, login_error: LoginError):
        self.client.log.error(f"Login error: {login_error}")
        if login_error.is_captcha():
            login_error.solve_captcha_wizard(self.client)

    def on_register_error(self, response: SignUpError):
        self.client.log.error(f"Register error: {response.message}")

    def _is_from_self(self, from_jid: str) -> bool:
        local = from_jid.split("@", 1)[0]
        username = local.rsplit("_", 1)[0] if "_" in local else local
        return username.lower() == self.client.username.lower()


def main():
    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env (or the environment) before starting the bot.")

    model = (env.get("OPENAI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    creds = _load_kik_creds()
    bot = MachiBot(creds, api_key, model)
    bot.client.wait_for_messages()


if __name__ == "__main__":
    main()
