import logging
import ssl
from http.client import HTTPSConnection
from typing import Optional

from kik_unofficial.configuration import env
from kik_unofficial.device_configuration import kik_version_info
from kik_unofficial.utilities.cryptographic_utilities import CryptographicUtils

log = logging.getLogger("kik_unofficial")

LOGIN_HOST = "api.kikprod.net"
LOGIN_PATH = "/mobile.login.v1.MobileLogin/Login"

# mobile.login.v1.Result from Kik 17.23
_RESULTS = {
    0: "OK",
    1: "SERVER_ERROR",
    2: "INVALID",
    3: "SERVICE_UNAVAILABLE",
    11: "NOT_REGISTERED",
    12: "INVALID_PASSWORD",
    13: "ACCT_TERMINATED",
    14: "MISSING_CREDS",
    16: "USER_TEMP_BANNED",
    17: "VERIFICATION_REQUIRED",
}


class MobileLoginError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MobileLoginSuccess:
    def __init__(self, kik_node: str, username: str, email: str):
        self.kik_node = kik_node
        self.username = username
        self.email = email


def login(username: str, password: str, device_id: str, android_id: str, recaptcha_token: Optional[str] = None) -> MobileLoginSuccess:
    """
    Kik 17.x ignores the old XMPP login IQ. Password login is a gRPC call that
    requires a reCAPTCHA Enterprise token from the official Android app.
    """
    token = (recaptcha_token if recaptcha_token is not None else env.get("RECAPTCHA_TOKEN") or "").strip()
    payload = _login_request(username, password, device_id, android_id, token)
    frame = b"\x00" + len(payload).to_bytes(4, "big") + payload
    connection = HTTPSConnection(LOGIN_HOST, 443, timeout=20, context=ssl.create_default_context())
    try:
        connection.request(
            "POST",
            LOGIN_PATH,
            body=frame,
            headers={
                "content-type": "application/grpc",
                "te": "trailers",
                "user-agent": f"Kik/{kik_version_info['kik_version']} (Android 14)",
            },
        )
        response = connection.getresponse()
        body = response.read()
        headers = {key.lower(): value for key, value in response.getheaders()}
    finally:
        connection.close()

    grpc_message = headers.get("grpc-message", "")
    if grpc_message:
        if "recaptcha_token" in grpc_message:
            raise MobileLoginError(
                "Kik requires a reCAPTCHA Enterprise token from the official Kik Android app before it will log this account in. "
                "This script cannot create that token. If you already have one, put it in RECAPTCHA_TOKEN in .env and start again."
            )
        raise MobileLoginError(f"Kik rejected login: {grpc_message}")

    fields = _parse_fields(_grpc_payload(body))
    result = fields.get(1, [(0, 0)])[0][1]
    result_name = _RESULTS.get(result, f"RESULT_{result}")
    if result != 0:
        reason = _first_string(fields, 3) or _first_string(fields, 9)
        detail = f" ({reason})" if reason else ""
        if result_name == "VERIFICATION_REQUIRED":
            raise MobileLoginError(
                "Kik requires extra verification from the official app before this account can log in"
                + detail
            )
        if result_name == "INVALID_PASSWORD":
            raise MobileLoginError("Kik rejected the password.")
        if result_name == "NOT_REGISTERED":
            raise MobileLoginError("Kik could not find that username.")
        if result_name == "ACCT_TERMINATED":
            raise MobileLoginError("Kik says this account is deactivated.")
        raise MobileLoginError(f"Kik login failed: {result_name}{detail}")

    jid = _first_string(fields, 6)
    if not jid:
        raise MobileLoginError("Kik accepted the password but did not return an account id.")
    node = jid.split("@", 1)[0]
    return MobileLoginSuccess(kik_node=node, username=_first_string(fields, 7) or username, email=_first_string(fields, 8) or "")


def _login_request(username: str, password: str, device_id: str, android_id: str, recaptcha_token: str) -> bytes:
    major, minor, bugfix, build = kik_version_info["kik_version"].split(".")
    passkey = CryptographicUtils.key_from_password(username, password)
    username_creds = _field_str(1, username) + _field_str(2, passkey)
    device = _field_varint(1, 2) + _field_str(2, device_id)  # prefix CAN
    locale = _field_str(1, "en_US")
    version = _field_varint(1, int(major)) + _field_varint(2, int(minor)) + _field_varint(3, int(bugfix)) + _field_str(4, build)
    android = (
        _field_str(1, "samsung")
        + _field_str(2, "unknown")
        + _field_str(3, "34")
        + _field_str(4, android_id)
        + _field_str(5, "1")
        + _field_str(6, "0")
        + _field_str(7, "unknown")
        + _field_str(8, "utm_source=google-play&utm_medium=organic")
        + _field_str(9, "14")
    )
    payload = _field_msg(1, username_creds) + _field_msg(3, device) + _field_msg(4, locale) + _field_msg(5, version) + _field_msg(6, android)
    if recaptcha_token:
        payload += _field_str(9, recaptcha_token)
    return payload


def _varint(number: int) -> bytes:
    out = bytearray()
    while True:
        byte = number & 0x7F
        number >>= 7
        if number:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field_str(number: int, value: str) -> bytes:
    raw = value.encode()
    return bytes([number << 3 | 2]) + _varint(len(raw)) + raw


def _field_varint(number: int, value: int) -> bytes:
    return bytes([number << 3 | 0]) + _varint(value)


def _field_msg(number: int, payload: bytes) -> bytes:
    return bytes([number << 3 | 2]) + _varint(len(payload)) + payload


def _grpc_payload(body: bytes) -> bytes:
    if len(body) < 5:
        return b""
    length = int.from_bytes(body[1:5], "big")
    return body[5 : 5 + length]


def _parse_fields(payload: bytes):
    fields = {}
    pos = 0
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _read_varint(payload, pos)
            fields.setdefault(field, []).append((wire, value))
        elif wire == 2:
            length, pos = _read_varint(payload, pos)
            fields.setdefault(field, []).append((wire, payload[pos : pos + length]))
            pos += length
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break
    return fields


def _read_varint(payload: bytes, pos: int):
    value = 0
    shift = 0
    while True:
        byte = payload[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _first_string(fields, number: int) -> str:
    for wire, value in fields.get(number, []):
        if wire == 2:
            try:
                return value.decode()
            except UnicodeDecodeError:
                return ""
    return ""
