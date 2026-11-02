"""Signed Kalshi Trade API client.

Credentials come from environment variables, never from files in this repo:
    KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH            (production)
    KALSHI_DEMO_KEY_ID, KALSHI_DEMO_PRIVATE_KEY_PATH  (demo)
On Windows, values saved for the user but not yet visible to this process
(no restart since setting them) are read from HKCU\\Environment.

Each request is signed over timestamp_ms + METHOD + path-without-query:
Ed25519 keys sign it directly, RSA keys use RSA-PSS/SHA-256.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

BASE_URLS = {
    "prod": "https://external-api.kalshi.com/trade-api/v2",
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
}
ENV_NAMES = {
    "prod": ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"),
    "demo": ("KALSHI_DEMO_KEY_ID", "KALSHI_DEMO_PRIVATE_KEY_PATH"),
}


class KalshiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status, self.body = status, body


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value or os.name != "nt":
        return value
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


class KalshiClient:
    def __init__(self, env: str = "prod", timeout: float = 20.0):
        if env not in BASE_URLS:
            raise ValueError(f"env must be one of {list(BASE_URLS)}")
        id_var, path_var = ENV_NAMES[env]
        self.key_id, key_path = _env(id_var), _env(path_var)
        if not self.key_id or not key_path:
            raise RuntimeError(f"set {id_var} and {path_var} first")
        with open(key_path, "rb") as f:
            self._key = serialization.load_pem_private_key(f.read(), password=None)
        self.env, self.base, self.timeout = env, BASE_URLS[env], timeout
        self._root = urllib.parse.urlparse(self.base).path  # /trade-api/v2

    def _sign(self, msg: bytes) -> str:
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            sig = self._key.sign(msg)
        elif isinstance(self._key, rsa.RSAPrivateKey):
            sig = self._key.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        else:
            raise TypeError(f"unsupported key type {type(self._key).__name__}")
        return base64.b64encode(sig).decode()

    def request(self, method: str, path: str, params: dict | None = None,
                body: dict | None = None, retries: int = 5) -> dict:
        """Signed request; backs off and retries on 429 (Kalshi sends no Retry-After)."""
        for attempt in range(retries + 1):
            try:
                return self._request_once(method, path, params, body)
            except KalshiError as exc:
                if exc.status != 429 or attempt == retries:
                    raise
                time.sleep(0.25 * 2 ** attempt)
        raise AssertionError("unreachable")

    def _request_once(self, method: str, path: str, params: dict | None,
                      body: dict | None) -> dict:
        method = method.upper()
        ts = str(int(time.time() * 1000))
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(f"{ts}{method}{self._root}{path}".encode()),
            "Accept": "application/json",
        }
        url = self.base + path + (f"?{urllib.parse.urlencode(params)}" if params else "")
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise KalshiError(exc.code, exc.read().decode(errors="replace")) from None
        return json.loads(raw) if raw else {}

    def get(self, path: str, **params) -> dict:
        return self.request("GET", path, params or None)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Read-only auth check: GET /account/limits")
    ap.add_argument("--env", default="prod", choices=list(BASE_URLS))
    print(json.dumps(KalshiClient(ap.parse_args().env).get("/account/limits"), indent=2))
