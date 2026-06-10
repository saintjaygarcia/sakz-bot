"""sakz_execution.py - Trading execution layer (PHASE 1: key vault only).

Phase 1 scope ONLY:
  * Per-user Bybit API key encryption at rest (Fernet = AES-128-CBC + HMAC).
  * Bybit V5 signed-request helper (testnet-aware).
  * Connection test via the PRIVATE wallet-balance endpoint.

NO order placement lives here yet - that arrives in Phase 2, once the vault and
connection flow are proven on TESTNET. API keys are expected to be created with
the "Trade" permission ONLY (never "Withdraw"), ideally IP-restricted to the
deployment egress IP.

Env vars:
  SAKZ_VAULT_KEY   urlsafe-base64 Fernet master key (REQUIRED to enable the
                   vault). Generate once with generate_vault_key() and store it
                   as a Railway secret - NEVER commit it.
  BYBIT_TESTNET    "true" (default) routes every signed call to the Bybit
                   testnet. Set "false" ONLY after full testnet validation.
"""
import os
import time
import hmac
import json
import hashlib
import logging
import requests

logger = logging.getLogger(__name__)

try:
    from config import HTTP_TIMEOUT
except Exception:
    HTTP_TIMEOUT = 15

# -- Testnet switch -----------------------------------------------------------
# Default TESTNET=ON during the build phase. Flip BYBIT_TESTNET=false ONLY after
# the full live-trading loop has been validated end-to-end on testnet.
BYBIT_TESTNET = os.environ.get("BYBIT_TESTNET", "true").strip().lower() in ("1", "true", "yes", "on")
_BASE_TESTNET = "https://api-testnet.bybit.com"
_BASE_PROD    = "https://api.bybit.com"
BYBIT_BASE    = _BASE_TESTNET if BYBIT_TESTNET else _BASE_PROD
RECV_WINDOW   = "5000"


def _base_url(testnet=None):
    if testnet is None:
        return BYBIT_BASE
    return _BASE_TESTNET if testnet else _BASE_PROD


# -- Encryption (keys at rest) ------------------------------------------------
_VAULT_KEY = os.environ.get("SAKZ_VAULT_KEY", "").strip()
try:
    from cryptography.fernet import Fernet, InvalidToken
    _fernet = Fernet(_VAULT_KEY.encode()) if _VAULT_KEY else None
    if _fernet:
        logger.info("[sakz_execution] Vault ready (encryption enabled) \u2705")
    else:
        logger.warning("[sakz_execution] SAKZ_VAULT_KEY not set - vault DISABLED")
except Exception as e:  # cryptography missing or bad key
    logger.error("[sakz_execution] Vault init failed: %s", e)
    Fernet = None
    InvalidToken = Exception
    _fernet = None


def vault_ready() -> bool:
    """True when SAKZ_VAULT_KEY is configured and encryption is usable."""
    return _fernet is not None


def generate_vault_key() -> str:
    """Generate a fresh Fernet master key. Run ONCE, store as SAKZ_VAULT_KEY."""
    from cryptography.fernet import Fernet as _F
    return _F.generate_key().decode()


def encrypt_secret(plaintext: str) -> str:
    if not _fernet:
        raise RuntimeError("Vault not configured (SAKZ_VAULT_KEY missing)")
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str) -> str:
    if not _fernet:
        raise RuntimeError("Vault not configured (SAKZ_VAULT_KEY missing)")
    return _fernet.decrypt(token.encode()).decode()


# -- Bybit V5 signed request --------------------------------------------------
def _sign(api_secret: str, payload: str) -> str:
    return hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def bybit_signed_request(api_key, api_secret, method, path, params=None, testnet=None):
    """Make a signed Bybit V5 request and return the parsed JSON dict.

    Signing (v5):
        sign = HMAC_SHA256(secret, timestamp + api_key + recv_window + payload)
          * GET  -> payload = sorted urlencoded query string
          * POST -> payload = raw compact JSON body
    Raises on network/timeout errors (callers should wrap in try/except).
    """
    base   = _base_url(testnet)
    params = params or {}
    ts     = str(int(time.time() * 1000))
    method = method.upper()

    if method == "GET":
        query   = "&".join(f"{k}={params[k]}" for k in sorted(params))
        payload = query
        sign    = _sign(api_secret, ts + api_key + RECV_WINDOW + payload)
        url     = f"{base}{path}" + (f"?{query}" if query else "")
        body    = None
    else:
        payload = json.dumps(params, separators=(",", ":")) if params else ""
        sign    = _sign(api_secret, ts + api_key + RECV_WINDOW + payload)
        url     = f"{base}{path}"
        body    = payload

    headers = {
        "X-BAPI-API-KEY":     api_key,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN":        sign,
        "X-BAPI-SIGN-TYPE":   "2",
        "Content-Type":       "application/json",
    }
    if method == "GET":
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    else:
        r = requests.post(url, headers=headers, data=body, timeout=HTTP_TIMEOUT)
    return r.json()


def test_connection(api_key, api_secret, testnet=None):
    """Validate a key pair against Bybit. Returns (ok: bool, detail: str).

    Reads the UNIFIED wallet balance - a Trade-scoped key can read this; it
    never needs Withdraw permission. Used by /connect to confirm the key works
    before we store it.
    """
    try:
        resp = bybit_signed_request(
            api_key, api_secret, "GET",
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED"},
            testnet=testnet,
        )
    except Exception as e:
        return False, f"network error: {e}"
    ret = resp.get("retCode")
    if ret == 0:
        try:
            acct   = resp["result"]["list"][0]
            equity = acct.get("totalEquity", "?")
            return True, f"equity={equity}"
        except Exception:
            return True, "connected"
    return False, resp.get("retMsg", f"retCode={ret}")
