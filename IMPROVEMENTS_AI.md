# sakz-bot — AI Improvement Pass

Focus: high-impact, **safe, behavior-preserving** reliability wins that don't
require live exchange/Telegram credentials to verify. Every change was checked
with `python -m py_compile` across all modules and an AST re-scan.

## 1. Restored observability on silently-swallowed exceptions (61 sites)

Broad handlers of the form `except Exception: pass` / `: continue` / `: return None`
were swallowing failures with **zero trace** — the worst failure mode for a
trading bot, because a failed order, bad fill, or dropped DB write looks
identical to success.

Each broad silent handler now logs the full stack trace at DEBUG before doing
the *exact same thing it did before* (control flow is unchanged):

```python
except Exception:
    logger.debug("suppressed exception in <function>", exc_info=True)
    # ... original pass / continue / return None preserved
```

Updated: `sakz_bot.py` (38), `sakz_exchanges.py` (10), `sakz_db.py` (8),
`sakz_scanner.py` (5).

**Intentional** narrow handlers were left untouched: `ImportError` (optional
deps), `json.JSONDecodeError` (WS frame skip), `ccxt.BadSymbol`,
`StopIteration`, and `TypeError`/`ValueError` parse guards. The deliberately
import-free modules (`sakz_signal_logic.py`, `sakz_prime.py`, etc.) were also
left alone to preserve their isolated-testability design.

To see the suppressed errors while debugging: `export SAKZ_LOG_LEVEL=DEBUG`.

## 2. Production-grade logging bootstrap

Replaced the bare `logging.basicConfig(level=INFO)` with `_setup_logging()`:

- **Env-configurable level**: `SAKZ_LOG_LEVEL` (default INFO).
- **Rotating file logs** (off unless set): `SAKZ_LOG_FILE`,
  `SAKZ_LOG_MAX_MB` (default 10), `SAKZ_LOG_BACKUPS` (default 5) — survives
  restarts on the Oracle Always-Free VM without unbounded disk growth.
- **Quieted noisy libraries** (httpx, urllib3, telegram, apscheduler,
  websockets, ccxt, matplotlib) so signal logs stay readable.
- Idempotent (safe if called twice).

## Verified

- `python -m py_compile *.py` → all modules compile.
- AST re-scan → 0 remaining broad silent `pass`/`continue` in runtime modules.

## Recommended next (needs live deps / runtime to validate — not auto-applied)

- **Pin `requirements.txt`** (pandas, numpy, requests, ta, scikit-learn, etc.
  are unpinned → non-reproducible deploys). Pin against your known-good venv.
- **Split the 17,385-line `sakz_bot.py` monolith** (matches your existing Code
  Review & Refactor Plan): extract the Telegram command handlers, the scan
  loop, and chart rendering into separate modules.
- **Triage the 360 broad `except Exception` handlers**: many should catch
  specific exceptions and/or surface a user/admin alert rather than DEBUG.
- **Replace remaining `print()` calls** (81) in runtime paths with `logger`.
