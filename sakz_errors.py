"""sakz_errors.py - structured error taxonomy (LEAF module).

Lets callers distinguish a real infrastructure failure (database / exchange
down) from a legitimate empty result ("no signals right now"). Raise these at
the boundary where the failure is *detected*, and catch them where you decide
whether the bot is quiet because nothing qualified vs. quiet because something
broke.

    try:
        conn = db_connect()
    except SakzDBError:
        # infrastructure problem - alert / back off, do NOT report "no signals"
        ...

Imports nothing project-local, so it is safe to import from any module without
risking an import cycle.
"""


class SakzError(Exception):
    """Base class for all Sakz-bot domain errors."""


class SakzDBError(SakzError):
    """Database / persistence failure (e.g. Turso or SQLite connection dropped)."""


class SakzExchangeError(SakzError):
    """Exchange / data-fetch failure (e.g. every OHLCV source unreachable)."""


class SakzSignalError(SakzError):
    """Signal-engine failure (unexpected error while scoring or generating a signal)."""
