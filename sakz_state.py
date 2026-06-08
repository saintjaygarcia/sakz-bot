"""sakz_state.py - shared mutable runtime state for the sakz bot.

LEAF MODULE: imports only the standard library; never import sakz_* here.
All layers (sakz_bot / sakz_commands / sakz_scanner) read & WRITE these via
state.<name> so reassignments stay visible across modules (a plain split of
these globals would silently desync because `global X` only rebinds within
the module that defines X).
"""

DEFAULT_OPTIM_PARAMS = {
    "atr_stop_mult":    1.0,
    "atr_target_mult":  1.6,
    "min_confidence":   4,
    "cap_osc":          3,
    "cap_mtf":          2,
    "cap_cross":        3,
    "cap_pos":          3,
    "cap_structure":    4,
    "rsi_oversold":     30,
    "rsi_overbought":   70,
}
OPTIM_BEST_PARAMS = DEFAULT_OPTIM_PARAMS.copy()
_autorefresh_installed = False
_btc_price_cache: dict = {}   # {'price': float, 'time': datetime}
_btc_regime_cache     = None
_btcd_cache: dict = {}
_last_portfolio_summary = None
_last_signal_bias: dict = {}
_pro_gainers_cache: list      = []
_pro_gainers_cache_ts: float  = 0.0
_scan_cache        = None   # { 'results': [...], 'time': datetime }
_scan_durations: list = []   # rolling list of recent scan durations (seconds)
_signal_card_cache = {}
last_scan_results     = []
last_scan_time        = None
previous_scan_results = []
price_history         = {}   # { 'BYBIT_BTCUSDT': [{time, price},...] }
safemode_users: set = set()
snail_unlocked     = set()               # chat_ids that have unlocked snail mode
user_tracking         = {}   # { chat_id: { trade_id: {signal, entry_price, start_time, interval, job} } }
