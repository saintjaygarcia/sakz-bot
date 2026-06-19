#!/usr/bin/env python3
"""Tests for the execution safety layer (sakz_orders) - fully offline.

Uses a throwaway temp DB and FAKE transports/ciphers, so it never touches the
network or real keys.
"""
import os, sys, tempfile, importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
_tmp = tempfile.mkdtemp()
os.environ["SAKZ_DB_PATH"] = os.path.join(_tmp, "orders_test.db")
for k in ("TURSO_URL", "TURSO_TOKEN", "TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN"):
    os.environ.pop(k, None)
os.environ["SAKZ_MAX_OPEN_POSITIONS"] = "2"
os.environ["SAKZ_MAX_NOTIONAL_PER_ORDER"] = "50"
os.environ["SAKZ_MAX_NOTIONAL_PER_SYMBOL"] = "60"
os.environ["SAKZ_MAX_TOTAL_NOTIONAL"] = "80"
os.environ["SAKZ_KELLY_FRACTION"] = "0.25"
os.environ["SAKZ_KELLY_CAP"] = "0.25"
os.environ["SAKZ_DAILY_LOSS_LIMIT_PCT"] = "0.05"
os.environ.pop("SAKZ_ALLOW_MAINNET", None)

import config; importlib.reload(config)
import sakz_db as db
import sakz_orders as orders
importlib.reload(orders)

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL") + "  - " + name)
    if not cond:
        fails.append(name)

DBC = db.db_connect
orders.orders_init_db(DBC)

def good_signal(symbol="BTCUSDT", bias="LONG", entry=100.0, sl=95.0, t1=115.0):
    return {"symbol": symbol, "exchange": "BYBIT", "bias": bias, "confidence": 9,
            "entry_price": entry, "stop_loss": sl, "t1": t1, "t2": t1*1.1, "t3": t1*1.2}

class FakeTx:
    """Records calls, returns a Bybit-style OK (or error) envelope."""
    def __init__(self, retCode=0, retMsg="OK"):
        self.calls = []; self.retCode = retCode; self.retMsg = retMsg
    def __call__(self, api_key, api_secret, method, path, params=None, testnet=None):
        self.calls.append({"path": path, "params": params, "testnet": testnet,
                           "orderLinkId": (params or {}).get("orderLinkId")})
        if self.retCode != 0:
            return {"retCode": self.retCode, "retMsg": self.retMsg}
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "EX-" + str(len(self.calls))}}

KEYS = lambda chat_id: ("fake_key", "fake_secret")

# ===================== gate + ledger basics =============================== #
conn = DBC(); cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('live_orders','trading_control','user_api_keys','pnl_ledger')")
tbls = {r[0] for r in cur.fetchall()}; conn.close()
check("all 4 tables created", tbls == {"live_orders", "trading_control", "user_api_keys", "pnl_ledger"})

check("normalize_side LONG->Buy", orders.normalize_side("LONG") == "Buy")
check("normalize_side short->Sell", orders.normalize_side("short") == "Sell")
check("normalize_side junk->None", orders.normalize_side("sideways") is None)

gate = orders.PreTradeGate(DBC)
check("invalid bias rejected", not gate.evaluate(good_signal(bias="meh"), equity=1000).ok)
check("incomplete signal rejected", not gate.evaluate({"bias": "LONG", "symbol": "X"}, equity=1000).ok)
check("long SL above entry rejected", not gate.evaluate(good_signal(entry=100, sl=105), equity=1000).ok)
check("short SL below entry rejected",
      not gate.evaluate(good_signal(bias="SHORT", entry=100, sl=95, t1=80), equity=1000).ok)

res = gate.evaluate(good_signal(), equity=1000)
check("happy-path gate ok", res.ok and res.side == "Buy" and res.qty > 0)
check("per-order notional cap enforced", res.notional <= 50.0 + 1e-9)
check("reward_risk computed (15/5=3)", abs(res.reward_risk - 3.0) < 1e-9)

orders.set_kill_switch(DBC, True, "manual test")
check("kill switch blocks", not gate.evaluate(good_signal(), equity=1000).ok)
check("is_halted true", orders.is_halted(DBC))
orders.set_kill_switch(DBC, False)
check("is_halted false after release", not orders.is_halted(DBC))

check("drawdown budget blocks", not gate.evaluate(good_signal(), equity=80.0, peak_equity=100.0).ok)
check("drawdown auto-halt engaged", orders.is_halted(DBC))
orders.set_kill_switch(DBC, False)

# ===================== live backend: send + idempotency =================== #
tx = FakeTx()
live = orders.LiveBackend(DBC, KEYS, signed_request=tx, testnet=True)
router = orders.OrderRouter(DBC, live)
out = router.submit(good_signal(), equity=1000, chat_id=1, order_link_id="olid-1")
check("live testnet submit -> submitted", out.get("status") == "submitted")
check("exactly one exchange send", len(tx.calls) == 1)
check("orderLinkId sent to exchange", tx.calls[0]["orderLinkId"] == "olid-1")
check("order create path used", tx.calls[0]["path"] == "/v5/order/create")

out2 = router.submit(good_signal(), equity=1000, chat_id=1, order_link_id="olid-1")
check("idempotent: recovered, no second send", out2.get("recovered") is True and len(tx.calls) == 1)

tx_bad = FakeTx(retCode=10001, retMsg="param error")
live_bad = orders.LiveBackend(DBC, KEYS, signed_request=tx_bad, testnet=True)
raised = False
try:
    res_bad = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="ETHUSDT"), equity=1000)
    live_bad.place(good_signal(symbol="ETHUSDT"), res_bad, chat_id=1, order_link_id="olid-bad")
except orders.SakzExecError:
    raised = True
check("bad retCode raises SakzExecError", raised)
conn = DBC(); cur = conn.cursor()
cur.execute("SELECT status FROM live_orders WHERE order_link_id='olid-bad'")
row = cur.fetchone(); conn.close()
check("failed order marked 'failed' in ledger", row is not None and row[0] == "failed")

live_main = orders.LiveBackend(DBC, KEYS, signed_request=FakeTx(), testnet=False)
refused = False
try:
    r = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="SOLUSDT"), equity=1000)
    live_main.place(good_signal(symbol="SOLUSDT"), r, chat_id=1, order_link_id="olid-main")
except orders.SakzExecError:
    refused = True
check("mainnet send refused without SAKZ_ALLOW_MAINNET", refused)

ok_env = orders.validated_request(FakeTx(), "k", "s", "GET", "/v5/x", {})
check("validated_request returns dict on retCode 0", isinstance(ok_env, dict))

tx2 = FakeTx()
live2 = orders.LiveBackend(DBC, KEYS, signed_request=tx2, testnet=True)
r2 = orders.OrderRouter(DBC, live2)
r2.submit(good_signal(symbol="ADAUSDT"), equity=1000, chat_id=1, order_link_id="olid-2")
blocked = r2.submit(good_signal(symbol="XRPUSDT"), equity=1000, chat_id=1, order_link_id="olid-3")
check("max open positions cap rejects 3rd", blocked.get("status") == "rejected")

# ===================== NEW: reset ledger, then advanced features ========== #
conn = DBC(); conn.execute("DELETE FROM live_orders"); conn.commit(); conn.close()

# --- rounding (pure) ---
check("round_down_step lot 0.001", abs(orders.round_down_step(1.2345, 0.001) - 1.234) < 1e-9)
check("round_down_step no-step passthrough", orders.round_down_step(5.0, 0) == 5.0)

# --- tick/lot rounding in LiveBackend ---
def instr(sym):
    return {"qtyStep": 0.01, "tickSize": 0.1, "minOrderQty": 0.01}
tx_r = FakeTx()
live_r = orders.LiveBackend(DBC, KEYS, signed_request=tx_r, testnet=True, get_instrument=instr)
g_r = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="DOTUSDT"), equity=1000)
live_r.place(good_signal(symbol="DOTUSDT"), g_r, chat_id=1, order_link_id="olid-round")
sent_qty = float(tx_r.calls[0]["params"]["qty"])
check("qty rounded to lot step (0.01)", sent_qty > 0 and abs(sent_qty * 100 - round(sent_qty * 100)) < 1e-6)

# --- min order qty enforced ---
def instr_big(sym):
    return {"qtyStep": 0.01, "tickSize": 0.1, "minOrderQty": 1000000.0}
live_big = orders.LiveBackend(DBC, KEYS, signed_request=FakeTx(), testnet=True, get_instrument=instr_big)
g_big = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="LINKUSDT"), equity=1000)
raised_min = False
try:
    live_big.place(good_signal(symbol="LINKUSDT"), g_big, chat_id=1, order_link_id="olid-min")
except orders.SakzExecError:
    raised_min = True
check("min order qty enforced", raised_min)

# --- per-user encrypted key vault (injected fake cipher) ---
fenc = lambda s: "enc:" + s
fdec = lambda s: s[4:] if s.startswith("enc:") else s
orders.store_user_keys(DBC, 4242, "MYKEY", "MYSECRET", testnet=True, encrypt=fenc)
k, s = orders.load_user_keys(DBC, 4242, decrypt=fdec)
check("key vault round-trip", (k, s) == ("MYKEY", "MYSECRET"))
getter = orders.make_key_getter(DBC, decrypt=fdec)
check("key getter returns tuple", getter(4242) == ("MYKEY", "MYSECRET"))
conn = DBC(); cur = conn.cursor()
cur.execute("SELECT api_key_enc FROM user_api_keys WHERE chat_id=4242")
enc_val = cur.fetchone()[0]; conn.close()
check("stored key is encrypted (not plaintext)", enc_val != "MYKEY")
raised_nokey = False
try:
    orders.load_user_keys(DBC, 999999, decrypt=fdec)
except orders.SakzExecError:
    raised_nokey = True
check("missing keys raises", raised_nokey)

# --- real Fernet vault round-trip (best effort; skipped if unavailable) ---
try:
    import sakz_execution as ex
    os.environ["SAKZ_VAULT_KEY"] = ex.generate_vault_key()
    importlib.reload(ex)
    if ex.vault_ready():
        orders.store_user_keys(DBC, 7, "AK", "AS", testnet=True)
        kk, ss = orders.load_user_keys(DBC, 7)
        check("real Fernet vault round-trip", (kk, ss) == ("AK", "AS"))
    else:
        print("SKIP  - real Fernet (vault not ready)")
except Exception as e:
    print("SKIP  - real Fernet:", e)

# --- win rate from signal_outcomes ---
conn = DBC()
conn.execute("""CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id INTEGER, exchange TEXT, symbol TEXT,
    bias TEXT, confidence INTEGER, entry_price REAL, stop_loss REAL, t1 REAL, t2 REAL,
    t3 REAL, scan_time TEXT, outcome TEXT)""")
for i in range(8):
    conn.execute("INSERT INTO signal_outcomes (signal_id,exchange,symbol,bias,confidence,"
                 "entry_price,stop_loss,t1,t2,t3,scan_time,outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (i, "BYBIT", "WINUSDT", "LONG", 9, 100, 95, 110, 120, 130, "now", "t1_hit"))
for i in range(2):
    conn.execute("INSERT INTO signal_outcomes (signal_id,exchange,symbol,bias,confidence,"
                 "entry_price,stop_loss,t1,t2,t3,scan_time,outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (i, "BYBIT", "WINUSDT", "LONG", 9, 100, 95, 110, 120, 130, "now", "sl_hit"))
conn.commit(); conn.close()
wr, n = orders.win_rate_for_symbol(DBC, "WINUSDT", min_sample=5)
check("win_rate_for_symbol 8/10 = 0.8", abs(wr - 0.8) < 1e-9 and n == 10)
wr2, n2 = orders.win_rate_for_symbol(DBC, "RAREUSDT", min_sample=5, default=0.55)
check("win_rate below sample -> default", wr2 == 0.55 and n2 == 0)

# --- daily-loss auto-halt ---
orders.set_kill_switch(DBC, False)
orders.record_realized_pnl(DBC, -60.0, symbol="BTCUSDT", chat_id=1)  # -60 vs 5% of 1000 = -50
check("daily_realized_pnl sums", abs(orders.daily_realized_pnl(DBC) + 60.0) < 1e-9)
g_dl = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="DLUSDT"), equity=1000)
check("daily loss limit blocks", (not g_dl.ok) and "daily loss" in g_dl.reason)
check("daily loss auto-halt engaged", orders.is_halted(DBC))
orders.set_kill_switch(DBC, False)
conn = DBC(); conn.execute("DELETE FROM pnl_ledger"); conn.commit(); conn.close()

# --- server-time sync ---
check("compute_time_offset", orders.compute_time_offset(1000, 600) == 400)
clk = orders.ServerClock(get_server_ms=lambda: 5000, local_ms=lambda: 4000)
check("ServerClock.sync offset", clk.sync() == 1000)
check("ServerClock.now_ms applies offset", clk.now_ms() == 5000)
import sakz_execution as ex2
ex2.set_time_offset_ms(123456)
check("set_time_offset_ms stored in sakz_execution", ex2.get_time_offset_ms() == 123456)
ex2.set_time_offset_ms(0)

# --- reconciliation maps exchange status -> ledger ---
class ReconTx:
    def __init__(self, status):
        self.status = status; self.calls = []
    def __call__(self, ak, sk, method, path, params=None, testnet=None):
        self.calls.append(path)
        if path == "/v5/order/create":
            return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "EX-R"}}
        return {"retCode": 0, "retMsg": "OK", "result": {"list": [{
            "orderId": "EX-R", "orderLinkId": (params or {}).get("orderLinkId"),
            "orderStatus": self.status, "cumExecQty": "0.5", "avgPrice": "100.0"}]}}
conn = DBC(); conn.execute("DELETE FROM live_orders"); conn.commit(); conn.close()
rtx = ReconTx("Filled")
live_rec = orders.LiveBackend(DBC, KEYS, signed_request=rtx, testnet=True)
g_rec = orders.PreTradeGate(DBC).evaluate(good_signal(symbol="RECUSDT"), equity=1000)
live_rec.place(good_signal(symbol="RECUSDT"), g_rec, chat_id=1, order_link_id="olid-rec")
summary = live_rec.reconcile()
conn = DBC(); cur = conn.cursor()
cur.execute("SELECT status, filled_qty FROM live_orders WHERE order_link_id='olid-rec'")
row = cur.fetchone(); conn.close()
check("reconcile maps Filled -> open", row[0] == "open")
check("reconcile records filled_qty", abs((row[1] or 0) - 0.5) < 1e-9)
check("reconcile summary updated>=1", summary.get("updated", 0) >= 1)

# --- admin status text ---
txt = orders.orders_status_text(DBC)
check("status text has Trading + Open lines", "Trading:" in txt and "Open positions:" in txt)

if fails:
    print("\nRESULT: ORDER-LAYER TEST FAILURES:", fails)
    sys.exit(1)
print("\nRESULT: ALL ORDER-LAYER CHECKS PASSED")
