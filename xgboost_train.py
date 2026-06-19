"""
xgboost_train.py — XGBoost win-probability model for SakzBot.

Exposes three names imported by sakz_bot_main.py:
    predict_signal(r)  → float  (0.0–1.0 win probability for a live signal dict)
    train(db_path)     → dict   (training metadata, saves model to xgb_signal_model.pkl)
    model_meta()       → dict   (info about the loaded model, or empty dict)
"""

import os
import pickle
import sqlite3
import logging

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = "xgb_signal_model.pkl"

# ── Feature extraction ────────────────────────────────────────────────────────

BIAS_MAP     = {"LONG": 1, "SHORT": 0}
EXCHANGE_MAP = {"BYBIT": 0, "BINANCE": 1, "MEXC": 2}
REGIME_MAP   = {"STRONG_BULL": 2, "BULL": 1, "NEUTRAL": 0, "BEAR": -1, "STRONG_BEAR": -2}
VOL_MAP      = {"LOW": 0, "RANGING": 1, "MEDIUM": 2, "HIGH": 3, "EXTREME": 4}


def _extract_features(r: dict) -> list:
    """
    Extract a fixed-length feature vector from a signal dict.
    Works for both live signals (from score_pair) and outcome rows (from DB).
    All features are numeric and robust to missing keys.
    """
    conf     = float(r.get("confidence", 5))
    bias_enc = BIAS_MAP.get(str(r.get("bias", "LONG")), 0)
    exch_enc = EXCHANGE_MAP.get(str(r.get("exchange", "BYBIT")), 0)

    price    = float(r.get("price", 0) or 0)
    entry_lo = float(r.get("entry_low",  price) or price)
    entry_hi = float(r.get("entry_high", price) or price)
    sl       = float(r.get("stop_loss",  price) or price)
    t1       = float(r.get("t1", price) or price)
    t2       = float(r.get("t2", price) or price)
    t3       = float(r.get("t3", price) or price)

    # Normalised distances (avoid division by zero)
    ref = price if price > 0 else 1.0
    t1_dist  = abs(t1  - price) / ref
    t2_dist  = abs(t2  - price) / ref
    t3_dist  = abs(t3  - price) / ref
    sl_dist  = abs(sl  - price) / ref
    ez_width = abs(entry_hi - entry_lo) / ref

    rr1 = t1_dist / sl_dist if sl_dist > 0 else 0.0
    rr2 = t2_dist / sl_dist if sl_dist > 0 else 0.0

    rsi4   = float(r.get("rsi4",   50) or 50)
    rsi_d  = float(r.get("rsi_d",  50) or 50)
    stoch_k = float(r.get("stoch_k", 50) or 50)
    funding = float(r.get("funding", 0) or 0)
    atr     = float(r.get("atr", 0) or 0)
    atr_pct = atr / ref if ref > 0 else 0.0

    dur_score    = float(r.get("dur_score", 0) or 0)
    hold_hours   = float(r.get("hold_hours", 24) or 24)
    counter_trend = 1 if r.get("counter_trend") else 0
    entry_in_zone = 1 if r.get("entry_in_zone", True) else 0
    entry_confirmed = int(r.get("entry_confirmed", -1) or -1)

    regime_enc = REGIME_MAP.get(str(r.get("btc_regime", "NEUTRAL")), 0)
    vol_enc    = VOL_MAP.get(str(r.get("vol_regime",  "MEDIUM")),  2)

    lev = r.get("leverage") or {}
    lev_suggested = float(lev.get("suggested", 1) if isinstance(lev, dict) else 1)

    return [
        conf, bias_enc, exch_enc,
        t1_dist, t2_dist, t3_dist, sl_dist, ez_width,
        rr1, rr2,
        rsi4, rsi_d, stoch_k,
        funding, atr_pct,
        dur_score, hold_hours,
        counter_trend, entry_in_zone, entry_confirmed,
        regime_enc, vol_enc,
        lev_suggested,
    ]

FEATURE_NAMES = [
    "confidence", "bias_enc", "exchange_enc",
    "t1_dist", "t2_dist", "t3_dist", "sl_dist", "ez_width",
    "rr1", "rr2",
    "rsi4", "rsi_d", "stoch_k",
    "funding", "atr_pct",
    "dur_score", "hold_hours",
    "counter_trend", "entry_in_zone", "entry_confirmed",
    "btc_regime_enc", "vol_regime_enc",
    "leverage_suggested",
]

# ── Model cache ───────────────────────────────────────────────────────────────

_model      = None   # XGBClassifier
_meta: dict = {}


def _load_model():
    global _model, _meta
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle  = pickle.load(f)
        _model = bundle["model"]
        _meta  = bundle.get("meta", {})
        logger.info("XGBoost model loaded from %s (AUC %.3f)", MODEL_PATH,
                    _meta.get("cv_roc_auc_mean", 0))
        return _model
    except Exception as e:
        logger.warning("Failed to load XGBoost model: %s", e)
        return None


def model_meta() -> dict:
    _load_model()
    return _meta


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_signal(r: dict) -> float:
    """
    Return win probability (0.0–1.0) for a live signal dict.
    Returns 0.5 (neutral) if the model is not yet trained.
    """
    mdl = _load_model()
    if mdl is None:
        return 0.5
    try:
        feats = np.array([_extract_features(r)], dtype=np.float32)
        prob  = float(mdl.predict_proba(feats)[0][1])
        return round(prob, 4)
    except Exception as e:
        logger.debug("XGB predict error: %s", e)
        return 0.5


# ── Training ──────────────────────────────────────────────────────────────────

# The trade resolver (sakz_bot.resolve/track jobs) records the realised result in
# signal_outcomes.outcome as one of: 't1_hit' | 't2_hit' | 't3_hit' | 'sl_hit' |
# 'pending' | 'expired'.  It NEVER writes the literals 'win'/'loss'.  A win is any
# target hit (t1/t2/t3); a loss is a stop-out (sl_hit).  This mirrors the win/loss
# accounting used throughout sakz_bot.py (see the stats jobs, e.g. ~line 4857).
WIN_OUTCOMES  = ("t1_hit", "t2_hit", "t3_hit")
LOSS_OUTCOMES = ("sl_hit",)
RESOLVED_OUTCOMES = WIN_OUTCOMES + LOSS_OUTCOMES


def _load_outcomes(db_path: str) -> tuple:
    """
    Pull resolved signal_outcomes rows from the DB.
    Returns (X, y) where:
        X  — numpy float32 array (n_samples, n_features)
        y  — numpy int array, 1=win (target hit) 0=loss (stop hit)
    Only resolved target/stop outcomes are used; 'pending'/'expired' are skipped.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in RESOLVED_OUTCOMES)
    rows = conn.execute(
        f"SELECT * FROM signal_outcomes WHERE outcome IN ({placeholders})",
        RESOLVED_OUTCOMES,
    ).fetchall()
    conn.close()

    X, y = [], []
    for row in rows:
        r = dict(row)
        # Re-hydrate fields that score_pair normally computes
        r.setdefault("entry_in_zone", 1)
        r.setdefault("dur_score",     5)
        r.setdefault("hold_hours",   24)
        r.setdefault("btc_regime",   "NEUTRAL")
        r.setdefault("vol_regime",   "MEDIUM")
        r.setdefault("counter_trend", 0)
        r.setdefault("leverage",     {})
        feats = _extract_features(r)
        label = 1 if r["outcome"] in WIN_OUTCOMES else 0
        X.append(feats)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(db_path: str = "sakz_data.db") -> dict:
    """
    Train an XGBoost win-probability classifier on historical signal outcomes.
    Saves the model to xgb_signal_model.pkl.
    Returns a metadata dict (n_samples, win_rate, cv_roc_auc_mean, top_features, …).
    Raises RuntimeError if there are too few labelled samples.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Run: pip install xgboost scikit-learn")

    X, y = _load_outcomes(db_path)

    if len(X) < 30:
        raise RuntimeError(
            f"Only {len(X)} resolved outcomes — need at least 30 to train. "
            "Keep running the bot to collect more signal results."
        )

    n_wins   = int(y.sum())
    n_losses = int(len(y) - n_wins)
    win_rate = n_wins / len(y)

    # Scale pos_weight to handle class imbalance
    scale_pw = n_losses / n_wins if n_wins > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=min(5, n_wins, n_losses), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    # Fit the base model (used for feature importances + as the calibration base)
    model.fit(X, y)

    # Feature importance (from the uncalibrated base model)
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])

    # ── PROBABILITY CALIBRATION ────────────────────────────────────
    # Raw tree predict_proba is usually miscalibrated, so a shown "78% win" was
    # not really 78%. Wrap the model in CalibratedClassifierCV so the win-% users
    # see is statistically honest. Isotonic needs more data; fall back to Platt
    # (sigmoid) on smaller samples.
    final_model = model
    cal_method  = None
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone
        cal_cv = min(3, n_wins, n_losses)
        if cal_cv >= 2:
            cal_method = "isotonic" if len(X) >= 200 else "sigmoid"
            calibrated = CalibratedClassifierCV(clone(model), method=cal_method, cv=cal_cv)
            calibrated.fit(X, y)
            final_model = calibrated
            logger.info("XGBoost probabilities calibrated via %s (cv=%d)", cal_method, cal_cv)
    except Exception as _cal_e:
        logger.warning("XGBoost calibration skipped (%s) — using raw probabilities", _cal_e)
        cal_method = None

    meta = {
        "n_samples":        len(X),
        "n_wins":           n_wins,
        "n_losses":         n_losses,
        "win_rate":         win_rate,
        "cv_roc_auc_mean":  float(cv_scores.mean()),
        "cv_roc_auc_std":   float(cv_scores.std()),
        "top_features":     top_features,
        "calibrated":       cal_method is not None,
        "calibration":      cal_method or "none",
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "meta": meta}, f)

    # Reload into cache
    global _model, _meta
    _model = final_model
    _meta  = meta

    logger.info(
        "XGBoost trained: %d samples | win_rate %.1f%% | CV AUC %.3f ± %.3f",
        len(X), win_rate * 100, meta["cv_roc_auc_mean"], meta["cv_roc_auc_std"]
    )
    return meta
