"""
rf_train.py — Random Forest win-probability model for SakzBot.

Exposes four names imported by sakz_bot_main.py:
    predict_signal(r)    → float  (0.0–1.0 win probability)
    train(db_path)       → dict   (training metadata, saves model to rf_signal_model.pkl)
    model_meta()         → dict   (info about the loaded model)
    consensus_verdict(p) → str    (human-readable label for a probability float)
"""

import os
import pickle
import sqlite3
import logging

import numpy as np

# Re-use the feature extraction from xgboost_train so both models use identical inputs.
try:
    from xgboost_train import _extract_features, FEATURE_NAMES, _load_outcomes
except ImportError:
    # Fallback: inline a minimal extractor if xgboost_train isn't available.
    FEATURE_NAMES = ["confidence", "bias_enc", "t1_dist", "sl_dist", "rr1", "rsi4", "dur_score"]

    def _extract_features(r: dict) -> list:
        price = float(r.get("price", 1) or 1)
        ref   = price if price > 0 else 1.0
        t1    = float(r.get("t1", price) or price)
        sl    = float(r.get("stop_loss", price) or price)
        t1_d  = abs(t1 - price) / ref
        sl_d  = abs(sl - price) / ref
        return [
            float(r.get("confidence", 5)),
            1 if str(r.get("bias","LONG")) == "LONG" else 0,
            t1_d, sl_d,
            t1_d / sl_d if sl_d > 0 else 0,
            float(r.get("rsi4", 50) or 50),
            float(r.get("dur_score", 0) or 0),
        ]

    # See xgboost_train._load_outcomes: outcomes are stored as t1/t2/t3_hit (win)
    # and sl_hit (loss); 'win'/'loss' are never written to the DB.
    WIN_OUTCOMES  = ("t1_hit", "t2_hit", "t3_hit")
    RESOLVED_OUTCOMES = WIN_OUTCOMES + ("sl_hit",)

    def _load_outcomes(db_path):
        try:
            from sakz_db import db_connect
            conn = db_connect()
        except Exception:
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
            X.append(_extract_features(r))
            y.append(1 if r["outcome"] in WIN_OUTCOMES else 0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


logger = logging.getLogger(__name__)

MODEL_PATH = "rf_signal_model.pkl"

# ── Verdict helper ────────────────────────────────────────────────────────────

def consensus_verdict(prob: float) -> str:
    """Convert a win probability float to a human-readable verdict."""
    if prob >= 0.75:  return "Strong edge ✅"
    if prob >= 0.62:  return "Moderate edge 🟡"
    if prob >= 0.50:  return "Slight edge ⚪"
    if prob >= 0.40:  return "Weak — caution ⚠️"
    return "Poor setup 🔴"


# ── Model cache ───────────────────────────────────────────────────────────────

_model      = None
_meta: dict = {}


def _load_model():
    global _model, _meta
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model = bundle["model"]
        _meta  = bundle.get("meta", {})
        logger.info("RF model loaded from %s (AUC %.3f)", MODEL_PATH,
                    _meta.get("cv_roc_auc_mean", 0))
        return _model
    except Exception as e:
        logger.warning("Failed to load RF model: %s", e)
        return None


def model_meta() -> dict:
    _load_model()
    return _meta


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_signal(r: dict) -> float:
    """Return win probability (0.0–1.0). Returns 0.5 if model not trained yet."""
    mdl = _load_model()
    if mdl is None:
        return 0.5
    try:
        feats = np.array([_extract_features(r)], dtype=np.float32)
        prob  = float(mdl.predict_proba(feats)[0][1])
        return round(prob, 4)
    except Exception as e:
        logger.debug("RF predict error: %s", e)
        return 0.5


# ── Training ──────────────────────────────────────────────────────────────────

def train(db_path: str = "sakz_data.db") -> dict:
    """
    Train a Random Forest win-probability classifier on historical signal outcomes.
    Saves the model to rf_signal_model.pkl.
    Returns a metadata dict.
    Raises RuntimeError if too few labelled samples.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Run: pip install scikit-learn")

    X, y = _load_outcomes(db_path)

    if len(X) < 30:
        raise RuntimeError(
            f"Only {len(X)} resolved outcomes — need at least 30 to train. "
            "Keep running the bot to collect more signal results."
        )

    n_wins   = int(y.sum())
    n_losses = int(len(y) - n_wins)
    win_rate = n_wins / len(y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=min(5, n_wins, n_losses), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    model.fit(X, y)

    importances  = dict(zip(FEATURE_NAMES, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])

    # ── PROBABILITY CALIBRATION ────────────────────────────────────
    # Honest win-% — wrap RF in CalibratedClassifierCV (isotonic with enough
    # data, else Platt/sigmoid). Falls back to raw probabilities on failure.
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
            logger.info("RF probabilities calibrated via %s (cv=%d)", cal_method, cal_cv)
    except Exception as _cal_e:
        logger.warning("RF calibration skipped (%s) — using raw probabilities", _cal_e)
        cal_method = None

    meta = {
        "n_samples":       len(X),
        "n_wins":          n_wins,
        "n_losses":        n_losses,
        "win_rate":        win_rate,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std":  float(cv_scores.std()),
        "top_features":    top_features,
        "calibrated":      cal_method is not None,
        "calibration":     cal_method or "none",
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "meta": meta}, f)

    global _model, _meta
    _model = final_model
    _meta  = meta

    logger.info(
        "RF trained: %d samples | win_rate %.1f%% | CV AUC %.3f ± %.3f",
        len(X), win_rate * 100, meta["cv_roc_auc_mean"], meta["cv_roc_auc_std"]
    )
    return meta
