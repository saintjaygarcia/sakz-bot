"""signal_types.py - Typed Signal container (non-invasive scaffold).

score_pair() currently returns a plain dict. This module adds a typed `Signal`
dataclass plus lossless dict<->Signal conversion so call sites can opt into type
safety incrementally WITHOUT touching the hot path. Any keys that are not declared
fields are preserved in `extra`, guaranteeing round-trip fidelity:

    Signal.from_dict(d).to_dict()           # superset of d (declared fields may
                                            # appear explicitly as None)
    Signal.from_dict(s.to_dict()) == s      # exact object round-trip

Field names mirror the common keys produced by score_pair; adapt as needed when
wiring it into the scoring path.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    bias: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    timeframe: Optional[str] = None
    reasons: List[Any] = field(default_factory=list)
    warnings: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def declared_fields(cls):
        return [f.name for f in fields(cls) if f.name != "extra"]

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        known = set(cls.declared_fields())
        kwargs = dict((k, data[k]) for k in list(data) if k in known)
        extra = dict((k, v) for k, v in data.items() if k not in known)
        obj = cls(**kwargs)
        obj.extra = extra
        return obj

    def to_dict(self):
        out = dict()
        for name in self.declared_fields():
            out[name] = getattr(self, name)
        out.update(self.extra)
        return out
