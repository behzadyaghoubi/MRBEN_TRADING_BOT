from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegimeLabel(Enum):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class RegimeConfig:
    thr_extreme: float = 0.020
    thr_high: float = 0.015
    thr_medium: float = 0.010


@dataclass
class RegimeSnapshot:
    volatility: float
    label: RegimeLabel


class RegimeClassifier:
    def __init__(self, cfg: RegimeConfig | None = None) -> None:
        self.cfg = cfg or RegimeConfig()

    def classify_by_volatility(self, volatility: float | None) -> RegimeSnapshot:
        if volatility is None:
            return RegimeSnapshot(volatility=0.0, label=RegimeLabel.UNKNOWN)
        v = float(volatility)
        if v > self.cfg.thr_extreme:
            label = RegimeLabel.EXTREME
        elif v > self.cfg.thr_high:
            label = RegimeLabel.HIGH
        elif v > self.cfg.thr_medium:
            label = RegimeLabel.MEDIUM
        else:
            label = RegimeLabel.LOW
        return RegimeSnapshot(volatility=v, label=label)


def create_regime_classifier(cfg: RegimeConfig | None = None) -> RegimeClassifier:
    return RegimeClassifier(cfg=cfg)
