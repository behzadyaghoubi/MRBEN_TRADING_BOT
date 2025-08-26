#!/usr/bin/env python3
"""
Policy Brain - Layer 2 of MR BEN AI Architecture
Intelligent decision maker for dynamic TP/SL/Trailing/Position sizing
"""
import json
import logging
import os

# Import our enhanced components
try:
    from utils.conformal import ConformalGate
    from utils.regime import detect_regime, get_regime_multipliers

    ENHANCED_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced AI components not available: {e}")
    ENHANCED_AI_AVAILABLE = False


class PolicyBrain:
    """
    Layer 2 AI Brain - Meta decision maker and policy controller

    Responsibilities:
    - Accept/reject decisions using Conformal + Meta-model
    - Dynamic TP/SL/Trailing parameter optimization
    - Regime-aware position sizing
    - Policy learning and adaptation
    """

    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger("PolicyBrain")
        self.config = self._load_config(config_path)

        # Initialize components
        self.conformal_gate = None
        self.policy_model = None
        self._init_components()

        # Policy parameters (will be learned via RL later)
        self.policy_params = {
            "base_tp1_r": 0.8,
            "base_tp2_r": 1.5,
            "base_sl_mult": 1.2,
            "base_tp_share": 0.5,
            "trailing_modes": ["off", "chandelier", "supertrend"],
            "size_factors": {"conservative": 0.7, "normal": 1.0, "aggressive": 1.3},
        }

        # Decision counters for monitoring
        self.stats = {
            "total_decisions": 0,
            "accepted": 0,
            "rejected_conformal": 0,
            "rejected_risk": 0,
            "rejected_regime": 0,
        }

    def _load_config(self, config_path: str) -> dict:
        """Load configuration with proper encoding handling"""
        default_config = {
            "ai_control": {
                "mode": "copilot",
                "conformal_alpha": 0.10,
                "tp_policy": {
                    "split": True,
                    "tp1_r": 0.8,
                    "tp2_r": 1.5,
                    "tp1_share": 0.5,
                    "breakeven_after_tp1": True,
                },
                "policy_learning": {
                    "enabled": True,
                    "update_frequency": 100,  # trades
                    "learning_rate": 0.001,
                },
            },
            "risk": {"max_daily_loss": 0.02, "max_trades_per_day": 10, "base_risk": 0.005},
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, encoding='utf-8') as f:
                    config = json.load(f)
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
                return default_config
        else:
            return default_config

    def _init_components(self):
        """Initialize AI components"""
        if ENHANCED_AI_AVAILABLE:
            try:
                # Initialize Conformal Gate
                self.conformal_gate = ConformalGate(
                    "models/meta_filter.joblib", "models/conformal.json"
                )
                if self.conformal_gate.is_available():
                    self.logger.info("✅ Policy Brain: Conformal gate initialized")
                else:
                    self.logger.warning("⚠️ Policy Brain: Conformal gate not available")
                    self.conformal_gate = None

                # Load RL Policy model (if exists)
                policy_path = "models/policy_rl.pt"
                if os.path.exists(policy_path):
                    try:
                        import torch

                        self.policy_model = torch.load(policy_path, map_location='cpu')
                        self.logger.info("✅ Policy Brain: RL policy model loaded")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Policy Brain: Failed to load RL model: {e}")
                        self.policy_model = None
                else:
                    self.logger.info("ℹ️ Policy Brain: No RL model found, using rule-based policy")

            except Exception as e:
                self.logger.error(f"❌ Policy Brain initialization failed: {e}")
                self.conformal_gate = None
                self.policy_model = None

    def propose(self, snapshot: dict) -> dict:
        """
        Main decision function - analyze snapshot and propose trading action

        Args:
            snapshot: {
                "features": {...},  # close, rsi, macd, atr, hour, dow, etc.
                "layer1": {"signal": -1|0|1, "confidence": 0..1, "score": -1..1},
                "market": {"spread": ..., "session": "London", "regime": "RANGE"},
                "context": {"daily_pnl": ..., "trades_today": ..., "equity": ...}
            }

        Returns:
            {
                "decision": "BUY"|"SELL"|"HOLD",
                "accept_reason": "conformal_ok|meta_pass|policy_ok|risk_reject...",
                "tp1_r": 0.8, "tp2_r": 1.5, "tp_share": 0.5,
                "sl_mult": 1.2, "trailing": "chandelier|supertrend|off",
                "size_factor": 1.0,
                "confidence": 0.0..1.0,
                "expected_value": float
            }
        """
        self.stats["total_decisions"] += 1

        try:
            # Extract key information
            features = snapshot.get("features", {})
            layer1 = snapshot.get("layer1", {})
            market = snapshot.get("market", {})
            context = snapshot.get("context", {})

            signal = layer1.get("signal", 0)
            confidence = layer1.get("confidence", 0.0)
            regime = market.get("regime", "UNKNOWN")

            # If no signal from Layer 1, return HOLD
            if signal == 0:
                return self._create_decision("HOLD", "no_layer1_signal", 0.0, 0.0)

            # Step 1: Conformal Gate Check
            conformal_result = self._check_conformal_gate(features, regime)
            if not conformal_result["accepted"]:
                self.stats["rejected_conformal"] += 1
                return self._create_decision(
                    "HOLD", "conformal_reject", conformal_result["probability"], 0.0
                )

            # Step 2: Risk Governor Pre-check
            risk_result = self._check_risk_constraints(context, market)
            if not risk_result["accepted"]:
                self.stats["rejected_risk"] += 1
                return self._create_decision("HOLD", risk_result["reason"], confidence, 0.0)

            # Step 3: Policy Decision (Rule-based or RL)
            policy_decision = self._make_policy_decision(
                features, layer1, market, context, conformal_result
            )

            # Step 4: Regime-aware parameter adjustment
            adjusted_decision = self._adjust_for_regime(policy_decision, regime, market)

            # Step 5: Calculate Expected Value
            ev = self._calculate_expected_value(adjusted_decision, conformal_result["probability"])
            adjusted_decision["expected_value"] = ev

            # Final acceptance
            if adjusted_decision["decision"] != "HOLD":
                self.stats["accepted"] += 1
                self.logger.info(
                    f"PolicyBrain ACCEPT: {adjusted_decision['decision']} | "
                    f"P={conformal_result['probability']:.3f} | "
                    f"Regime={regime} | EV={ev:.3f} | "
                    f"TP1={adjusted_decision['tp1_r']:.2f}R"
                )

            return adjusted_decision

        except Exception as e:
            self.logger.error(f"Error in PolicyBrain.propose: {e}")
            return self._create_decision("HOLD", "error", 0.0, 0.0)

    def _check_conformal_gate(self, features: dict, regime: str) -> dict:
        """Check conformal prediction gate"""
        if self.conformal_gate is None:
            return {"accepted": True, "probability": 0.5, "nonconformity": 0.5}

        try:
            accepted, prob, nonconf = self.conformal_gate.accept(features)
            return {"accepted": accepted, "probability": prob, "nonconformity": nonconf}
        except Exception as e:
            self.logger.warning(f"Conformal gate error: {e}")
            return {"accepted": False, "probability": 0.0, "nonconformity": 1.0}

    def _check_risk_constraints(self, context: dict, market: dict) -> dict:
        """Check risk management constraints"""
        max_daily_loss = self.config["risk"]["max_daily_loss"]
        max_trades = self.config["risk"]["max_trades_per_day"]

        daily_pnl = context.get("daily_pnl", 0.0)
        trades_today = context.get("trades_today", 0)
        equity = context.get("equity", 10000)
        spread = market.get("spread", 0)

        # Check daily loss limit
        if daily_pnl <= -abs(max_daily_loss * equity):
            return {"accepted": False, "reason": "daily_loss_limit"}

        # Check daily trade limit
        if trades_today >= max_trades:
            return {"accepted": False, "reason": "daily_trade_limit"}

        # Check spread (basic implementation)
        if spread > 100:  # Example threshold
            return {"accepted": False, "reason": "spread_too_wide"}

        return {"accepted": True, "reason": "risk_ok"}

    def _make_policy_decision(
        self, features: dict, layer1: dict, market: dict, context: dict, conformal: dict
    ) -> dict:
        """Make policy decision using RL model or rule-based approach"""

        signal = layer1.get("signal", 0)
        confidence = layer1.get("confidence", 0.0)

        decision = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

        if self.policy_model is not None:
            # Use RL model for policy decisions
            policy_params = self._rl_policy_decision(features, layer1, market, conformal)
        else:
            # Use rule-based policy
            policy_params = self._rule_based_policy(features, layer1, market, conformal)

        return {
            "decision": decision,
            "accept_reason": "policy_ok",
            "confidence": confidence,
            **policy_params,
        }

    def _rule_based_policy(
        self, features: dict, layer1: dict, market: dict, conformal: dict
    ) -> dict:
        """Rule-based policy for TP/SL/Trailing decisions"""

        confidence = layer1.get("confidence", 0.0)
        conformal_prob = conformal.get("probability", 0.5)

        # Adjust TP/SL based on confidence and conformal probability
        confidence_mult = min(1.2, 0.8 + 0.4 * confidence)
        conformal_mult = min(1.15, 0.9 + 0.25 * conformal_prob)

        tp1_r = self.policy_params["base_tp1_r"] * confidence_mult
        tp2_r = self.policy_params["base_tp2_r"] * conformal_mult

        # Adjust TP share based on confidence
        tp_share = self.policy_params["base_tp_share"]
        if confidence > 0.8:
            tp_share = 0.3  # Take less profit early when very confident
        elif confidence < 0.6:
            tp_share = 0.7  # Take more profit early when less confident

        # Choose trailing mode based on volatility proxy
        atr = features.get("atr", 50)
        close = features.get("close", 3400)
        volatility = atr / close if close > 0 else 0.01

        if volatility > 0.015:  # High volatility
            trailing = "chandelier"
        elif volatility > 0.008:  # Medium volatility
            trailing = "supertrend"
        else:  # Low volatility
            trailing = "off"

        # Position sizing factor
        if conformal_prob > 0.8 and confidence > 0.7:
            size_factor = self.policy_params["size_factors"]["aggressive"]
        elif conformal_prob > 0.6 and confidence > 0.6:
            size_factor = self.policy_params["size_factors"]["normal"]
        else:
            size_factor = self.policy_params["size_factors"]["conservative"]

        return {
            "tp1_r": tp1_r,
            "tp2_r": tp2_r,
            "tp_share": tp_share,
            "sl_mult": self.policy_params["base_sl_mult"],
            "trailing": trailing,
            "size_factor": size_factor,
        }

    def _rl_policy_decision(
        self, features: dict, layer1: dict, market: dict, conformal: dict
    ) -> dict:
        """RL-based policy decision (placeholder for future implementation)"""
        # This will be implemented when we add the RL training pipeline
        # For now, fall back to rule-based
        return self._rule_based_policy(features, layer1, market, conformal)

    def _adjust_for_regime(self, decision: dict, regime: str, market: dict) -> dict:
        """Adjust parameters based on market regime"""
        if not ENHANCED_AI_AVAILABLE:
            return decision

        try:
            regime_mult = get_regime_multipliers(regime)

            # Adjust TP/SL based on regime
            decision["tp1_r"] *= regime_mult.get("tp_multiplier", 1.0)
            decision["tp2_r"] *= regime_mult.get("tp_multiplier", 1.0)
            decision["sl_mult"] *= regime_mult.get("sl_multiplier", 1.0)
            decision["size_factor"] *= regime_mult.get("position_size", 1.0)

            # Adjust confidence boost
            decision["confidence"] *= regime_mult.get("confidence_boost", 1.0)
            decision["confidence"] = min(1.0, decision["confidence"])

            return decision

        except Exception as e:
            self.logger.warning(f"Regime adjustment error: {e}")
            return decision

    def _calculate_expected_value(self, decision: dict, prob_success: float) -> float:
        """Calculate expected value of the trade"""
        if decision["decision"] == "HOLD":
            return 0.0

        try:
            # Simplified EV calculation
            # EV = P(win) * Avg_Win - P(loss) * Avg_Loss

            tp1_r = decision.get("tp1_r", 0.8)
            tp2_r = decision.get("tp2_r", 1.5)
            tp_share = decision.get("tp_share", 0.5)
            sl_mult = decision.get("sl_mult", 1.2)

            # Expected win (weighted average of TP1 and TP2)
            expected_win = tp_share * tp1_r + (1 - tp_share) * tp2_r

            # Expected loss
            expected_loss = sl_mult

            # Probability of loss (inverse of conformal probability)
            prob_loss = 1.0 - prob_success

            # EV calculation
            ev = prob_success * expected_win - prob_loss * expected_loss

            return float(ev)

        except Exception as e:
            self.logger.warning(f"EV calculation error: {e}")
            return 0.0

    def _create_decision(self, decision: str, reason: str, confidence: float, ev: float) -> dict:
        """Create a standardized decision dictionary"""
        return {
            "decision": decision,
            "accept_reason": reason,
            "tp1_r": self.policy_params["base_tp1_r"],
            "tp2_r": self.policy_params["base_tp2_r"],
            "tp_share": self.policy_params["base_tp_share"],
            "sl_mult": self.policy_params["base_sl_mult"],
            "trailing": "off",
            "size_factor": 1.0,
            "confidence": confidence,
            "expected_value": ev,
        }

    def get_statistics(self) -> dict:
        """Get decision statistics"""
        total = max(1, self.stats["total_decisions"])
        return {
            "total_decisions": self.stats["total_decisions"],
            "acceptance_rate": self.stats["accepted"] / total,
            "conformal_reject_rate": self.stats["rejected_conformal"] / total,
            "risk_reject_rate": self.stats["rejected_risk"] / total,
            "regime_reject_rate": self.stats["rejected_regime"] / total,
        }

    def update_policy(self, trade_result: dict):
        """Update policy based on trade results (for online learning)"""
        # This will be implemented as part of the continual learning pipeline
        pass
