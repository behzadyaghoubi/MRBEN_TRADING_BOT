#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN AI Enhanced Live Trading System
Complete 3-Layer AI Architecture Implementation

Layer 1: Signal Engines (LSTM + ML Filter + Technical)
Layer 2: Meta Brain (Policy AI + Conformal + Dynamic TP/SL)
Layer 3: Safety & Governor (Risk limits + Kill-switch + Execution modes)
"""
import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# MT5 Integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader5 not available, using demo mode")
    MT5_AVAILABLE = False

# AI Models and Enhanced Components
try:
    from tensorflow.keras.models import load_model
    import joblib
    from ai_filter import AISignalFilter
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI models not available: {e}")
    AI_AVAILABLE = False

# Our enhanced AI architecture
try:
    from services.policy_brain import PolicyBrain
    from services.risk_governor import RiskGovernor, ExecutionMode
    from services.evaluator import PerformanceEvaluator
    from utils.regime import detect_regime
    from utils.conformal import ConformalGate
    ENHANCED_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced AI components not available: {e}")
    ENHANCED_AI_AVAILABLE = False

class EnhancedAIConfig:
    """Enhanced configuration for AI trading system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration with enhanced AI parameters"""
        default_config = {
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 5,
                "bars": 500,
                "sleep_seconds": 10,
                "retry_delay": 5,
                "consecutive_signals_required": 1
            },
            "ai_control": {
                "mode": "copilot",
                "conformal_alpha": 0.10,
                "emergency_stop_threshold": 0.02,
                "kill_switch_threshold": 0.05,
                "max_consecutive_losses": 5
            },
            "risk": {
                "fixed_volume": 0.01,
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10,
                "base_risk": 0.005
            },
            "logging": {
                "logs_dir": "logs"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove JSON comments
                    import re
                    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                    config = json.loads(content)
                
                # Merge configurations
                for key, value in config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
        
        # Set attributes
        for section, params in default_config.items():
            if isinstance(params, dict):
                for param, value in params.items():
                    setattr(self, param.upper(), value)
            else:
                setattr(self, section.upper(), params)

class EnhancedSignalGenerator:
    """Enhanced signal generator combining all Layer 1 engines"""
    
    def __init__(self, config, lstm_model, lstm_scaler, ml_filter):
        self.config = config
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler
        self.ml_filter = ml_filter
        self.logger = logging.getLogger("SignalGenerator")
    
    def generate_enhanced_signal(self, df: pd.DataFrame) -> Dict:
        """Generate signal from all Layer 1 engines"""
        try:
            # LSTM Signal
            lstm_signal = self._lstm_prediction(df)
            
            # ML Filter Signal
            ml_signal = self._ml_filter_prediction(df)
            
            # Technical Analysis Signal
            tech_signal = self._technical_analysis(df)
            
            # Ensemble combination
            ensemble_signal = self._combine_signals(lstm_signal, ml_signal, tech_signal)
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal: {e}")
            return {'signal': 0, 'confidence': 0.0, 'source': 'Error'}
    
    def _lstm_prediction(self, df: pd.DataFrame) -> Dict:
        """LSTM model prediction"""
        if self.lstm_model is None or len(df) < 60:
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0}
        
        try:
            # Prepare LSTM input
            features = ['close', 'high', 'low', 'open', 'rsi', 'macd', 'atr']
            available_features = [f for f in features if f in df.columns]
            
            data = df[available_features].tail(60).values
            
            if self.lstm_scaler:
                data = self.lstm_scaler.transform(data)
            
            X = data.reshape(1, 60, len(available_features))
            
            # Prediction
            pred = self.lstm_model.predict(X, verbose=0)[0][0]
            
            # Convert to signal
            if pred > 0.6:
                signal = 1
                confidence = pred
            elif pred < 0.4:
                signal = -1
                confidence = 1 - pred
            else:
                signal = 0
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'score': float(pred - 0.5) * 2  # -1 to 1 scale
            }
            
        except Exception as e:
            self.logger.warning(f"LSTM prediction error: {e}")
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0}
    
    def _ml_filter_prediction(self, df: pd.DataFrame) -> Dict:
        """ML Filter prediction"""
        if self.ml_filter is None:
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0}
        
        try:
            # Use ML filter
            result = self.ml_filter.should_trade(df)
            
            signal = result.get('signal', 0)
            confidence = result.get('probability', 0.5)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': (confidence - 0.5) * 2 * signal if signal != 0 else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"ML Filter prediction error: {e}")
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0}
    
    def _technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Technical analysis signal"""
        try:
            # Simple technical analysis
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            
            # RSI signal
            rsi = last.get('rsi', 50)
            rsi_signal = 1 if rsi < 30 else -1 if rsi > 70 else 0
            
            # MACD signal
            macd = last.get('macd', 0)
            macd_signal_line = last.get('macd_signal', 0)
            macd_signal = 1 if macd > macd_signal_line else -1
            
            # Price momentum
            price_change = (last['close'] - prev['close']) / prev['close']
            momentum_signal = 1 if price_change > 0.001 else -1 if price_change < -0.001 else 0
            
            # Combine technical signals
            signals = [rsi_signal, macd_signal, momentum_signal]
            avg_signal = sum(signals) / len(signals)
            
            if avg_signal > 0.33:
                signal = 1
                confidence = min(0.8, 0.5 + abs(avg_signal) * 0.3)
            elif avg_signal < -0.33:
                signal = -1
                confidence = min(0.8, 0.5 + abs(avg_signal) * 0.3)
            else:
                signal = 0
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': avg_signal
            }
            
        except Exception as e:
            self.logger.warning(f"Technical analysis error: {e}")
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0}
    
    def _combine_signals(self, lstm: Dict, ml: Dict, tech: Dict) -> Dict:
        """Combine all signals into ensemble"""
        
        # Weights for ensemble
        weights = [0.5, 0.3, 0.2]  # LSTM, ML, Technical
        
        # Weighted signal
        signals = [lstm['signal'], ml['signal'], tech['signal']]
        confidences = [lstm['confidence'], ml['confidence'], tech['confidence']]
        scores = [lstm['score'], ml['score'], tech['score']]
        
        # Ensemble signal
        weighted_signal = sum(s * w for s, w in zip(signals, weights))
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        # Final decision
        if weighted_signal > 0.3:
            final_signal = 1
        elif weighted_signal < -0.3:
            final_signal = -1
        else:
            final_signal = 0
        
        return {
            'signal': final_signal,
            'confidence': weighted_confidence,
            'score': weighted_score,
            'source': 'LSTM_ML_Technical_Ensemble',
            'components': {
                'lstm': lstm,
                'ml': ml,
                'technical': tech
            }
        }

class EnhancedAILiveTrader:
    """
    Complete 3-Layer AI Enhanced Live Trading System
    """
    
    def __init__(self, config_path: str = "config.json"):
        # Configuration
        self.config = EnhancedAIConfig(config_path)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize Layer 1: Signal Engines
        self._init_layer1_engines()
        
        # Initialize Layer 2: Meta Brain
        self._init_layer2_brain()
        
        # Initialize Layer 3: Risk Governor
        self._init_layer3_governor()
        
        # Initialize Evaluation System
        self._init_evaluation_system()
        
        # AI Control Mode
        self.ai_control = self.config.config_data.get("ai_control", {})
        self.ai_mode = self.ai_control.get("mode", "shadow").lower()  # "shadow" | "copilot" | "autopilot"
        
        # Trading state
        self.running = False
        self.trading_thread = None
        self.consecutive_signals = 0
        self.last_signal = 0
        
        # Performance tracking
        self.session_stats = {
            "start_time": datetime.now(),
            "signals_generated": 0,
            "trades_executed": 0,
            "conformal_accepted": 0,
            "conformal_rejected": 0,
            "risk_rejected": 0
        }
        
        self.logger.info("🤖 Enhanced AI Live Trading System initialized")
        self.logger.info(f"📊 Execution Mode: {self.risk_governor.get_execution_mode().value}")
        
    def _setup_logger(self):
        """Setup comprehensive logging"""
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        logger = logging.getLogger('EnhancedAITrader')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(self.config.LOGS_DIR, 'enhanced_ai_trader.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_layer1_engines(self):
        """Initialize Layer 1: Signal Engines"""
        self.logger.info("🔧 Initializing Layer 1: Signal Engines...")
        
        # Load LSTM model
        self.lstm_model, self.lstm_scaler = self._load_lstm_model()
        
        # Load ML filter
        self.ml_filter = self._load_ml_filter()
        
        # Initialize signal generator
        self.signal_generator = EnhancedSignalGenerator(
            self.config, self.lstm_model, self.lstm_scaler, self.ml_filter
        )
        
        self.logger.info("✅ Layer 1: Signal Engines initialized")
    
    def _init_layer2_brain(self):
        """Initialize Layer 2: Meta Brain"""
        self.logger.info("🧠 Initializing Layer 2: Meta Brain...")
        
        if ENHANCED_AI_AVAILABLE:
            self.policy_brain = PolicyBrain(self.config.config_path)
            
            # Log conformal parameters if available
            if self.policy_brain and hasattr(self.policy_brain, 'conformal_gate') and self.policy_brain.conformal_gate:
                if hasattr(self.policy_brain.conformal_gate, 'conf') and self.policy_brain.conformal_gate.conf:
                    thr = float(self.policy_brain.conformal_gate.conf.get("nonconf_threshold", 0.0))
                    self.logger.info(f"✅ Conformal loaded. accept if p >= {1.0 - thr:.3f}")
                else:
                    self.logger.warning("⚠️ Conformal gate loaded but no threshold configuration found")
            else:
                self.logger.warning("⚠️ Conformal gate not available in PolicyBrain")
        else:
            self.policy_brain = None
            self.logger.warning("⚠️ Layer 2: PolicyBrain not available, using simplified logic")
        
        self.logger.info("✅ Layer 2: Meta Brain initialized")
    
    def _init_layer3_governor(self):
        """Initialize Layer 3: Risk Governor"""
        self.logger.info("🛡️ Initializing Layer 3: Risk Governor...")
        
        if ENHANCED_AI_AVAILABLE:
            self.risk_governor = RiskGovernor(self.config.config_path)
        else:
            self.risk_governor = None
            self.logger.warning("⚠️ Layer 3: RiskGovernor not available, using basic risk management")
        
        self.logger.info("✅ Layer 3: Risk Governor initialized")
    
    def _init_evaluation_system(self):
        """Initialize Performance Evaluation System"""
        self.logger.info("📊 Initializing Evaluation System...")
        
        if ENHANCED_AI_AVAILABLE:
            self.evaluator = PerformanceEvaluator(self.config.config_path)
        else:
            self.evaluator = None
            self.logger.warning("⚠️ Evaluation System not available")
        
        self.logger.info("✅ Evaluation System initialized")
    
    def _load_lstm_model(self):
        """Load LSTM model and scaler"""
        try:
            model_path = 'models/lstm_balanced_model.h5'
            scaler_path = 'models/lstm_balanced_scaler.save'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                lstm_model = load_model(model_path)
                lstm_scaler = joblib.load(scaler_path)
                self.logger.info("✅ LSTM Model loaded successfully")
                return lstm_model, lstm_scaler
            else:
                self.logger.warning("⚠️ LSTM model files not found")
                return None, None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading LSTM model: {e}")
            return None, None
    
    def _load_ml_filter(self):
        """Load ML filter"""
        try:
            ml_filter_path = 'models/mrben_ai_signal_filter_xgb.joblib'
            
            if os.path.exists(ml_filter_path):
                ml_filter = AISignalFilter(model_path=ml_filter_path)
                self.logger.info("✅ ML Filter loaded successfully")
                return ml_filter
            else:
                self.logger.warning("⚠️ ML Filter not found")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading ML Filter: {e}")
            return None
    
    def start(self):
        """Start the enhanced AI trading system"""
        self.logger.info("🚀 Starting Enhanced AI Trading System...")
        
        # Log system configuration
        self._log_system_config()
        
        self.running = True
        
        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self._enhanced_trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info("✅ Enhanced AI Trading System started successfully!")
    
    def stop(self):
        """Stop the trading system"""
        self.logger.info("🛑 Stopping Enhanced AI Trading System...")
        self.running = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        # Generate final report
        self._generate_session_report()
        
        self.logger.info("✅ Enhanced AI Trading System stopped")
    
    def _log_system_config(self):
        """Log current system configuration"""
        self.logger.info("=" * 60)
        self.logger.info("🤖 ENHANCED AI TRADING SYSTEM CONFIGURATION")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Symbol: {self.config.SYMBOL}")
        self.logger.info(f"📊 Timeframe: M{self.config.TIMEFRAME}")
        self.logger.info(f"📊 Volume: {self.config.FIXED_VOLUME}")
        
        if self.risk_governor:
            mode = self.risk_governor.get_execution_mode().value
            self.logger.info(f"🎛️ Execution Mode: {mode.upper()}")
            
        self.logger.info(f"🧠 Layer 1: {'✅' if self.lstm_model else '❌'} LSTM + {'✅' if self.ml_filter else '❌'} ML Filter")
        self.logger.info(f"🧠 Layer 2: {'✅' if self.policy_brain else '❌'} Policy Brain")
        self.logger.info(f"🛡️ Layer 3: {'✅' if self.risk_governor else '❌'} Risk Governor")
        self.logger.info(f"📊 Evaluator: {'✅' if self.evaluator else '❌'} Performance Tracking")
        self.logger.info("=" * 60)
    
    def _enhanced_trading_loop(self):
        """Enhanced AI trading loop with 3-layer architecture"""
        
        while self.running:
            try:
                # Get market data
                df = self._get_market_data()
                
                if df is None or len(df) < 50:
                    self.logger.warning("Insufficient market data, skipping iteration")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                
                # LAYER 1: Generate signals from all engines
                layer1_signal = self.signal_generator.generate_enhanced_signal(df)
                
                # Build snapshot for Layer 2 & 3
                snapshot = self._build_market_snapshot(df, layer1_signal)
                
                # LAYER 2: Policy Brain decision
                if self.policy_brain:
                    brain_decision = self.policy_brain.propose(snapshot)
                else:
                    brain_decision = self._fallback_decision(layer1_signal)
                
                # Track conformal acceptance
                if brain_decision["accept_reason"] == "conformal_reject":
                    self.session_stats["conformal_rejected"] += 1
                else:
                    self.session_stats["conformal_accepted"] += 1
                
                # LAYER 3: Risk Governor validation
                if self.risk_governor and brain_decision["decision"] != "HOLD":
                    governor_result = self.risk_governor.validate_trade(brain_decision, snapshot["market"])
                    
                    if not governor_result["approved"]:
                        self.session_stats["risk_rejected"] += 1
                        self.logger.info(f"🛡️ Risk Governor REJECTED: {governor_result['reason']}")
                        brain_decision = {"decision": "HOLD", "accept_reason": governor_result["reason"]}
                
                # Log comprehensive decision
                self._log_ai_decision(layer1_signal, brain_decision, snapshot)
                
                # Statistics tracking
                self.session_stats["signals_generated"] += 1
                if brain_decision["accept_reason"] != "conformal_reject":
                    self.session_stats["conformal_accepted"] += 1
                else:
                    self.session_stats["conformal_rejected"] += 1
                
                # Execute based on AI mode
                should_execute = brain_decision["decision"] != "HOLD"
                
                if should_execute:
                    # اگر Shadow است، فقط لاگ کنیم و نرویم
                    if self.ai_mode == "shadow":
                        self.session_stats["would_exec"] += 1
                        # همان TPهایی که در _execute_trade می‌فرستی را اینجا محاسبه کن برای لاگ:
                        is_buy = (layer1_signal['signal'] == 1)
                        entry_price = float(df['close'].iloc[-1])
                        
                        # Get tick data if available
                        tick = self.data_manager.get_current_tick()
                        if tick:
                            entry_price = tick['ask'] if is_buy else tick['bid']
                        
                        # Calculate SL/TP using simplified logic for shadow
                        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 50.0
                        sl_mult = brain_decision.get("sl_mult", 1.2)
                        risk = atr * sl_mult
                        sl_price = entry_price - (risk if is_buy else -risk)
                        
                        volume_total = self._volume_for_trade(entry_price, sl_price)
                        
                        # Get TP policy from brain decision or config
                        tp_policy = brain_decision.get("tp_policy", {
                            "split": True,
                            "tp1_r": 0.8,
                            "tp2_r": 1.5,
                            "tp1_share": 0.5
                        })
                        
                        if tp_policy.get("split", True):
                            tp1 = self._r_to_price(entry_price, sl_price, layer1_signal['signal'], tp_policy.get("tp1_r", 0.8))
                            tp2 = self._r_to_price(entry_price, sl_price, layer1_signal['signal'], max(tp_policy.get("tp2_r", 1.5), 1.2))
                            v1 = round(volume_total * tp_policy.get("tp1_share", 0.5), 2)
                            v2 = max(0.01, round(volume_total - v1, 2))  # Minimum lot size
                            
                            self._log_shadow("BUY" if is_buy else "SELL", entry_price, sl_price,
                                             {"split": True, "tp1": tp1, "tp2": tp2, "vol1": v1, "vol2": v2})
                        else:
                            tp_price = self._r_to_price(entry_price, sl_price, layer1_signal['signal'], 1.5)
                            self._log_shadow("BUY" if is_buy else "SELL", entry_price, sl_price,
                                             {"split": False, "tp1": tp_price})
                        continue  # مهم: اجرا نکن
                    
                    # Co-pilot & Autopilot: اجرا
                    elif self.ai_mode in ["copilot", "autopilot"]:
                        self._execute_enhanced_trade(brain_decision, df, snapshot)
                        self.session_stats["trades_executed"] += 1
                        self.logger.info(f"✅ Trade executed in {self.ai_mode.upper()} mode")
                
                # Periodic statistics reporting
                if (datetime.now() - self.session_stats.get("last_stats_log", datetime.now())).total_seconds() >= 600:  # هر 10 دقیقه
                    s = self.session_stats
                    seen = s.get("signals_generated", 0)
                    accepted = s.get("conformal_accepted", 0)
                    risk_reject = s.get("risk_rejected", 0)
                    would_exec = s.get("would_exec", 0)
                    
                    ar = (accepted / seen) if seen else 0.0
                    wr = (would_exec / seen) if seen else 0.0
                    rr = (risk_reject / seen) if seen else 0.0
                    
                    self.logger.info(f"[STATS] seen={seen} accept_rate={ar:.2f} risk_reject_rate={rr:.2f} would_exec_rate={wr:.2f}")
                    s["last_stats_log"] = datetime.now()
                
                # --- Breakeven after TP1 for split positions ---
                self._check_breakeven_after_tp1()
                
                # Sleep before next iteration
                time.sleep(self.config.SLEEP_SECONDS)
                
            except Exception as e:
                self.logger.error(f"Error in enhanced trading loop: {e}")
                time.sleep(self.config.RETRY_DELAY)
    
    def _build_market_snapshot(self, df: pd.DataFrame, layer1_signal: Dict) -> Dict:
        """Build comprehensive market snapshot for AI layers"""
        
        last_row = df.iloc[-1]
        
        # Extract features
        features = {
            "close": float(last_row['close']),
            "ret": float(df['close'].pct_change().iloc[-1]) if len(df) > 1 else 0.0,
            "sma_20": float(last_row.get('sma_20', 0.0)),
            "sma_50": float(last_row.get('sma_50', 0.0)),
            "atr": float(last_row.get('atr', 0.0)),
            "rsi": float(last_row.get('rsi', 50.0)),
            "macd": float(last_row.get('macd', 0.0)),
            "macd_signal": float(last_row.get('macd_signal', 0.0)),
            "hour": float(last_row['time'].hour if 'time' in df.columns else datetime.now().hour),
            "dow": float(last_row['time'].dayofweek if 'time' in df.columns else datetime.now().weekday())
        }
        
        # Detect regime
        regime = "UNKNOWN"
        if ENHANCED_AI_AVAILABLE:
            try:
                regime = detect_regime(last_row)
            except:
                regime = "UNKNOWN"
        
        # Market conditions
        market = {
            "spread": 30,  # Simplified
            "session": "london",  # Simplified
            "regime": regime
        }
        
        # Context
        context = {
            "daily_pnl": 0.0,  # Would need real tracking
            "trades_today": self.session_stats["trades_executed"],
            "equity": 10000.0  # Simplified
        }
        
        return {
            "features": features,
            "layer1": layer1_signal,
            "market": market,
            "context": context
        }
    
    def _fallback_decision(self, layer1_signal: Dict) -> Dict:
        """Fallback decision when PolicyBrain is not available"""
        signal = layer1_signal.get("signal", 0)
        confidence = layer1_signal.get("confidence", 0.0)
        
        if signal != 0 and confidence > 0.6:
            decision = "BUY" if signal == 1 else "SELL"
            return {
                "decision": decision,
                "accept_reason": "fallback_high_confidence",
                "tp1_r": 0.8,
                "tp2_r": 1.5,
                "tp_share": 0.5,
                "sl_mult": 1.2,
                "trailing": "off",
                "size_factor": 1.0,
                "confidence": confidence,
                "expected_value": 0.0
            }
        else:
            return {
                "decision": "HOLD",
                "accept_reason": "fallback_low_confidence",
                "confidence": confidence,
                "expected_value": 0.0
            }
    
    def _log_ai_decision(self, layer1: Dict, brain_decision: Dict, snapshot: Dict):
        """Log comprehensive AI decision process"""
        
        signal_strength = "🟢 STRONG" if layer1["confidence"] > 0.7 else "🟡 MEDIUM" if layer1["confidence"] > 0.5 else "🔴 WEAK"
        
        self.logger.info(
            f"🤖 AI DECISION | L1: {layer1['signal']} ({signal_strength}) | "
            f"L2: {brain_decision['decision']} | Reason: {brain_decision['accept_reason']} | "
            f"Regime: {snapshot['market']['regime']} | "
            f"EV: {brain_decision.get('expected_value', 0.0):.3f}"
        )
        
        # Detailed component breakdown
        if 'components' in layer1:
            comp = layer1['components']
            self.logger.debug(
                f"📊 Layer 1 Components: LSTM({comp['lstm']['signal']},{comp['lstm']['confidence']:.2f}) | "
                f"ML({comp['ml']['signal']},{comp['ml']['confidence']:.2f}) | "
                f"Tech({comp['technical']['signal']},{comp['technical']['confidence']:.2f})"
            )
    
    def _r_to_price(self, entry_price: float, sl_price: float, signal: int, R: float) -> float:
        """Convert R-multiple to price"""
        risk = abs(entry_price - sl_price)
        side = 1 if signal == 1 else -1
        return entry_price + (side * risk * R)
    
    def _volume_for_trade(self, entry_price: float, sl_price: float) -> float:
        """Calculate volume for trade based on risk"""
        # Simplified volume calculation - should be enhanced based on actual risk management
        return getattr(self.config, 'FIXED_VOLUME', 0.1)
    
    def _log_shadow(self, side: str, entry_price: float, sl_price: float, tp_info: dict):
        """
        Log shadow execution with proper TP details
        tp_info: {"split": bool, "tp1": float|None, "tp2": float|None, "vol1": float|None, "vol2": float|None}
        """
        if tp_info.get("split", False):
            self.logger.info(
                f"[SHADOW] Would execute {side} px={entry_price:.2f} SL={sl_price:.2f} "
                f"TP1={tp_info['tp1']:.2f}(v={tp_info['vol1']}) TP2={tp_info['tp2']:.2f}(v={tp_info['vol2']})"
            )
        else:
            self.logger.info(
                f"[SHADOW] Would execute {side} px={entry_price:.2f} SL={sl_price:.2f} TP={tp_info['tp1']:.2f}"
            )
    
    def _log_shadow_execution(self, decision: Dict, df: pd.DataFrame, snapshot: Dict):
        """Log what would be executed in Shadow mode - Legacy method"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 50
            side = 1 if decision["decision"] == "BUY" else -1
            
            entry_price = current_price
            risk = atr * decision.get("sl_mult", 1.2)
            sl_price = entry_price - (side * risk)
            tp1_price = entry_price + (side * risk * decision.get("tp1_r", 0.8))
            tp2_price = entry_price + (side * risk * decision.get("tp2_r", 1.5))
            
            base_volume = self.config.FIXED_VOLUME
            size_factor = decision.get("size_factor", 1.0)
            adjusted_volume = base_volume * size_factor
            
            self.logger.info(
                f"[SHADOW] Would execute {decision['decision']} at px={entry_price:.2f} "
                f"SL={sl_price:.2f} TP1={tp1_price:.2f} TP2={tp2_price:.2f} "
                f"Vol={adjusted_volume:.3f} Conf={decision.get('confidence', 0.0):.3f} "
                f"EV={decision.get('expected_value', 0.0):.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in shadow execution logging: {e}")
    
    def _check_breakeven_after_tp1(self):
        """Check and apply breakeven for remaining positions after TP1 hit"""
        try:
            # Only proceed if MT5 is available and we're using split TP
            if not MT5_AVAILABLE:
                return
                
            tp_policy = self.ai_control.get("tp_policy", {})
            if not tp_policy.get("split", False) or not tp_policy.get("breakeven_after_tp1", True):
                return
            
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=self.config.SYMBOL) or []
            
            # اگر دقیقاً 1 پوزیشن باز مانده و نوعش با آخرین سیگنال اخیر هم‌راستاست، SL را BE کن
            if len(positions) == 1:
                p = positions[0]
                entry_like = float(p.price_open)
                # فقط اگر SL پایین‌تر (برای BUY) یا بالاتر (برای SELL) از BE است، جابجا کن
                is_buy = (p.type == 0)  # 0 = BUY, 1 = SELL
                need_be = False
                
                if is_buy and (p.sl is None or p.sl < entry_like):
                    need_be = True
                elif not is_buy and (p.sl is None or p.sl > entry_like):
                    need_be = True
                
                if need_be:
                    # Modify stop loss to breakeven
                    modify_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": self.config.SYMBOL,
                        "position": int(p.ticket),
                        "sl": float(entry_like),
                        "tp": float(p.tp) if p.tp else 0.0
                    }
                    
                    result = mt5.order_send(modify_request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(f"✅ BE applied to position #{p.ticket} at {entry_like:.2f}")
                    else:
                        self.logger.warning(f"⚠️ Failed to apply BE to position #{p.ticket}: {result.comment if result else 'Unknown error'}")
                        
        except Exception as e:
            self.logger.error(f"Error in breakeven check: {e}")
    
    def _execute_enhanced_trade(self, decision: Dict, df: pd.DataFrame, snapshot: Dict):
        """Execute trade with enhanced AI parameters"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate enhanced TP/SL using AI decision
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 50
            side = 1 if decision["decision"] == "BUY" else -1
            
            entry_price = current_price
            risk = atr * decision.get("sl_mult", 1.2)
            sl_price = entry_price - (side * risk)
            
            tp1_price = entry_price + (side * risk * decision.get("tp1_r", 0.8))
            tp2_price = entry_price + (side * risk * decision.get("tp2_r", 1.5))
            
            # Calculate position size with AI factor
            base_volume = self.config.FIXED_VOLUME
            size_factor = decision.get("size_factor", 1.0)
            adjusted_volume = base_volume * size_factor
            
            # Start trade tracking if evaluator available
            trade_id = None
            if self.evaluator:
                trade_data = {
                    "symbol": self.config.SYMBOL,
                    "side": decision["decision"],
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp1_price": tp1_price,
                    "tp2_price": tp2_price,
                    "volume": adjusted_volume,
                    "confidence": decision.get("confidence", 0.0),
                    "regime": snapshot["market"]["regime"],
                    "conformal_prob": decision.get("conformal_prob", 0.5)
                }
                trade_id = self.evaluator.start_trade_tracking(trade_data)
            
            # Log enhanced execution
            self.logger.info(
                f"🎯 ENHANCED EXECUTION: {decision['decision']} {self.config.SYMBOL} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | "
                f"TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f} | "
                f"Vol: {adjusted_volume:.2f} (x{size_factor:.1f}) | "
                f"Trail: {decision.get('trailing', 'off')} | "
                f"Confidence: {decision.get('confidence', 0.0):.3f} | "
                f"EV: {decision.get('expected_value', 0.0):.3f}"
            )
            
            # Update risk governor state
            if self.risk_governor:
                self.risk_governor.update_session_state({"executed": True, "pnl": 0.0})
            
            # Save trade log (simplified)
            self._save_enhanced_trade_log(decision, entry_price, sl_price, tp1_price, tp2_price, 
                                        adjusted_volume, snapshot, trade_id)
            
        except Exception as e:
            self.logger.error(f"Error executing enhanced trade: {e}")
    
    def _save_enhanced_trade_log(self, decision: Dict, entry: float, sl: float, 
                               tp1: float, tp2: float, volume: float, 
                               snapshot: Dict, trade_id: Optional[str]):
        """Save comprehensive trade log"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        trade_log = {
            'timestamp': timestamp,
            'trade_id': trade_id,
            'symbol': self.config.SYMBOL,
            'action': decision["decision"],
            'entry_price': entry,
            'sl_price': sl,
            'tp1_price': tp1,
            'tp2_price': tp2,
            'volume': volume,
            'confidence': decision.get("confidence", 0.0),
            'expected_value': decision.get("expected_value", 0.0),
            'regime': snapshot["market"]["regime"],
            'trailing_mode': decision.get("trailing", "off"),
            'size_factor': decision.get("size_factor", 1.0),
            'conformal_prob': decision.get("conformal_prob", 0.5),
            'ai_decision_reason': decision.get("accept_reason", "unknown"),
            'layer1_components': snapshot["layer1"].get("components", {}),
            'execution_mode': self.risk_governor.get_execution_mode().value if self.risk_governor else "unknown"
        }
        
        # Save to CSV
        trade_file = os.path.join(self.config.LOGS_DIR, 'enhanced_ai_trades.csv')
        trade_df = pd.DataFrame([trade_log])
        
        if os.path.exists(trade_file):
            trade_df.to_csv(trade_file, mode='a', header=False, index=False)
        else:
            trade_df.to_csv(trade_file, index=False)
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data (simplified implementation)"""
        # This would normally get real MT5 data
        # For now, return synthetic data for testing
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        np.random.seed(42)
        
        base_price = 3400
        returns = np.random.normal(0, 0.001, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices
        })
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = 50 + 50 * (df['close'].pct_change().rolling(14).mean() / 
                               (df['close'].pct_change().rolling(14).std() + 1e-9))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['atr'] = df['close'].diff().abs().rolling(14).mean() * 3.0
        
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def _generate_session_report(self):
        """Generate comprehensive session report"""
        
        duration = datetime.now() - self.session_stats["start_time"]
        
        self.logger.info("=" * 60)
        self.logger.info("📊 ENHANCED AI TRADING SESSION REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Session Duration: {duration}")
        self.logger.info(f"📡 Signals Generated: {self.session_stats['signals_generated']}")
        self.logger.info(f"🎯 Trades Executed: {self.session_stats['trades_executed']}")
        self.logger.info(f"✅ Conformal Accepted: {self.session_stats['conformal_accepted']}")
        self.logger.info(f"❌ Conformal Rejected: {self.session_stats['conformal_rejected']}")
        self.logger.info(f"🛡️ Risk Rejected: {self.session_stats['risk_rejected']}")
        
        # Performance metrics from evaluator
        if self.evaluator:
            metrics = self.evaluator.calculate_performance_metrics()
            if "error" not in metrics:
                self.logger.info(f"📈 Win Rate: {metrics.get('win_rate', 0.0):.1%}")
                self.logger.info(f"💰 Total PnL: {metrics.get('total_pnl', 0.0):.2f}")
                self.logger.info(f"📊 Avg R-Multiple: {metrics.get('avg_r_multiple', 0.0):.2f}")
                self.logger.info(f"🎯 MFE/MAE Ratio: {metrics.get('mfe_mae_ratio', 0.0):.2f}")
        
        # AI system statistics
        if self.policy_brain:
            brain_stats = self.policy_brain.get_statistics()
            self.logger.info(f"🧠 Brain Acceptance Rate: {brain_stats.get('acceptance_rate', 0.0):.1%}")
        
        if self.risk_governor:
            governor_status = self.risk_governor.get_status_report()
            self.logger.info(f"🛡️ Emergency Stop: {governor_status.get('emergency_stop', False)}")
            self.logger.info(f"🔴 Kill Switch: {governor_status.get('kill_switch_active', False)}")
        
        self.logger.info("=" * 60)

def main():
    """Main function"""
    print("🤖 MR BEN Enhanced AI Trading System")
    print("=" * 50)
    
    # Create and start enhanced AI trader
    trader = EnhancedAILiveTrader()
    
    try:
        trader.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal...")
        trader.stop()
        print("✅ Enhanced AI System shutdown complete")

if __name__ == "__main__":
    main()
