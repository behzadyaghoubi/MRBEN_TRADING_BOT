"""
Rule-Based Strategy for MR BEN Pro Strategy

Professional trading rules based on market structure and technical analysis:
- Trend Continuation (TC) patterns
- Breakout-Retest (BR) patterns
- Reversal (RV) patterns
- Support/Resistance confluence
- Multiple timeframe confirmation
- Price Action integration
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import TechnicalIndicators, atr, bollinger_bands, ema, rsi
from .pa import PAResult, PriceActionValidator
from .structure import MarketStructure

warnings.filterwarnings('ignore')


@dataclass
class RuleDecision:
    """Decision from rule-based strategy evaluation"""

    side: int  # +1 buy, -1 sell, 0 none
    score: float  # 0-1 confidence score
    tags: list[str]  # Rule tags for identification
    context: dict  # Detailed context (trend, bos, distances, etc.)
    rule_type: str  # 'TC', 'BR', 'RV', or 'MIXED'
    pa_confirmation: list[PAResult] | None = None  # Price Action confirmation


class RuleBasedStrategy:
    """Professional rule-based trading strategy with Price Action integration"""

    def __init__(self, config: dict):
        """
        Initialize the rule-based strategy

        Args:
            config: Configuration dictionary with strategy parameters
        """
        self.config = config
        self.indicators = TechnicalIndicators()

        # Strategy parameters
        self.ema_fast = config.get('ema_fast', 20)
        self.ema_slow = config.get('ema_slow', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.atr_period = config.get('atr_period', 14)
        self.min_pullback_atr = config.get('min_pullback_atr', 0.5)
        self.max_pullback_atr = config.get('max_pullback_atr', 2.0)
        self.min_rejection_atr = config.get('min_rejection_atr', 0.3)

        # Price Action integration parameters
        self.pa_enabled = config.get('pa_enabled', True)
        self.pa_weight = config.get('pa_weight', 0.3)  # Weight of PA in final scoring
        self.min_pa_confidence = config.get('min_pa_confidence', 0.6)

        # Initialize Price Action validator
        if self.pa_enabled:
            pa_config = config.get('pa_config', {})
            self.pa_validator = PriceActionValidator(pa_config)
        else:
            self.pa_validator = None

    def evaluate_rules(self, df: pd.DataFrame, structure: MarketStructure) -> RuleDecision:
        """
        Evaluate all trading rules and return the best decision

        Args:
            df: Price data DataFrame
            structure: Market structure analysis

        Returns:
            RuleDecision with best trading opportunity
        """
        # Calculate technical indicators
        df = self._add_indicators(df)

        # Evaluate different rule types
        tc_decision = self._evaluate_trend_continuation(df, structure)
        br_decision = self._evaluate_breakout_retest(df, structure)
        rv_decision = self._evaluate_reversal(df, structure)

        # Combine decisions and select best
        decisions = [d for d in [tc_decision, br_decision, rv_decision] if d.side != 0]

        if not decisions:
            return RuleDecision(
                side=0,
                score=0.0,
                tags=['NO_SIGNAL'],
                context={'reason': 'No rule conditions met'},
                rule_type='NONE',
            )

        # Select decision with highest score
        best_decision = max(decisions, key=lambda x: x.score)

        # Add market structure context
        best_decision.context.update(
            {
                'trend': structure.last_trend,
                'trend_strength': structure.trend_strength,
                'structure_quality': structure.structure_quality,
                'last_bos': structure.last_bos,
                'last_choch': structure.last_choch,
            }
        )

        # Integrate Price Action validation if enabled
        if self.pa_enabled and self.pa_validator:
            best_decision = self._integrate_price_action(best_decision, df, structure)

        return best_decision

    def _integrate_price_action(
        self, decision: RuleDecision, df: pd.DataFrame, structure: MarketStructure
    ) -> RuleDecision:
        """Integrate Price Action validation with rule decision"""
        try:
            # Get Price Action patterns
            pa_patterns = self.pa_validator.validate_price_action(df)

            # Filter patterns by direction and confidence
            relevant_patterns = []
            for pattern in pa_patterns:
                if (
                    (pattern.direction == "BULLISH" and decision.side == 1)
                    or (pattern.direction == "BEARISH" and decision.side == -1)
                    or (pattern.direction == "NEUTRAL")
                ):
                    if pattern.confidence >= self.min_pa_confidence:
                        relevant_patterns.append(pattern)

            # Add volume confirmation
            if relevant_patterns:
                relevant_patterns = self.pa_validator.add_volume_confirmation(relevant_patterns, df)

            # Calculate PA-enhanced score
            if relevant_patterns:
                pa_score = self._calculate_pa_enhanced_score(decision, relevant_patterns)
                decision.score = pa_score
                decision.pa_confirmation = relevant_patterns
                decision.tags.extend(['PA_CONFIRMED'])

                # Add PA context
                decision.context['pa_patterns'] = len(relevant_patterns)
                decision.context['pa_confidence'] = np.mean(
                    [p.confidence for p in relevant_patterns]
                )
                decision.context['pa_strength'] = np.mean([p.strength for p in relevant_patterns])

        except Exception as e:
            # Log error but continue without PA integration
            decision.context['pa_error'] = str(e)

        return decision

    def _calculate_pa_enhanced_score(
        self, decision: RuleDecision, pa_patterns: list[PAResult]
    ) -> float:
        """Calculate PA-enhanced score"""
        if not pa_patterns:
            return decision.score

        # Calculate PA contribution
        pa_confidence = np.mean([p.confidence for p in pa_patterns])
        pa_strength = np.mean([p.strength for p in pa_patterns])
        volume_bonus = 0.1 if any(p.volume_confirmed for p in pa_patterns) else 0.0

        # Combine original score with PA enhancement
        pa_contribution = (pa_confidence + pa_strength + volume_bonus) / 3
        enhanced_score = decision.score * (1 - self.pa_weight) + pa_contribution * self.pa_weight

        return min(max(enhanced_score, 0.0), 1.0)

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame"""
        # Moving averages
        df['ema_fast'] = ema(df['close'], self.ema_fast)
        df['ema_slow'] = ema(df['close'], self.ema_slow)

        # Momentum
        df['rsi'] = rsi(df['close'], self.rsi_period)

        # Volatility
        df['atr'] = atr(df['high'], df['low'], df['close'], self.atr_period)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])

        # Price relationships
        df['price_vs_ema_fast'] = (df['close'] - df['ema_fast']) / df['ema_fast']
        df['price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['ema_slow']
        df['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']

        return df

    def _evaluate_trend_continuation(
        self, df: pd.DataFrame, structure: MarketStructure
    ) -> RuleDecision:
        """
        Evaluate Trend Continuation (TC) patterns with enhanced PA integration

        Pattern: trend=UP & pullback to EMA & bullish rejection → BUY
        Pattern: trend=DOWN & pullback to EMA & bearish rejection → SELL
        """
        if len(df) < 3:
            return RuleDecision(0, 0.0, [], {}, 'TC')

        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]

        # Check trend conditions
        trend_up = (
            structure.last_trend == 'UP' and structure.trend_strength > 0.6 and ema_fast > ema_slow
        )

        trend_down = (
            structure.last_trend == 'DOWN'
            and structure.trend_strength > 0.6
            and ema_fast < ema_slow
        )

        if not (trend_up or trend_down):
            return RuleDecision(0, 0.0, [], {}, 'TC')

        # Check pullback conditions
        if trend_up:
            # Bullish TC: Price should be near EMA20/50
            pullback_to_ema = (
                abs(current_price - ema_fast) / current_atr < self.max_pullback_atr
                and abs(current_price - ema_slow) / current_atr < self.max_pullback_atr
            )

            if pullback_to_ema:
                # Check for bullish rejection
                bullish_rejection = self._check_bullish_rejection(df)

                if bullish_rejection:
                    score = self._calculate_tc_score(df, structure, 'UP')
                    return RuleDecision(
                        side=1,
                        score=score,
                        tags=['TC_UP', 'PULLBACK', 'BULLISH_REJECTION'],
                        context={
                            'trend': 'UP',
                            'pullback_level': 'EMA',
                            'rejection_type': 'bullish',
                            'ema_distance': abs(current_price - ema_fast) / current_atr,
                        },
                        rule_type='TC',
                    )

        elif trend_down:
            # Bearish TC: Price should be near EMA20/50
            pullback_to_ema = (
                abs(current_price - ema_fast) / current_atr < self.max_pullback_atr
                and abs(current_price - ema_slow) / current_atr < self.max_pullback_atr
            )

            if pullback_to_ema:
                # Check for bearish rejection
                bearish_rejection = self._check_bearish_rejection(df)

                if bearish_rejection:
                    score = self._calculate_tc_score(df, structure, 'DOWN')
                    return RuleDecision(
                        side=-1,
                        score=score,
                        tags=['TC_DOWN', 'PULLBACK', 'BEARISH_REJECTION'],
                        context={
                            'trend': 'DOWN',
                            'pullback_level': 'EMA',
                            'rejection_type': 'bearish',
                            'ema_distance': abs(current_price - ema_fast) / current_atr,
                        },
                        rule_type='TC',
                    )

        return RuleDecision(0, 0.0, [], {}, 'TC')

    def _evaluate_breakout_retest(
        self, df: pd.DataFrame, structure: MarketStructure
    ) -> RuleDecision:
        """
        Evaluate Breakout-Retest (BR) patterns with PA enhancement

        Pattern: BOS occurred + retest of broken level + confirmation → trade in breakout direction
        """
        if not structure.last_bos:
            return RuleDecision(0, 0.0, [], {}, 'BR')

        bos = structure.last_bos
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]

        # Check if we're in retest phase
        if bos['type'] == 'BULLISH':
            # Bullish breakout, look for retest of resistance turned support
            retest_level = bos['break_price']
            retest_distance = abs(current_price - retest_level) / current_atr

            if retest_distance < 1.0:  # Price near breakout level
                # Check for bullish confirmation
                bullish_confirmation = self._check_bullish_confirmation(df)

                if bullish_confirmation:
                    score = self._calculate_br_score(df, structure, 'BULLISH')
                    return RuleDecision(
                        side=1,
                        score=score,
                        tags=['BR_BULLISH', 'RETEST', 'BULLISH_CONFIRMATION'],
                        context={
                            'breakout_type': 'BULLISH',
                            'retest_level': retest_level,
                            'retest_distance': retest_distance,
                            'confirmation_type': 'bullish',
                        },
                        rule_type='BR',
                    )

        elif bos['type'] == 'BEARISH':
            # Bearish breakout, look for retest of support turned resistance
            retest_level = bos['break_price']
            retest_distance = abs(current_price - retest_level) / current_atr

            if retest_distance < 1.0:  # Price near breakout level
                # Check for bearish confirmation
                bearish_confirmation = self._check_bearish_confirmation(df)

                if bearish_confirmation:
                    score = self._calculate_br_score(df, structure, 'BEARISH')
                    return RuleDecision(
                        side=-1,
                        score=score,
                        tags=['BR_BEARISH', 'RETEST', 'BEARISH_CONFIRMATION'],
                        context={
                            'breakout_type': 'BEARISH',
                            'retest_level': retest_level,
                            'retest_distance': retest_distance,
                            'confirmation_type': 'bearish',
                        },
                        rule_type='BR',
                    )

        return RuleDecision(0, 0.0, [], {}, 'BR')

    def _evaluate_reversal(self, df: pd.DataFrame, structure: MarketStructure) -> RuleDecision:
        """
        Evaluate Reversal (RV) patterns with enhanced PA detection

        Pattern: Sweep of liquidity + powerful reversal + divergence (optional) → counter-trend position
        """
        if len(df) < 5:
            return RuleDecision(0, 0.0, [], {}, 'RV')

        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]

        # Check for liquidity sweep
        liquidity_sweep = self._check_liquidity_sweep(df)

        if not liquidity_sweep:
            return RuleDecision(0, 0.0, [], {}, 'RV')

        # Check for powerful reversal
        if liquidity_sweep['type'] == 'BULLISH':
            # Bullish reversal after bearish sweep
            reversal_confirmation = self._check_bullish_reversal(df)

            if reversal_confirmation:
                # Check for RSI divergence (optional)
                rsi_divergence = self._check_bullish_rsi_divergence(df)

                score = self._calculate_rv_score(df, structure, 'BULLISH', rsi_divergence)
                return RuleDecision(
                    side=1,
                    score=score,
                    tags=['RV_BULLISH', 'LIQUIDITY_SWEEP', 'REVERSAL'],
                    context={
                        'reversal_type': 'BULLISH',
                        'sweep_type': 'bearish_liquidity',
                        'rsi_divergence': rsi_divergence,
                        'reversal_strength': reversal_confirmation['strength'],
                    },
                    rule_type='RV',
                )

        elif liquidity_sweep['type'] == 'BEARISH':
            # Bearish reversal after bullish sweep
            reversal_confirmation = self._check_bearish_reversal(df)

            if reversal_confirmation:
                # Check for RSI divergence (optional)
                rsi_divergence = self._check_bearish_rsi_divergence(df)

                score = self._calculate_rv_score(df, structure, 'BEARISH', rsi_divergence)
                return RuleDecision(
                    side=-1,
                    score=score,
                    tags=['RV_BEARISH', 'LIQUIDITY_SWEEP', 'REVERSAL'],
                    context={
                        'reversal_type': 'BEARISH',
                        'sweep_type': 'bullish_liquidity',
                        'rsi_divergence': rsi_divergence,
                        'reversal_strength': reversal_confirmation['strength'],
                    },
                    rule_type='RV',
                )

        return RuleDecision(0, 0.0, [], {}, 'RV')

    def _check_bullish_rejection(self, df: pd.DataFrame) -> bool:
        """Check for bullish rejection pattern with enhanced PA logic"""
        if len(df) < 3:
            return False

        # Look for bullish rejection in last 3 bars
        for i in range(len(df) - 3, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            close = df['close'].iloc[i]
            open_price = df['open'].iloc[i]

            # Bullish rejection: long lower shadow, close above open
            body_size = abs(close - open_price)
            lower_shadow = min(open_price, close) - low
            upper_shadow = high - max(open_price, close)

            if (
                lower_shadow > body_size * 2  # Long lower shadow
                and close > open_price  # Bullish close
                and lower_shadow > upper_shadow
            ):  # Lower shadow longer than upper
                return True

        return False

    def _check_bearish_rejection(self, df: pd.DataFrame) -> bool:
        """Check for bearish rejection pattern with enhanced PA logic"""
        if len(df) < 3:
            return False

        # Look for bearish rejection in last 3 bars
        for i in range(len(df) - 3, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            close = df['close'].iloc[i]
            open_price = df['open'].iloc[i]

            # Bearish rejection: long upper shadow, close below open
            body_size = abs(close - open_price)
            lower_shadow = min(open_price, close) - low
            upper_shadow = high - max(open_price, close)

            if (
                upper_shadow > body_size * 2  # Long upper shadow
                and close < open_price  # Bearish close
                and upper_shadow > lower_shadow
            ):  # Upper shadow longer than lower
                return True

        return False

    def _check_bullish_confirmation(self, df: pd.DataFrame) -> bool:
        """Check for bullish confirmation after retest"""
        if len(df) < 2:
            return False

        # Check last 2 bars for bullish momentum
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        last_high = df['high'].iloc[-1]
        last_low = df['low'].iloc[-1]

        # Price should be moving up
        price_momentum = last_close > prev_close

        # Strong close near high
        strong_close = (last_close - last_low) / (last_high - last_low) > 0.6

        return price_momentum and strong_close

    def _check_bearish_confirmation(self, df: pd.DataFrame) -> bool:
        """Check for bearish confirmation after retest"""
        if len(df) < 2:
            return False

        # Check last 2 bars for bearish momentum
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        last_high = df['high'].iloc[-1]
        last_low = df['low'].iloc[-1]

        # Price should be moving down
        price_momentum = last_close < prev_close

        # Strong close near low
        strong_close = (last_high - last_close) / (last_high - last_low) > 0.6

        return price_momentum and strong_close

    def _check_liquidity_sweep(self, df: pd.DataFrame) -> dict | None:
        """Check for liquidity sweep pattern"""
        if len(df) < 5:
            return None

        # Look for sweep in last 5 bars
        for i in range(len(df) - 5, len(df) - 1):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]

            # Check if this bar swept previous highs/lows
            prev_highs = df['high'].iloc[max(0, i - 3) : i]
            prev_lows = df['low'].iloc[max(0, i - 3) : i]

            if len(prev_highs) > 0 and len(prev_lows) > 0:
                # Bullish sweep: sweeps previous lows but closes above
                if current_low < prev_lows.min() and current_close > prev_lows.min():
                    return {
                        'type': 'BULLISH',
                        'index': i,
                        'sweep_level': prev_lows.min(),
                        'close_above': True,
                    }

                # Bearish sweep: sweeps previous highs but closes below
                elif current_high > prev_highs.max() and current_close < prev_highs.max():
                    return {
                        'type': 'BEARISH',
                        'index': i,
                        'sweep_level': prev_highs.max(),
                        'close_below': True,
                    }

        return None

    def _check_bullish_reversal(self, df: pd.DataFrame) -> dict | None:
        """Check for bullish reversal after sweep"""
        if len(df) < 3:
            return None

        # Check last 3 bars for bullish reversal
        last_3_bars = df.tail(3)

        # Price should be moving up
        price_trend = last_3_bars['close'].iloc[-1] > last_3_bars['close'].iloc[0]

        # Strong bullish bars
        strong_bullish = all(
            last_3_bars['close'].iloc[i] > last_3_bars['open'].iloc[i]
            for i in range(len(last_3_bars))
        )

        if price_trend and strong_bullish:
            # Calculate reversal strength
            reversal_range = last_3_bars['high'].max() - last_3_bars['low'].min()
            avg_atr = last_3_bars['atr'].mean()
            strength = min(reversal_range / avg_atr, 1.0) if avg_atr > 0 else 0.5

            return {'strength': strength}

        return None

    def _check_bearish_reversal(self, df: pd.DataFrame) -> dict | None:
        """Check for bearish reversal after sweep"""
        if len(df) < 3:
            return None

        # Check last 3 bars for bearish reversal
        last_3_bars = df.tail(3)

        # Price should be moving down
        price_trend = last_3_bars['close'].iloc[-1] < last_3_bars['close'].iloc[0]

        # Strong bearish bars
        strong_bearish = all(
            last_3_bars['close'].iloc[i] < last_3_bars['open'].iloc[i]
            for i in range(len(last_3_bars))
        )

        if price_trend and strong_bearish:
            # Calculate reversal strength
            reversal_range = last_3_bars['high'].max() - last_3_bars['low'].min()
            avg_atr = last_3_bars['atr'].mean()
            strength = min(reversal_range / avg_atr, 1.0) if avg_atr > 0 else 0.5

            return {'strength': strength}

        return None

    def _check_bullish_rsi_divergence(self, df: pd.DataFrame) -> bool:
        """Check for bullish RSI divergence"""
        if len(df) < 10:
            return False

        # Look for price making lower lows while RSI makes higher lows
        last_10 = df.tail(10)
        price_lows = last_10['low'].rolling(3).min()
        rsi_lows = last_10['rsi'].rolling(3).min()

        # Check if last 2 price lows are decreasing while RSI lows are increasing
        if len(price_lows) >= 4 and len(rsi_lows) >= 4:
            price_decreasing = price_lows.iloc[-1] < price_lows.iloc[-3]
            rsi_increasing = rsi_lows.iloc[-1] > rsi_lows.iloc[-3]

            return price_decreasing and rsi_increasing

        return False

    def _check_bearish_rsi_divergence(self, df: pd.DataFrame) -> bool:
        """Check for bearish RSI divergence"""
        if len(df) < 10:
            return False

        # Look for price making higher highs while RSI makes lower highs
        last_10 = df.tail(10)
        price_highs = last_10['high'].rolling(3).max()
        rsi_highs = last_10['rsi'].rolling(3).max()

        # Check if last 2 price highs are increasing while RSI highs are decreasing
        if len(price_highs) >= 4 and len(rsi_highs) >= 4:
            price_increasing = price_highs.iloc[-1] > price_highs.iloc[-3]
            rsi_decreasing = rsi_highs.iloc[-1] < rsi_highs.iloc[-3]

            return price_increasing and rsi_decreasing

        return False

    def _calculate_tc_score(
        self, df: pd.DataFrame, structure: MarketStructure, trend: str
    ) -> float:
        """Calculate score for Trend Continuation pattern"""
        base_score = 0.6

        # Trend strength bonus
        trend_bonus = structure.trend_strength * 0.2

        # Structure quality bonus
        quality_bonus = structure.structure_quality * 0.1

        # EMA alignment bonus
        ema_alignment = 0.1 if abs(df['ema_spread'].iloc[-1]) > 0.01 else 0.0

        # RSI confirmation bonus
        current_rsi = df['rsi'].iloc[-1]
        if trend == 'UP' and 30 < current_rsi < 70:
            rsi_bonus = 0.1
        elif trend == 'DOWN' and 30 < current_rsi < 70:
            rsi_bonus = 0.1
        else:
            rsi_bonus = 0.0

        total_score = base_score + trend_bonus + quality_bonus + ema_alignment + rsi_bonus
        return min(max(total_score, 0.0), 1.0)

    def _calculate_br_score(
        self, df: pd.DataFrame, structure: MarketStructure, breakout_type: str
    ) -> float:
        """Calculate score for Breakout-Retest pattern"""
        base_score = 0.7

        # BOS strength bonus
        bos_strength = structure.last_bos['strength'] * 0.2 if structure.last_bos else 0.0

        # Structure quality bonus
        quality_bonus = structure.structure_quality * 0.1

        # Volume confirmation (if available)
        volume_bonus = 0.0
        if 'volume' in df.columns:
            avg_volume = df['volume'].tail(10).mean()
            current_volume = df['volume'].iloc[-1]
            if current_volume > avg_volume * 1.2:
                volume_bonus = 0.1

        total_score = base_score + bos_strength + quality_bonus + volume_bonus
        return min(max(total_score, 0.0), 1.0)

    def _calculate_rv_score(
        self, df: pd.DataFrame, structure: MarketStructure, reversal_type: str, rsi_divergence: bool
    ) -> float:
        """Calculate score for Reversal pattern"""
        base_score = 0.5  # Lower base score for reversals (riskier)

        # Reversal strength bonus
        reversal_strength = 0.2  # Will be updated with actual reversal data

        # RSI divergence bonus
        divergence_bonus = 0.1 if rsi_divergence else 0.0

        # Structure quality bonus (lower for reversals)
        quality_bonus = structure.structure_quality * 0.05

        # Trend exhaustion bonus
        trend_exhaustion = 0.1 if structure.trend_strength > 0.8 else 0.0

        total_score = (
            base_score + reversal_strength + divergence_bonus + quality_bonus + trend_exhaustion
        )
        return min(max(total_score, 0.0), 1.0)


# Convenience function for quick evaluation
def evaluate_rules(df: pd.DataFrame, structure: MarketStructure, config: dict) -> RuleDecision:
    """
    Quick rule evaluation with Price Action integration

    Args:
        df: DataFrame with OHLCV data
        structure: Market structure analysis
        config: Strategy configuration

    Returns:
        RuleDecision object
    """
    strategy = RuleBasedStrategy(config)
    return strategy.evaluate_rules(df, structure)
