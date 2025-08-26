"""
Market Structure Analysis for MR BEN Pro Strategy

Professional market structure analysis including:
- Swing high/low detection
- Higher Highs (HH) / Lower Lows (LL) identification
- Break of Structure (BOS) detection
- Change of Character (CHOCH) identification
- Trend direction and strength analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    type: str  # 'high' or 'low'
    strength: float  # 0-1, based on surrounding bars
    volume: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None

@dataclass
class MarketStructure:
    """Complete market structure analysis"""
    swings: List[SwingPoint]
    last_trend: str  # 'UP', 'DOWN', 'RANGE'
    last_bos: Optional[Dict] = None
    last_choch: Optional[Dict] = None
    trend_strength: float = 0.0  # 0-1
    structure_quality: float = 0.0  # 0-1

class MarketStructureAnalyzer:
    """Analyzes market structure for professional trading"""
    
    def __init__(self, left_bars: int = 2, right_bars: int = 2, min_swing_distance: float = 0.001):
        """
        Initialize the market structure analyzer
        
        Args:
            left_bars: Number of bars to look left for swing confirmation
            right_bars: Number of bars to look right for swing confirmation
            min_swing_distance: Minimum price distance for swing (as fraction of price)
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.min_swing_distance = min_swing_distance
    
    def detect_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect swing highs and lows in the price data
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            List of SwingPoint objects
        """
        swings = []
        high_series = df['high']
        low_series = df['low']
        close_series = df['close']
        volume_series = df.get('volume', pd.Series([1.0] * len(df)))
        
        for i in range(self.left_bars, len(df) - self.right_bars):
            # Check for swing high
            if self._is_swing_high(high_series, i):
                strength = self._calculate_swing_strength(high_series, low_series, i, 'high')
                swing = SwingPoint(
                    index=i,
                    price=high_series.iloc[i],
                    type='high',
                    strength=strength,
                    volume=volume_series.iloc[i] if not volume_series.empty else None,
                    timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else None
                )
                swings.append(swing)
            
            # Check for swing low
            if self._is_swing_low(low_series, i):
                strength = self._calculate_swing_strength(high_series, low_series, i, 'low')
                swing = SwingPoint(
                    index=i,
                    price=low_series.iloc[i],
                    type='low',
                    strength=strength,
                    volume=volume_series.iloc[i] if not volume_series.empty else None,
                    timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else None
                )
                swings.append(swing)
        
        # Sort swings by index
        swings.sort(key=lambda x: x.index)
        return swings
    
    def _is_swing_high(self, high_series: pd.Series, index: int) -> bool:
        """Check if bar at index is a swing high"""
        current_high = high_series.iloc[index]
        
        # Check left bars
        for i in range(1, self.left_bars + 1):
            if high_series.iloc[index - i] >= current_high:
                return False
        
        # Check right bars
        for i in range(1, self.right_bars + 1):
            if high_series.iloc[index + i] >= current_high:
                return False
        
        return True
    
    def _is_swing_low(self, low_series: pd.Series, index: int) -> bool:
        """Check if bar at index is a swing low"""
        current_low = low_series.iloc[index]
        
        # Check left bars
        for i in range(1, self.left_bars + 1):
            if low_series.iloc[index - i] <= current_low:
                return False
        
        # Check right bars
        for i in range(1, self.right_bars + 1):
            if low_series.iloc[index + i] <= current_low:
                return False
        
        return True
    
    def _calculate_swing_strength(self, high_series: pd.Series, low_series: pd.Series, 
                                 index: int, swing_type: str) -> float:
        """Calculate the strength of a swing point (0-1)"""
        if swing_type == 'high':
            current_price = high_series.iloc[index]
            left_prices = [high_series.iloc[index - i] for i in range(1, self.left_bars + 1)]
            right_prices = [high_series.iloc[index + i] for i in range(1, self.right_bars + 1)]
            
            # Calculate how much higher this swing is compared to surrounding bars
            left_diff = current_price - max(left_prices) if left_prices else 0
            right_diff = current_price - max(right_prices) if right_prices else 0
            
            # Normalize by ATR or price range
            price_range = high_series.max() - low_series.min()
            if price_range > 0:
                strength = (left_diff + right_diff) / (2 * price_range)
                return min(max(strength, 0.0), 1.0)
        
        elif swing_type == 'low':
            current_price = low_series.iloc[index]
            left_prices = [low_series.iloc[index - i] for i in range(1, self.left_bars + 1)]
            right_prices = [low_series.iloc[index + i] for i in range(1, self.right_bars + 1)]
            
            # Calculate how much lower this swing is compared to surrounding bars
            left_diff = min(left_prices) - current_price if left_prices else 0
            right_diff = min(right_prices) - current_price if right_prices else 0
            
            # Normalize by ATR or price range
            price_range = high_series.max() - low_series.min()
            if price_range > 0:
                strength = (left_diff + right_diff) / (2 * price_range)
                return min(max(strength, 0.0), 1.0)
        
        return 0.5  # Default strength
    
    def identify_trend(self, swings: List[SwingPoint], df: pd.DataFrame) -> str:
        """
        Identify the current trend based on swing points
        
        Args:
            swings: List of swing points
            df: Price data DataFrame
            
        Returns:
            Trend direction: 'UP', 'DOWN', or 'RANGE'
        """
        if len(swings) < 3:
            return 'RANGE'
        
        # Get recent swings (last 3-5)
        recent_swings = swings[-5:] if len(swings) >= 5 else swings
        
        # Count higher highs and lower lows
        higher_highs = 0
        lower_lows = 0
        
        for i in range(1, len(recent_swings)):
            if recent_swings[i].type == 'high' and recent_swings[i-1].type == 'high':
                if recent_swings[i].price > recent_swings[i-1].price:
                    higher_highs += 1
            elif recent_swings[i].type == 'low' and recent_swings[i-1].type == 'low':
                if recent_swings[i].price < recent_swings[i-1].price:
                    lower_lows += 1
        
        # Determine trend
        if higher_highs > lower_lows and higher_highs >= 2:
            return 'UP'
        elif lower_lows > higher_highs and lower_lows >= 2:
            return 'DOWN'
        else:
            return 'RANGE'
    
    def detect_bos(self, swings: List[SwingPoint], df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Break of Structure (BOS)
        
        Args:
            swings: List of swing points
            df: Price data DataFrame
            
        Returns:
            BOS information or None if no BOS detected
        """
        if len(swings) < 2:
            return None
        
        # Look for recent BOS
        for i in range(len(swings) - 1, 0, -1):
            current_swing = swings[i]
            prev_swing = swings[i-1]
            
            if current_swing.type == 'high' and prev_swing.type == 'low':
                # Potential bullish BOS
                if current_swing.price > prev_swing.price:
                    return {
                        'type': 'BULLISH',
                        'swing_index': i,
                        'break_price': current_swing.price,
                        'break_index': current_swing.index,
                        'strength': current_swing.strength
                    }
            
            elif current_swing.type == 'low' and prev_swing.type == 'high':
                # Potential bearish BOS
                if current_swing.price < prev_swing.price:
                    return {
                        'type': 'BEARISH',
                        'swing_index': i,
                        'break_price': current_swing.price,
                        'break_index': current_swing.index,
                        'strength': current_swing.strength
                    }
        
        return None
    
    def detect_choch(self, swings: List[SwingPoint], df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Change of Character (CHOCH)
        
        Args:
            swings: List of swing points
            df: Price data DataFrame
            
        Returns:
            CHOCH information or None if no CHOCH detected
        """
        if len(swings) < 3:
            return None
        
        # Look for recent CHOCH
        for i in range(len(swings) - 2, 0, -1):
            current_swing = swings[i]
            prev_swing = swings[i-1]
            next_swing = swings[i+1]
            
            # Bullish CHOCH: Low -> High -> Low (but second low is higher)
            if (prev_swing.type == 'low' and 
                current_swing.type == 'high' and 
                next_swing.type == 'low'):
                if next_swing.price > prev_swing.price:
                    return {
                        'type': 'BULLISH',
                        'swing_index': i,
                        'change_price': current_swing.price,
                        'change_index': current_swing.index,
                        'strength': current_swing.strength
                    }
            
            # Bearish CHOCH: High -> Low -> High (but second high is lower)
            elif (prev_swing.type == 'high' and 
                  current_swing.type == 'low' and 
                  next_swing.type == 'high'):
                if next_swing.price < prev_swing.price:
                    return {
                        'type': 'BEARISH',
                        'swing_index': i,
                        'change_price': current_swing.price,
                        'change_index': current_swing.index,
                        'strength': current_swing.strength
                    }
        
        return None
    
    def calculate_trend_strength(self, swings: List[SwingPoint], df: pd.DataFrame) -> float:
        """
        Calculate trend strength (0-1)
        
        Args:
            swings: List of swing points
            df: Price data DataFrame
            
        Returns:
            Trend strength as float between 0 and 1
        """
        if len(swings) < 3:
            return 0.0
        
        # Get recent swings
        recent_swings = swings[-5:] if len(swings) >= 5 else swings
        
        # Calculate average swing strength
        avg_strength = np.mean([s.strength for s in recent_swings])
        
        # Calculate trend consistency
        trend_consistency = 0.0
        if len(recent_swings) >= 2:
            consistent_moves = 0
            total_moves = 0
            
            for i in range(1, len(recent_swings)):
                if recent_swings[i].type == recent_swings[i-1].type:
                    total_moves += 1
                    if recent_swings[i].type == 'high':
                        if recent_swings[i].price > recent_swings[i-1].price:
                            consistent_moves += 1
                    else:  # low
                        if recent_swings[i].price < recent_swings[i-1].price:
                            consistent_moves += 1
            
            if total_moves > 0:
                trend_consistency = consistent_moves / total_moves
        
        # Combine strength and consistency
        trend_strength = (avg_strength + trend_consistency) / 2
        return min(max(trend_strength, 0.0), 1.0)
    
    def calculate_structure_quality(self, swings: List[SwingPoint], df: pd.DataFrame) -> float:
        """
        Calculate overall structure quality (0-1)
        
        Args:
            swings: List of swing points
            df: Price data DataFrame
            
        Returns:
            Structure quality as float between 0 and 1
        """
        if len(swings) < 3:
            return 0.0
        
        # Factors that contribute to quality:
        # 1. Number of swings (more = better, up to a point)
        swing_count_score = min(len(swings) / 10.0, 1.0)
        
        # 2. Average swing strength
        avg_strength = np.mean([s.strength for s in swings])
        
        # 3. Swing distribution (not too clustered)
        indices = [s.index for s in swings]
        if len(indices) > 1:
            gaps = [indices[i] - indices[i-1] for i in range(1, len(indices))]
            avg_gap = np.mean(gaps)
            gap_score = min(avg_gap / 20.0, 1.0)  # Normalize
        else:
            gap_score = 0.5
        
        # 4. Recent swing quality
        recent_swings = swings[-3:] if len(swings) >= 3 else swings
        recent_quality = np.mean([s.strength for s in recent_swings])
        
        # Combine all factors
        quality = (swing_count_score + avg_strength + gap_score + recent_quality) / 4
        return min(max(quality, 0.0), 1.0)
    
    def analyze_structure(self, df: pd.DataFrame) -> MarketStructure:
        """
        Complete market structure analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            MarketStructure object with complete analysis
        """
        # Detect swings
        swings = self.detect_swings(df)
        
        # Identify trend
        trend = self.identify_trend(swings, df)
        
        # Detect BOS and CHOCH
        bos = self.detect_bos(swings, df)
        choch = self.detect_choch(swings, df)
        
        # Calculate metrics
        trend_strength = self.calculate_trend_strength(swings, df)
        structure_quality = self.calculate_structure_quality(swings, df)
        
        return MarketStructure(
            swings=swings,
            last_trend=trend,
            last_bos=bos,
            last_choch=choch,
            trend_strength=trend_strength,
            structure_quality=structure_quality
        )
    
    def get_support_resistance_levels(self, swings: List[SwingPoint], 
                                    current_price: float, 
                                    lookback_bars: int = 100) -> Dict:
        """
        Get current support and resistance levels
        
        Args:
            swings: List of swing points
            current_price: Current market price
            lookback_bars: Number of bars to look back
            
        Returns:
            Dictionary with support and resistance levels
        """
        if not swings:
            return {'support': [], 'resistance': []}
        
        # Filter recent swings
        recent_swings = [s for s in swings if s.index >= len(swings) - lookback_bars]
        
        # Separate highs and lows
        highs = [s for s in recent_swings if s.type == 'high']
        lows = [s for s in recent_swings if s.type == 'low']
        
        # Find nearest levels
        support_levels = []
        resistance_levels = []
        
        for low in lows:
            if low.price < current_price:
                support_levels.append({
                    'price': low.price,
                    'strength': low.strength,
                    'index': low.index
                })
        
        for high in highs:
            if high.price > current_price:
                resistance_levels.append({
                    'price': high.price,
                    'strength': high.strength,
                    'index': high.index
                })
        
        # Sort by distance to current price
        support_levels.sort(key=lambda x: abs(x['price'] - current_price))
        resistance_levels.sort(key=lambda x: abs(x['price'] - current_price))
        
        return {
            'support': support_levels[:3],  # Top 3 nearest support levels
            'resistance': resistance_levels[:3]  # Top 3 nearest resistance levels
        }

# Convenience function for quick analysis
def analyze_market_structure(df: pd.DataFrame, 
                           left_bars: int = 2, 
                           right_bars: int = 2) -> MarketStructure:
    """
    Quick market structure analysis
    
    Args:
        df: DataFrame with OHLCV data
        left_bars: Bars to look left for swing confirmation
        right_bars: Bars to look right for swing confirmation
        
    Returns:
        MarketStructure object
    """
    analyzer = MarketStructureAnalyzer(left_bars, right_bars)
    return analyzer.analyze_structure(df)
