#!/usr/bin/env python3
"""
Lightweight performance monitoring for MR BEN Trading System.
Tracks latency, memory usage, and other key metrics.
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta


class PerformanceMonitor:
    """Lightweight performance monitoring system."""
    
    def __init__(self, log_interval: int = 60, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            log_interval: Seconds between performance log entries
            max_history: Maximum number of historical measurements to keep
        """
        self.log_interval = log_interval
        self.max_history = max_history
        self.logger = logging.getLogger("PerformanceMonitor")
        
        # Metrics storage
        self.latency_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.cpu_history = deque(maxlen=max_history)
        self.trade_latency_history = deque(maxlen=max_history)
        
        # Current measurements
        self.current_measurements = defaultdict(float)
        self.measurement_count = defaultdict(int)
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.last_log_time = time.time()
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._collect_metrics()
                
                # Log metrics periodically
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    self._log_performance_summary()
                    self.last_log_time = current_time
                    
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(5)  # Wait before retrying
                
    def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.memory_history.append({
                'timestamp': datetime.now(),
                'rss_mb': memory_mb,
                'vms_mb': memory_info.vms / 1024 / 1024
            })
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.cpu_history.append({
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent
            })
            
        except Exception as e:
            self.logger.debug(f"Could not collect system metrics: {e}")
            
    def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency."""
        self.latency_history.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'latency_ms': latency_ms
        })
        
        # Update running averages
        self.current_measurements[f'avg_latency_{operation}'] = (
            (self.current_measurements[f'avg_latency_{operation}'] * 
             self.measurement_count[f'avg_latency_{operation}'] + latency_ms) /
            (self.measurement_count[f'avg_latency_{operation}'] + 1)
        )
        self.measurement_count[f'avg_latency_{operation}'] += 1
        
    def record_trade_latency(self, latency_ms: float):
        """Record trade execution latency."""
        self.trade_latency_history.append({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms
        })
        
    def _log_performance_summary(self):
        """Log performance summary."""
        if not self.latency_history and not self.memory_history:
            return
            
        summary = self.get_performance_summary()
        
        self.logger.info(
            f"Performance Summary - "
            f"Memory: {summary['memory']['current_rss']:.1f}MB, "
            f"CPU: {summary['cpu']['current']:.1f}%, "
            f"Avg Trade Latency: {summary['trade_latency']['average']:.1f}ms"
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        summary = {
            'memory': self._get_memory_summary(),
            'cpu': self._get_cpu_summary(),
            'latency': self._get_latency_summary(),
            'trade_latency': self._get_trade_latency_summary()
        }
        
        return summary
        
    def _get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {'current_rss': 0, 'peak_rss': 0, 'average_rss': 0}
            
        current = self.memory_history[-1]['rss_mb'] if self.memory_history else 0
        peak = max(m['rss_mb'] for m in self.memory_history) if self.memory_history else 0
        average = sum(m['rss_mb'] for m in self.memory_history) / len(self.memory_history) if self.memory_history else 0
        
        return {
            'current_rss': current,
            'peak_rss': peak,
            'average_rss': average
        }
        
    def _get_cpu_summary(self) -> Dict[str, Any]:
        """Get CPU usage summary."""
        if not self.cpu_history:
            return {'current': 0, 'peak': 0, 'average': 0}
            
        current = self.cpu_history[-1]['cpu_percent'] if self.cpu_history else 0
        peak = max(m['cpu_percent'] for m in self.cpu_history) if self.cpu_history else 0
        average = sum(m['cpu_percent'] for m in self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        
        return {
            'current': current,
            'peak': peak,
            'average': average
        }
        
    def _get_latency_summary(self) -> Dict[str, Any]:
        """Get operation latency summary."""
        if not self.latency_history:
            return {'total_operations': 0, 'average_latency': 0}
            
        total_ops = len(self.latency_history)
        avg_latency = sum(m['latency_ms'] for m in self.latency_history) / total_ops
        
        return {
            'total_operations': total_ops,
            'average_latency': avg_latency
        }
        
    def _get_trade_latency_summary(self) -> Dict[str, Any]:
        """Get trade latency summary."""
        if not self.trade_latency_history:
            return {'total_trades': 0, 'average': 0, 'min': 0, 'max': 0}
            
        latencies = [m['latency_ms'] for m in self.trade_latency_history]
        total_trades = len(latencies)
        average = sum(latencies) / total_trades
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        return {
            'total_trades': total_trades,
            'average': average,
            'min': min_latency,
            'max': max_latency
        }
        
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_latency = [
            m for m in self.latency_history 
            if m['timestamp'] > cutoff_time
        ]
        
        recent_memory = [
            m for m in self.memory_history 
            if m['timestamp'] > cutoff_time
        ]
        
        recent_cpu = [
            m for m in self.cpu_history 
            if m['timestamp'] > cutoff_time
        ]
        
        return {
            'latency': recent_latency,
            'memory': recent_memory,
            'cpu': recent_cpu,
            'time_window_minutes': minutes
        }
        
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.latency_history.clear()
        self.memory_history.clear()
        self.cpu_history.clear()
        self.trade_latency_history.clear()
        self.current_measurements.clear()
        self.measurement_count.clear()
        self.logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def start_performance_monitoring(log_interval: int = 60):
    """Start global performance monitoring."""
    performance_monitor.log_interval = log_interval
    performance_monitor.start_monitoring()
    

def stop_performance_monitoring():
    """Stop global performance monitoring."""
    performance_monitor.stop_monitoring()
    

def record_operation_latency(operation: str, latency_ms: float):
    """Record operation latency using global monitor."""
    performance_monitor.record_latency(operation, latency_ms)
    

def record_trade_latency(latency_ms: float):
    """Record trade execution latency using global monitor."""
    performance_monitor.record_trade_latency(latency_ms)
    

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary from global monitor."""
    return performance_monitor.get_performance_summary()
