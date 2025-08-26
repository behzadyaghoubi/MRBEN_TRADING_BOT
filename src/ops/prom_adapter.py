#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Prometheus Adapter for Monitoring
"""

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_metrics_snapshot() -> Dict[str, Any]:
    """
    Get metrics snapshot from the dashboard
    This function should be imported from the existing dashboard
    """
    try:
        # Try to import from existing dashboard
        from src.agent.dashboard import DashboardIntegration
        dashboard = DashboardIntegration({})
        return dashboard.get_metrics()
    except ImportError:
        # Fallback to basic metrics
        return {
            "uptime_seconds": int(time.time() % 86400),  # Seconds since midnight
            "cycle_count": 0,
            "total_trades": 0,
            "error_rate": 0.0,
            "memory_mb": 0.0
        }

class PromHandler(BaseHTTPRequestHandler):
    """Prometheus metrics handler"""
    
    def do_GET(self):
        if self.path != "/prom":
            self.send_response(404)
            self.end_headers()
            return
        
        try:
            # Get metrics from dashboard
            data = get_metrics_snapshot()
            
            # Convert to Prometheus format
            lines = []
            lines.append(f"mrben_uptime_seconds {int(data.get('uptime_seconds', 0))}")
            lines.append(f"mrben_cycles_total {int(data.get('cycle_count', 0))}")
            lines.append(f"mrben_total_trades {int(data.get('total_trades', 0))}")
            lines.append(f"mrben_error_rate {float(data.get('error_rate', 0.0))}")
            lines.append(f"mrben_memory_mb {float(data.get('memory_mb', 0.0))}")
            
            # Add timestamp
            lines.append(f"# Timestamp: {int(time.time())}")
            
            body = "\n".join(lines) + "\n"
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            
            logger.debug(f"Prometheus metrics served: {len(lines)} metrics")
            
        except Exception as e:
            logger.error(f"Error serving Prometheus metrics: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode("utf-8"))
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"Prometheus: {format % args}")

def start_prometheus_adapter(port: int = 9100) -> HTTPServer:
    """Start the Prometheus metrics server"""
    try:
        server = HTTPServer(("127.0.0.1", port), PromHandler)
        
        # Start server in daemon thread
        def serve_forever():
            try:
                server.serve_forever()
            except Exception as e:
                logger.error(f"Prometheus server error: {e}")
        
        thread = threading.Thread(target=serve_forever, daemon=True)
        thread.start()
        
        logger.info(f"Prometheus adapter started at http://127.0.0.1:{port}/prom")
        return server
        
    except Exception as e:
        logger.error(f"Failed to start Prometheus adapter: {e}")
        raise

if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.INFO)
    server = start_prometheus_adapter(9100)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus adapter")
        server.shutdown()
