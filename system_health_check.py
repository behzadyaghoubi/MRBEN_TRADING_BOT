#!/usr/bin/env python3
"""
MR BEN Trading System - Comprehensive Health Check
==================================================
Tests all components of the trading system to ensure everything is working properly.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logger():
    """Setup logging for health check."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("HealthCheck")

class SystemHealthChecker:
    """Comprehensive system health checker for MR BEN trading system."""
    
    def __init__(self):
        self.logger = setup_logger()
        self.results = {
            "overall_status": "UNKNOWN",
            "components": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment and dependencies."""
        self.logger.info("🔍 Checking Python environment...")
        
        result = {
            "status": "OK",
            "python_version": sys.version,
            "missing_packages": [],
            "installed_packages": {}
        }
        
        required_packages = [
            "pandas", "numpy", "matplotlib", "seaborn", 
            "tensorflow", "scikit-learn", "joblib", "scipy"
        ]
        
        for package in required_packages:
            try:
                module = __import__(package)
                result["installed_packages"][package] = getattr(module, "__version__", "unknown")
            except ImportError:
                result["missing_packages"].append(package)
                result["status"] = "ERROR"
        
        if result["missing_packages"]:
            self.results["errors"].append(f"Missing packages: {', '.join(result['missing_packages'])}")
        
        self.results["components"]["python_environment"] = result
        return result
    
    def check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files exist and are valid."""
        self.logger.info("🔍 Checking configuration files...")
        
        result = {
            "status": "OK",
            "files": {},
            "errors": []
        }
        
        config_files = [
            "config/settings.json",
            "src/settings.json",
            "requirements.txt"
        ]
        
        for file_path in config_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        if file_path.endswith('.json'):
                            json.load(f)
                    result["files"][file_path] = "OK"
                except Exception as e:
                    result["files"][file_path] = f"ERROR: {str(e)}"
                    result["status"] = "ERROR"
                    result["errors"].append(f"Invalid {file_path}: {str(e)}")
            else:
                result["files"][file_path] = "MISSING"
                result["status"] = "ERROR"
                result["errors"].append(f"Missing file: {file_path}")
        
        if result["errors"]:
            self.results["errors"].extend(result["errors"])
        
        self.results["components"]["configuration"] = result
        return result
    
    def check_ai_models(self) -> Dict[str, Any]:
        """Check AI models exist and are loadable."""
        self.logger.info("🔍 Checking AI models...")
        
        result = {
            "status": "OK",
            "models": {},
            "errors": []
        }
        
        model_files = [
            "models/mrben_ai_signal_filter_xgb.joblib",
            "models/mrben_lstm_model.h5",
            "models/lstm_balanced_model.h5"
        ]
        
        for model_path in model_files:
            if os.path.exists(model_path):
                try:
                    # Try to load the model
                    if model_path.endswith('.joblib'):
                        import joblib
                        model = joblib.load(model_path)
                        result["models"][model_path] = f"OK (Type: {type(model).__name__})"
                    elif model_path.endswith('.h5'):
                        from tensorflow import keras
                        model = keras.models.load_model(model_path)
                        result["models"][model_path] = f"OK (Type: {type(model).__name__})"
                except Exception as e:
                    result["models"][model_path] = f"ERROR: {str(e)}"
                    result["status"] = "ERROR"
                    result["errors"].append(f"Failed to load {model_path}: {str(e)}")
            else:
                result["models"][model_path] = "MISSING"
                result["status"] = "WARNING"
                result["warnings"].append(f"Missing model: {model_path}")
        
        if result["errors"]:
            self.results["errors"].extend(result["errors"])
        if result["warnings"]:
            self.results["warnings"].extend(result["warnings"])
        
        self.results["components"]["ai_models"] = result
        return result
    
    def check_data_files(self) -> Dict[str, Any]:
        """Check data files exist."""
        self.logger.info("🔍 Checking data files...")
        
        result = {
            "status": "OK",
            "files": {},
            "warnings": []
        }
        
        data_files = [
            "data/XAUUSD_PRO_M5_live.csv",
            "data/lstm_train_data.csv",
            "data/lstm_signals_features.csv"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                result["files"][file_path] = f"OK ({size} bytes)"
            else:
                result["files"][file_path] = "MISSING"
                result["status"] = "WARNING"
                result["warnings"].append(f"Missing data file: {file_path}")
        
        if result["warnings"]:
            self.results["warnings"].extend(result["warnings"])
        
        self.results["components"]["data_files"] = result
        return result
    
    def check_strategy_modules(self) -> Dict[str, Any]:
        """Check strategy modules can be imported."""
        self.logger.info("🔍 Checking strategy modules...")
        
        result = {
            "status": "OK",
            "modules": {},
            "errors": []
        }
        
        strategy_modules = [
            "strategies.book_strategy",
            "ai_filter",
            "risk_manager"
        ]
        
        for module_name in strategy_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                result["modules"][module_name] = "OK"
            except Exception as e:
                result["modules"][module_name] = f"ERROR: {str(e)}"
                result["status"] = "ERROR"
                result["errors"].append(f"Failed to import {module_name}: {str(e)}")
        
        if result["errors"]:
            self.results["errors"].extend(result["errors"])
        
        self.results["components"]["strategy_modules"] = result
        return result
    
    def check_mt5_connection(self) -> Dict[str, Any]:
        """Check MetaTrader 5 connection."""
        self.logger.info("🔍 Checking MetaTrader 5 connection...")
        
        result = {
            "status": "UNKNOWN",
            "connection": "NOT_TESTED",
            "account_info": None,
            "errors": []
        }
        
        try:
            import MetaTrader5 as mt5
            
            # Try to initialize MT5
            if mt5.initialize():
                result["connection"] = "CONNECTED"
                
                # Try to get account info
                account_info = mt5.account_info()
                if account_info:
                    result["account_info"] = {
                        "login": account_info.login,
                        "balance": account_info.balance,
                        "equity": account_info.equity,
                        "server": account_info.server
                    }
                    result["status"] = "OK"
                else:
                    result["status"] = "WARNING"
                    result["warnings"] = ["Could not get account info"]
                
                mt5.shutdown()
            else:
                result["connection"] = "FAILED"
                result["status"] = "ERROR"
                result["errors"].append("Failed to initialize MT5")
                
        except ImportError:
            result["connection"] = "NOT_AVAILABLE"
            result["status"] = "ERROR"
            result["errors"].append("MetaTrader5 package not installed")
        except Exception as e:
            result["connection"] = "ERROR"
            result["status"] = "ERROR"
            result["errors"].append(f"MT5 connection error: {str(e)}")
        
        if result["errors"]:
            self.results["errors"].extend(result["errors"])
        
        self.results["components"]["mt5_connection"] = result
        return result
    
    def check_main_runner(self) -> Dict[str, Any]:
        """Check main runner script."""
        self.logger.info("🔍 Checking main runner script...")
        
        result = {
            "status": "OK",
            "files": {},
            "errors": []
        }
        
        runner_files = [
            "src/main_runner.py",
            "live_loop.py",
            "run_all.py"
        ]
        
        for file_path in runner_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Basic syntax check
                        compile(content, file_path, 'exec')
                    result["files"][file_path] = "OK"
                except Exception as e:
                    result["files"][file_path] = f"ERROR: {str(e)}"
                    result["status"] = "ERROR"
                    result["errors"].append(f"Syntax error in {file_path}: {str(e)}")
            else:
                result["files"][file_path] = "MISSING"
                result["status"] = "ERROR"
                result["errors"].append(f"Missing file: {file_path}")
        
        if result["errors"]:
            self.results["errors"].extend(result["errors"])
        
        self.results["components"]["main_runner"] = result
        return result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        self.logger.info("🚀 Starting MR BEN Trading System Health Check...")
        
        checks = [
            self.check_python_environment,
            self.check_configuration_files,
            self.check_ai_models,
            self.check_data_files,
            self.check_strategy_modules,
            self.check_mt5_connection,
            self.check_main_runner
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.logger.error(f"Error during {check.__name__}: {e}")
                self.results["errors"].append(f"Check {check.__name__} failed: {str(e)}")
        
        # Determine overall status
        if self.results["errors"]:
            self.results["overall_status"] = "NOT OK"
        elif self.results["warnings"]:
            self.results["overall_status"] = "OK (with warnings)"
        else:
            self.results["overall_status"] = "OK"
        
        return self.results
    
    def print_report(self):
        """Print comprehensive health check report."""
        print("\n" + "="*60)
        print("🏥 MR BEN TRADING SYSTEM - HEALTH CHECK REPORT")
        print("="*60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Overall Status: {self.results['overall_status']}")
        print()
        
        # Print component status
        for component_name, component_result in self.results["components"].items():
            status_emoji = "✅" if component_result["status"] == "OK" else "❌" if component_result["status"] == "ERROR" else "⚠️"
            print(f"{status_emoji} {component_name.upper()}: {component_result['status']}")
            
            # Print details for each component
            for key, value in component_result.items():
                if key != "status" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"   📋 {sub_key}: {sub_value}")
        
        # Print errors
        if self.results["errors"]:
            print("\n❌ ERRORS:")
            for error in self.results["errors"]:
                print(f"   • {error}")
        
        # Print warnings
        if self.results["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"   • {warning}")
        
        # Print recommendations
        if self.results["overall_status"] == "NOT OK":
            print("\n🔧 RECOMMENDATIONS:")
            print("   • Install missing Python packages")
            print("   • Fix configuration file errors")
            print("   • Ensure all required files exist")
            print("   • Check MetaTrader 5 connection")
        elif self.results["overall_status"] == "OK (with warnings)":
            print("\n🔧 RECOMMENDATIONS:")
            print("   • Address warnings for optimal performance")
            print("   • Consider adding missing data files")
        else:
            print("\n🎉 SYSTEM STATUS: READY FOR LIVE TRADING!")
            print("   • All components are working properly")
            print("   • System is ready to execute trades")
        
        print("\n" + "="*60)

def main():
    """Main function to run health check."""
    checker = SystemHealthChecker()
    results = checker.run_all_checks()
    checker.print_report()
    
    # Return appropriate exit code
    if results["overall_status"] == "NOT OK":
        sys.exit(1)
    elif results["overall_status"] == "OK (with warnings)":
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 