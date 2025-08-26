"""
Decision store for MR BEN AI Agent system.
Provides storage and retrieval of all decisions with JSONL and Parquet support.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from .schemas import (
    DecisionOutcome, SupervisorDecision, RiskOfficerDecision,
    DecisionStatus, TradingMode
)


class DecisionStore:
    """
    Decision store for tracking and storing all AI agent decisions.
    
    Supports multiple storage formats:
    - JSONL for human-readable logs
    - Parquet for efficient querying
    - In-memory for active decisions
    """
    
    def __init__(
        self,
        storage_dir: str = "artifacts/decisions",
        max_memory_decisions: int = 1000,
        enable_jsonl: bool = True,
        enable_parquet: bool = True
    ):
        """
        Initialize the decision store.
        
        Args:
            storage_dir: Directory for storing decision files
            max_memory_decisions: Maximum decisions to keep in memory
            enable_jsonl: Enable JSONL storage
            enable_parquet: Enable Parquet storage
        """
        self.storage_dir = Path(storage_dir)
        self.max_memory_decisions = max_memory_decisions
        self.enable_jsonl = enable_jsonl
        self.enable_parquet = enable_parquet
        self.logger = logging.getLogger("DecisionStore")
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.decisions: Dict[str, DecisionOutcome] = {}
        self.decision_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Storage files
        self.jsonl_file = self.storage_dir / "decisions.jsonl"
        self.parquet_file = self.storage_dir / "decisions.parquet"
        
        # Statistics
        self.total_decisions = 0
        self.storage_stats = {
            "jsonl_writes": 0,
            "parquet_writes": 0,
            "memory_cleanups": 0
        }
        
        self.logger.info(f"Decision store initialized at {self.storage_dir}")
    
    def store_decision(self, decision: DecisionOutcome) -> str:
        """
        Store a decision in the store.
        
        Args:
            decision: Decision to store
            
        Returns:
            Decision ID
        """
        decision_id = decision.decision_id
        
        # Store in memory
        self.decisions[decision_id] = decision
        self.decision_metadata[decision_id] = {
            "stored_at": datetime.utcnow(),
            "trading_mode": decision.supervisor_decision.context.trading_mode,
            "tool_name": decision.supervisor_decision.proposal.tool_name,
            "status": decision.success,
            "risk_level": decision.supervisor_decision.risk_level
        }
        
        # Persist to storage
        self._persist_decision(decision)
        
        # Cleanup if needed
        if len(self.decisions) > self.max_memory_decisions:
            self._cleanup_memory()
        
        self.total_decisions += 1
        self.logger.info(f"Stored decision {decision_id} (Total: {self.total_decisions})")
        
        return decision_id
    
    def get_decision(self, decision_id: str) -> Optional[DecisionOutcome]:
        """Get a decision by ID."""
        return self.decisions.get(decision_id)
    
    def get_decisions_by_tool(self, tool_name: str, limit: int = 100) -> List[DecisionOutcome]:
        """Get decisions by tool name."""
        decisions = []
        for decision in self.decisions.values():
            if decision.supervisor_decision.proposal.tool_name == tool_name:
                decisions.append(decision)
                if len(decisions) >= limit:
                    break
        
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)
    
    def get_decisions_by_status(self, status: bool, limit: int = 100) -> List[DecisionOutcome]:
        """Get decisions by success status."""
        decisions = []
        for decision in self.decisions.values():
            if decision.success == status:
                decisions.append(decision)
                if len(decisions) >= limit:
                    break
        
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)
    
    def get_decisions_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[DecisionOutcome]:
        """Get decisions within a time range."""
        decisions = []
        for decision in self.decisions.values():
            if start_time <= decision.timestamp <= end_time:
                decisions.append(decision)
                if len(decisions) >= limit:
                    break
        
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)
    
    def get_decisions_by_risk_level(self, risk_level: str, limit: int = 100) -> List[DecisionOutcome]:
        """Get decisions by risk level."""
        decisions = []
        for decision in self.decisions.values():
            if decision.supervisor_decision.risk_level == risk_level:
                decisions.append(decision)
                if len(decisions) >= limit:
                    break
        
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)
    
    def get_recent_decisions(self, hours: int = 24, limit: int = 100) -> List[DecisionOutcome]:
        """Get decisions from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.get_decisions_by_time_range(cutoff_time, datetime.utcnow(), limit)
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored decisions."""
        if not self.decisions:
            return {
                "total_decisions": 0,
                "successful_decisions": 0,
                "failed_decisions": 0,
                "tool_breakdown": {},
                "risk_level_breakdown": {},
                "trading_mode_breakdown": {},
                "time_breakdown": {}
            }
        
        # Basic counts
        total = len(self.decisions)
        successful = sum(1 for d in self.decisions.values() if d.success)
        failed = total - successful
        
        # Tool breakdown
        tool_breakdown = {}
        for decision in self.decisions.values():
            tool_name = decision.supervisor_decision.proposal.tool_name
            tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1
        
        # Risk level breakdown
        risk_breakdown = {}
        for decision in self.decisions.values():
            risk_level = decision.supervisor_decision.risk_level
            risk_breakdown[risk_level] = risk_breakdown.get(risk_level, 0) + 1
        
        # Trading mode breakdown
        mode_breakdown = {}
        for decision in self.decisions.values():
            mode = decision.supervisor_decision.context.trading_mode
            mode_breakdown[mode] = mode_breakdown.get(mode, 0) + 1
        
        # Time breakdown (last 7 days)
        time_breakdown = {}
        for i in range(7):
            date = (datetime.utcnow() - timedelta(days=i)).date()
            time_breakdown[date.isoformat()] = 0
        
        for decision in self.decisions.values():
            decision_date = decision.timestamp.date()
            if decision_date in time_breakdown:
                time_breakdown[decision_date.isoformat()] += 1
        
        return {
            "total_decisions": total,
            "successful_decisions": successful,
            "failed_decisions": failed,
            "success_rate": successful / total if total > 0 else 0,
            "tool_breakdown": tool_breakdown,
            "risk_level_breakdown": risk_breakdown,
            "trading_mode_breakdown": mode_breakdown,
            "time_breakdown": time_breakdown,
            "storage_stats": self.storage_stats
        }
    
    def search_decisions(
        self,
        query: str,
        fields: List[str] = None,
        limit: int = 100
    ) -> List[DecisionOutcome]:
        """
        Search decisions by text query.
        
        Args:
            query: Search query
            fields: Fields to search in (default: all text fields)
            limit: Maximum results to return
            
        Returns:
            List of matching decisions
        """
        if fields is None:
            fields = ["tool_name", "reasoning", "risk_assessment", "expected_outcome"]
        
        query_lower = query.lower()
        matches = []
        
        for decision in self.decisions.values():
            for field in fields:
                if field == "tool_name":
                    value = decision.supervisor_decision.proposal.tool_name
                elif field == "reasoning":
                    value = decision.supervisor_decision.proposal.reasoning
                elif field == "risk_assessment":
                    value = decision.supervisor_decision.proposal.risk_assessment
                elif field == "expected_outcome":
                    value = decision.supervisor_decision.proposal.expected_outcome
                else:
                    continue
                
                if query_lower in str(value).lower():
                    matches.append(decision)
                    break
                
                if len(matches) >= limit:
                    break
            
            if len(matches) >= limit:
                break
        
        return sorted(matches, key=lambda d: d.timestamp, reverse=True)
    
    def export_decisions(
        self,
        format: str = "jsonl",
        filepath: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export decisions to file.
        
        Args:
            format: Export format (jsonl, json, csv)
            filepath: Output file path
            filters: Optional filters to apply
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = self.storage_dir / f"decisions_export_{timestamp}.{format}"
        
        # Apply filters if specified
        decisions_to_export = self._apply_filters(filters) if filters else list(self.decisions.values())
        
        if format == "jsonl":
            return self._export_jsonl(decisions_to_export, filepath)
        elif format == "json":
            return self._export_json(decisions_to_export, filepath)
        elif format == "csv":
            return self._export_csv(decisions_to_export, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[DecisionOutcome]:
        """Apply filters to decisions."""
        filtered_decisions = list(self.decisions.values())
        
        for key, value in filters.items():
            if key == "tool_name":
                filtered_decisions = [d for d in filtered_decisions 
                                   if d.supervisor_decision.proposal.tool_name == value]
            elif key == "success":
                filtered_decisions = [d for d in filtered_decisions if d.success == value]
            elif key == "risk_level":
                filtered_decisions = [d for d in filtered_decisions 
                                   if d.supervisor_decision.risk_level == value]
            elif key == "trading_mode":
                filtered_decisions = [d for d in filtered_decisions 
                                   if d.supervisor_decision.context.trading_mode == value]
            elif key == "start_time":
                filtered_decisions = [d for d in filtered_decisions 
                                   if d.timestamp >= value]
            elif key == "end_time":
                filtered_decisions = [d for d in filtered_decisions 
                                   if d.timestamp <= value]
        
        return filtered_decisions
    
    def _export_jsonl(self, decisions: List[DecisionOutcome], filepath: Union[str, Path]) -> str:
        """Export decisions to JSONL format."""
        filepath = Path(filepath)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for decision in decisions:
                f.write(json.dumps(decision.dict(), default=str) + '\n')
        
        self.logger.info(f"Exported {len(decisions)} decisions to {filepath}")
        return str(filepath)
    
    def _export_json(self, decisions: List[DecisionOutcome], filepath: Union[str, Path]) -> str:
        """Export decisions to JSON format."""
        filepath = Path(filepath)
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_decisions": len(decisions),
            "decisions": [decision.dict() for decision in decisions]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(decisions)} decisions to {filepath}")
        return str(filepath)
    
    def _export_csv(self, decisions: List[DecisionOutcome], filepath: Union[str, Path]) -> str:
        """Export decisions to CSV format."""
        filepath = Path(filepath)
        
        import csv
        
        # Define CSV headers
        headers = [
            "decision_id", "timestamp", "tool_name", "success", "risk_level",
            "trading_mode", "supervisor_confidence", "risk_officer_approval",
            "constraints_count", "execution_time"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for decision in decisions:
                row = [
                    decision.decision_id,
                    decision.timestamp.isoformat(),
                    decision.supervisor_decision.proposal.tool_name,
                    decision.success,
                    decision.supervisor_decision.risk_level,
                    decision.supervisor_decision.context.trading_mode,
                    decision.supervisor_decision.confidence,
                    decision.risk_officer_decision.final_approval,
                    len(decision.risk_officer_decision.approved_constraints),
                    decision.execution_time.isoformat() if decision.execution_time else ""
                ]
                writer.writerow(row)
        
        self.logger.info(f"Exported {len(decisions)} decisions to {filepath}")
        return str(filepath)
    
    def _persist_decision(self, decision: DecisionOutcome):
        """Persist decision to storage files."""
        try:
            # JSONL storage
            if self.enable_jsonl:
                self._write_jsonl(decision)
                self.storage_stats["jsonl_writes"] += 1
            
            # Parquet storage (if pandas is available)
            if self.enable_parquet:
                try:
                    self._write_parquet(decision)
                    self.storage_stats["parquet_writes"] += 1
                except ImportError:
                    self.logger.warning("Pandas not available, skipping Parquet storage")
                    self.enable_parquet = False
        
        except Exception as e:
            self.logger.error(f"Failed to persist decision {decision.decision_id}: {e}")
    
    def _write_jsonl(self, decision: DecisionOutcome):
        """Write decision to JSONL file."""
        with open(self.jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision.dict(), default=str) + '\n')
    
    def _write_parquet(self, decision: DecisionOutcome):
        """Write decision to Parquet file."""
        try:
            import pandas as pd
            
            # Convert decision to DataFrame row
            decision_dict = decision.dict()
            decision_dict['timestamp'] = pd.to_datetime(decision_dict['timestamp'])
            if decision_dict.get('execution_time'):
                decision_dict['execution_time'] = pd.to_datetime(decision_dict['execution_time'])
            
            df_row = pd.DataFrame([decision_dict])
            
            # Append to existing file or create new one
            if self.parquet_file.exists():
                existing_df = pd.read_parquet(self.parquet_file)
                updated_df = pd.concat([existing_df, df_row], ignore_index=True)
            else:
                updated_df = df_row
            
            updated_df.to_parquet(self.parquet_file, index=False)
            
        except ImportError:
            raise ImportError("Pandas is required for Parquet storage")
    
    def _cleanup_memory(self):
        """Clean up old decisions from memory."""
        if len(self.decisions) <= self.max_memory_decisions:
            return
        
        # Sort by timestamp and keep only the most recent
        sorted_decisions = sorted(
            self.decisions.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep the most recent decisions
        decisions_to_keep = dict(sorted_decisions[:self.max_memory_decisions])
        
        # Remove old decisions
        removed_count = len(self.decisions) - len(decisions_to_keep)
        self.decisions = decisions_to_keep
        
        # Clean up metadata
        self.decision_metadata = {
            k: v for k, v in self.decision_metadata.items() 
            if k in self.decisions
        }
        
        self.storage_stats["memory_cleanups"] += 1
        self.logger.info(f"Memory cleanup: removed {removed_count} old decisions")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage."""
        storage_info = {
            "storage_directory": str(self.storage_dir),
            "jsonl_enabled": self.enable_jsonl,
            "parquet_enabled": self.enable_parquet,
            "jsonl_file_size": self.jsonl_file.stat().st_size if self.jsonl_file.exists() else 0,
            "parquet_file_size": self.parquet_file.stat().st_size if self.parquet_file.exists() else 0,
            "memory_usage": len(self.decisions),
            "max_memory_decisions": self.max_memory_decisions,
            "total_decisions_stored": self.total_decisions,
            "storage_stats": self.storage_stats
        }
        
        return storage_info
    
    def clear_all(self):
        """Clear all stored decisions (use with caution)."""
        self.decisions.clear()
        self.decision_metadata.clear()
        self.total_decisions = 0
        
        # Remove storage files
        if self.jsonl_file.exists():
            self.jsonl_file.unlink()
        if self.parquet_file.exists():
            self.parquet_file.unlink()
        
        self.logger.warning("All decisions cleared from store")
    
    def backup(self, backup_dir: str = None) -> str:
        """Create a backup of all decisions."""
        if backup_dir is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.storage_dir / f"backup_{timestamp}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Export all decisions
        jsonl_backup = backup_path / "decisions.jsonl"
        json_backup = backup_path / "decisions.json"
        csv_backup = backup_path / "decisions.csv"
        
        self._export_jsonl(list(self.decisions.values()), jsonl_backup)
        self._export_json(list(self.decisions.values()), json_backup)
        self._export_csv(list(self.decisions.values()), csv_backup)
        
        # Copy original files if they exist
        if self.jsonl_file.exists():
            import shutil
            shutil.copy2(self.jsonl_file, backup_path / "original_decisions.jsonl")
        
        self.logger.info(f"Backup created at {backup_path}")
        return str(backup_path)
