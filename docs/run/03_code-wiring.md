# MR BEN One-Command LIVE Run - Code Wiring Verification

**Timestamp**: 2025-08-18
**Phase**: Code verification for one-command execution
**Status**: ✅ VERIFIED

## One-Command Execution Setup

### Main Entry Point Verification ✅
**File**: `live_trader_clean.py` (lines 3150-3172)
```python
if __name__ == "__main__":
    # Check if CLI arguments are provided
    if len(sys.argv) > 1:
        # Use new CLI interface
        raise SystemExit(main())
    else:
        # NEW: run main with default LIVE settings (no legacy_main)
        sys.argv += [
            "live",
            "--mode", "live",
            "--symbol", "XAUUSD.PRO",
            "--agent",
            "--regime",
            "--log-level", "INFO"
        ]
        raise SystemExit(main())
```

**Status**: ✅ IMPLEMENTED
- Default behavior: `python live_trader_clean.py` → LIVE mode with agent and regime
- No arguments → automatic LIVE configuration injection
- Legacy mode completely removed

### CLI Argument Parser ✅
**File**: `live_trader_clean.py` (lines 222-300)
**Status**: ✅ COMPLETE
- `live` subcommand with `--mode live/paper`
- `paper` subcommand (alias for live --mode paper)
- `--agent` flag for GPT-5 supervision
- `--agent-mode` choices: observe/guard/auto
- `--regime` flag for regime detection
- `--log-level` for logging control

### Main Function Flow ✅
**File**: `live_trader_clean.py` (lines 766-800)
**Status**: ✅ IMPLEMENTED
```python
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Handle paper as alias for live --mode paper
    if args.cmd == "paper":
        args.cmd = "live"
        args.mode = "paper"

    # Execute appropriate command
    if args.cmd == "live":
        return cmd_live(args)
```

### cmd_live Function ✅
**File**: `live_trader_clean.py` (lines 577-676)
**Status**: ✅ IMPLEMENTED
- Calls `core.start()` (no mock loop)
- Attaches agent: `core.agent = bridge`
- Agent mode resolution: `_resolve_agent_mode(args, cfg)`
- Agent components initialization: `core._initialize_agent_components()`

### Agent Mode Resolution ✅
**File**: `live_trader_clean.py` (lines 200-220)
**Status**: ✅ IMPLEMENTED
```python
def _resolve_agent_mode(args, cfg):
    if getattr(args, "agent_mode", None):
        return args.agent_mode
    env = os.getenv("AGENT_MODE")
    if env in {"observe","guard","auto"}:
        return env
    try:
        return cfg.config_data.get("agent", {}).get("mode", "guard")
    except Exception:
        return "guard"
```

### Production Preflight ✅
**File**: `live_trader_clean.py` (lines 1000-1050)
**Status**: ✅ IMPLEMENTED
- `_preflight_production()` called at start of `start()`
- Validates credentials, symbol availability, kill-switch
- Raises `RuntimeError` for critical failures

### Dashboard Independence ✅
**File**: `live_trader_clean.py` (lines 400-450)
**Status**: ✅ IMPLEMENTED
- Dashboard starts independently of agent
- Always enabled by default: `dash_cfg.get("enabled", True)`
- Port configurable: `dash_cfg.get("port", 8765)`

## Verification Summary

### ✅ Code Points Verified
1. **Main Entry**: `__main__` default argv injection present
2. **CLI Flow**: `main()` → `cmd_live()` → `core.start()`
3. **Agent Integration**: `core.agent = bridge` properly set
4. **Preflight**: `_preflight_production()` called at start
5. **Dashboard**: Independent of agent, always enabled
6. **Mode Resolution**: CLI → ENV → config.json → "guard" default

### ✅ One-Command Behavior
- `python live_trader_clean.py` → LIVE mode with agent supervision
- `python live_trader_clean.py paper` → Paper mode with agent supervision
- `python live_trader_clean.py live --mode live` → Explicit LIVE mode
- All modes support `--agent`, `--regime`, `--log-level`

### ✅ Safety Features
- Production preflight checks before trading
- Kill-switch detection (`RUN_STOP` file)
- Agent supervision in guard mode by default
- Dashboard monitoring always available

## Next Steps
1. ✅ Paper sanity test completed
2. ✅ Code wiring verified
3. 🔄 Prepare configuration for LIVE mode
4. 🔄 Execute first LIVE run
5. 🔄 Verify live trade execution
