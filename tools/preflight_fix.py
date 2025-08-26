# tools/preflight_fix.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DOCS_VAL = ROOT / "docs" / "validation"
DOCS_REP = ROOT / "docs" / "reports"
DOCS_OPS_IMG = ROOT / "docs" / "ops" / "img"
TARGET = ROOT / "live_trader_clean.py"


def mkdirs():
    for p in [DOCS_VAL, DOCS_REP, DOCS_OPS_IMG]:
        p.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created dirs:\n  - {DOCS_VAL}\n  - {DOCS_REP}\n  - {DOCS_OPS_IMG}")


def write_env():
    # python version
    (DOCS_VAL / "00_python.txt").write_text(f"{sys.version}\n", encoding="utf-8", errors="ignore")
    # pip freeze
    try:
        out = subprocess.check_output([PY, "-m", "pip", "freeze"], text=True, errors="ignore")
    except Exception as e:
        out = f"# pip freeze failed: {e}\n"
    (DOCS_VAL / "00_env.txt").write_text(out, encoding="utf-8", errors="ignore")
    print("[OK] Wrote environment snapshots to docs/validation/00_*")


def compile_file(path: Path):
    try:
        subprocess.check_call([PY, "-m", "py_compile", str(path)])
        print(f"[OK] Syntax check passed: {path.name}")
        return True
    except subprocess.CalledProcessError:
        print(f"[!] Syntax/encoding check failed for {path.name}")
        return False


def detect_and_fix_encoding(path: Path) -> bool:
    raw = path.read_bytes()
    # If file already declares utf-8, we still might have bad bytes; try robust normalization.
    try:
        # Try charset-normalizer if available (bundled with requests in many envs)
        try:
            from charset_normalizer import from_bytes

            result = from_bytes(raw).best()
            if result:
                text = str(result)
                path.write_text(text, encoding="utf-8")
                print("[OK] Re-encoded using charset-normalizer → UTF-8")
                return True
        except Exception:
            pass

        # Fallback 1: try cp1252 then UTF-8 re-write
        try:
            text = raw.decode("cp1252")
            path.write_text(text, encoding="utf-8")
            print("[OK] Re-encoded from cp1252 → UTF-8 (fallback)")
            return True
        except Exception:
            pass

        # Fallback 2: decode with 'latin-1' lossless then write UTF-8
        text = raw.decode("latin-1")
        path.write_text(text, encoding="utf-8")
        print("[OK] Re-encoded from latin-1 → UTF-8 (fallback)")
        return True
    except Exception as e:
        print(f"[ERR] Re-encode failed: {e}")
        return False


def ensure_coding_cookie(path: Path):
    # Make sure file has a proper UTF-8 coding cookie
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        cookie = "# -*- coding: utf-8 -*-"
        if not any("coding:" in ln for ln in lines[:2]):
            # Insert cookie on second line (after shebang if present)
            if lines and lines[0].startswith("#!"):
                lines.insert(1, cookie)
            else:
                lines.insert(0, cookie)
            path.write_text("\n".join(lines), encoding="utf-8")
            print("[OK] Added UTF-8 coding cookie to the file header")
    except Exception as e:
        print(f"[WARN] Could not ensure coding cookie: {e}")


def main():
    print("== MR BEN Preflight & Unicode Repair ==")
    mkdirs()
    write_env()

    if not TARGET.exists():
        print(f"[ERR] Missing file: {TARGET}")
        sys.exit(1)

    # First attempt compile
    if compile_file(TARGET):
        print("[DONE] Preflight complete.")
        return

    print("[INFO] Attempting Unicode repair…")
    fixed = detect_and_fix_encoding(TARGET)
    ensure_coding_cookie(TARGET)

    if not fixed:
        print("[ERR] Could not auto-fix encoding; manual edit may be required.")
        sys.exit(2)

    # Re-compile after fix
    if compile_file(TARGET):
        print("[DONE] Preflight complete after repair.")
        return
    else:
        print(
            "[ERR] Still failing after repair; please open file and inspect around the reported byte position."
        )
        # Optional: dump context near reported position if needed.


if __name__ == "__main__":
    main()
