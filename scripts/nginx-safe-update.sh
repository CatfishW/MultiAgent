#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
URL_PATH=""
ALIAS_DIR=""
SPA="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --path)
      URL_PATH="$2"
      shift 2
      ;;
    --alias)
      ALIAS_DIR="$2"
      shift 2
      ;;
    --spa)
      SPA="true"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$URL_PATH" || -z "$ALIAS_DIR" ]]; then
  echo "Usage: $0 --config <nginx.conf> --path </mount/> --alias </dir/> [--spa]" >&2
  exit 2
fi

python3 - "$CONFIG" "$URL_PATH" "$ALIAS_DIR" "$SPA" <<'PY'
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
url_path = sys.argv[2]
alias_dir = sys.argv[3]
spa_mode = sys.argv[4].lower() == "true"

if not config_path.exists():
    raise SystemExit(f"Config does not exist: {config_path}")

if not url_path.startswith("/"):
    url_path = "/" + url_path
if not url_path.endswith("/"):
    url_path = url_path + "/"

if not alias_dir.endswith("/"):
    alias_dir = alias_dir + "/"

original = config_path.read_text(encoding="utf-8")

block_lines = [
    f"    location {url_path} {{",
    f"        alias {alias_dir};",
    "        index index.html;",
]
if spa_mode:
    block_lines.append(f"        try_files $uri $uri/ {url_path}index.html;")
block_lines.append("    }")
block = "\n".join(block_lines)

pattern = re.compile(rf"(?ms)^\s*location\s+{re.escape(url_path)}\s*\{{.*?^\s*\}}")
if pattern.search(original):
    updated = pattern.sub(block, original, count=1)
else:
    stripped = original.rstrip()
    last_close = stripped.rfind("}")
    if last_close == -1:
        raise SystemExit("Could not find closing brace in nginx config")
    updated = stripped[:last_close].rstrip() + "\n\n" + block + "\n" + stripped[last_close:] + "\n"

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
backup = config_path.with_name(config_path.name + f".bak.{timestamp}")
backup.write_text(original, encoding="utf-8")
config_path.write_text(updated, encoding="utf-8")

print(f"Backup: {backup}")
print(f"Updated location: {url_path}")
print(f"Config: {config_path}")
PY
