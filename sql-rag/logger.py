import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("sql-rag") / "runs.jsonl"

def log_run(record: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        **record,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
