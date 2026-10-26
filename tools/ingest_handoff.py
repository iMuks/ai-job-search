#!/usr/bin/env python3
"""Ingest job files dropped into job_scraper/inbox/ by an external watcher.

The ios-jobs poller (a separate tool, outside this repo) writes each batch of
new roles as a JSON array of {id, title, company, location, posted, url,
source} objects. This converts them to seen_jobs.json entries with status
`new`, so `/rank` picks them up exactly like `/scrape` output, then moves each
processed file to job_scraper/inbox_processed/ (outside the watched folder).

Dedup reuses scrape_state.select (URL, company+title, tracker), so a role the
workspace already knows is counted and dropped, never stored twice.

Usage:
  python3 tools/ingest_handoff.py            # ingest every inbox file
  python3 tools/ingest_handoff.py --dry-run  # report only, write nothing

Prints one JSON object: {"files": n, "added": [keys], "already_seen": n, ...}.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from job_key import make_key
from rank_state import STATE, TRACKER, load_state, save_state, tracker_pairs
from scrape_state import select

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "job_scraper" / "inbox"
PORTAL = "ios-jobs-watch"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def to_entry(job: dict, today: str) -> dict:
    posted = job.get("posted") or ""
    return {
        "title": str(job.get("title") or "").strip(),
        "company": str(job.get("company") or "").strip(),
        "url": str(job.get("url") or "").strip(),
        "location": str(job.get("location") or "").strip(),
        "first_seen": today,
        "posted_date": posted if DATE_RE.match(posted) else None,
        "deadline": None,
        "fit": "medium",          # placeholder; /rank does the real scoring
        "status": "new",
        "portal": PORTAL,
        "source": "cli",
        "origin": job.get("source"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(p for p in INBOX.glob("*.json") if not p.name.startswith("."))
    today = date.today().isoformat()
    entries, bad_files = [], []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("not a JSON array")
            entries += [to_entry(j, today) for j in data if isinstance(j, dict)]
        except (OSError, ValueError) as exc:
            bad_files.append({"file": path.name, "error": str(exc)})

    if STATE.exists():
        doc, seen = load_state(STATE)
    else:
        doc = {"seen": {}}
        seen = doc["seen"]
    selected, counts = select(entries, seen, tracker_pairs(TRACKER))

    added, collisions = [], []
    for job in selected:
        key = make_key(job["company"], job["title"], job["url"])
        if key in seen:
            collisions.append(key)
            continue
        seen[key] = job
        added.append(key)

    if not args.dry_run:
        if added:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            save_state(STATE, doc)
        done = INBOX.parent / "inbox_processed"
        done.mkdir(parents=True, exist_ok=True)
        bad = {b["file"] for b in bad_files}
        for path in files:
            target = done / (("failed-" if path.name in bad else "") + path.name)
            path.replace(target)

    print(json.dumps({"files": len(files), "added": added, "key_collisions": collisions,
                      "bad_files": bad_files, **counts}, ensure_ascii=False))
    return 1 if bad_files else 0


if __name__ == "__main__":
    sys.exit(main())
