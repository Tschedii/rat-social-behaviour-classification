"""SQLite-backed log of every deployment-time prediction, reviewable and correctable, and
exportable into the training pipeline's manifest/label-CSV format for a retraining cycle.

Design (see the "continual retraining" discussion this was built from):
  - Every window inferred by 6_deployment/infer_session.py is logged here automatically — not
    just corrections — so any prediction can later be pulled up and reviewed.
  - A human reviews via 6_deployment/review_gui.py, confidence-sorted ascending (least confident
    first) for active-learning-style triage: correcting the model's most uncertain calls is
    far higher value per minute spent than reviewing everything uniformly.
  - Only rows a human has explicitly confirmed or corrected (status != 'pending') are ever
    eligible for export into a retraining cycle — an unreviewed prediction, even a
    high-confidence one, is never treated as ground truth.
  - Rows track exported_at so a retraining export never double-includes a window, and the
    ORIGINAL deployment eval split is never touched by any of this — corrections only ever
    become new TRAIN rows (see export_corrections_for_retrain.py).

One row = one (session, camera set, rat, window_start) prediction.
"""
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".rat_deploy_corrections.db"
DEFAULT_SUBDIRS = {"rgb": "cutout_all_objects", "mask1": "mask_object_1", "mask2": "mask_object_2"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_label TEXT NOT NULL,
    camera_dirs TEXT NOT NULL,      -- JSON list, 6 entries, cam_1..cam_6 order
    rat INTEGER NOT NULL,           -- 1 or 2 (which mask/object was used)
    window_start INTEGER NOT NULL,
    seq_len INTEGER NOT NULL DEFAULT 32,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    checkpoint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | confirmed | corrected
    corrected_label TEXT,
    reviewer_note TEXT,
    reviewed_at TEXT,
    exported_at TEXT,
    subdirs TEXT                    -- JSON {"rgb", "mask1", "mask2"}: the camera subfolder names used
);
CREATE INDEX IF NOT EXISTS idx_status_conf ON predictions(status, confidence);
"""


def init_db(db_path=None):
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if "subdirs" not in columns:  # logs created before subfolder names were recorded
        conn.execute("ALTER TABLE predictions ADD COLUMN subdirs TEXT")
    conn.commit()
    return conn


def normalize_subdirs(subdirs):
    """{"rgb", "mask1", "mask2"} from either that form or infer_session's {"rgb", 1, 2}."""
    subdirs = subdirs or {}
    return {
        "rgb": subdirs.get("rgb") or DEFAULT_SUBDIRS["rgb"],
        "mask1": subdirs.get("mask1") or subdirs.get(1) or DEFAULT_SUBDIRS["mask1"],
        "mask2": subdirs.get("mask2") or subdirs.get(2) or DEFAULT_SUBDIRS["mask2"],
    }


def row_subdirs(row, fallback=None):
    """The subfolder names a logged row was inferred with. Rows logged before they were
    recorded get `fallback` (or the defaults); set_subdirs() can fill those in."""
    if row.get("subdirs"):
        return normalize_subdirs(json.loads(row["subdirs"]))
    return normalize_subdirs(fallback)


def row_mask_subdir(row, fallback=None):
    return row_subdirs(row, fallback)[f"mask{row['rat']}"]


def log_window_predictions(db_path, session_label, camera_dirs, checkpoint, rat, window_predictions,
                           subdirs=None):
    """window_predictions: list of {"window_start":, "label":, "confidence":} as produced by
    infer_session.SessionInferer.run()'s internal per-window predictions (see its
    `on_window` hook). Call once per (rat, run). subdirs: the camera subfolder names the
    inference read ({"rgb", 1, 2} or {"rgb", "mask1", "mask2"}; default names if None), so the
    review tools and the retraining export read the same folders."""
    conn = init_db(db_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    cam_json = json.dumps([str(c) for c in camera_dirs])
    sub_json = json.dumps(normalize_subdirs(subdirs))
    rows = [
        (session_label, cam_json, rat, wp["window_start"], wp.get("seq_len", 32),
         wp["label"], wp["confidence"], str(checkpoint), now, sub_json)
        for wp in window_predictions
    ]
    conn.executemany(
        "INSERT INTO predictions (session_label, camera_dirs, rat, window_start, seq_len, "
        "predicted_label, confidence, checkpoint, created_at, subdirs) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    n = conn.total_changes
    conn.close()
    return len(rows)


def get_review_queue(db_path, limit=50, status="pending"):
    """Confidence-ascending: least-confident predictions first (active-learning triage)."""
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions WHERE status = ? ORDER BY confidence ASC LIMIT ?",
        (status, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def confirm_correct(db_path, row_id, note=None):
    conn = init_db(db_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        "UPDATE predictions SET status='confirmed', reviewed_at=?, reviewer_note=? WHERE id=?",
        (now, note, row_id),
    )
    conn.commit()
    conn.close()


def set_correction(db_path, row_id, corrected_label, note=None):
    conn = init_db(db_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        "UPDATE predictions SET status='corrected', corrected_label=?, reviewed_at=?, "
        "reviewer_note=? WHERE id=?",
        (corrected_label, now, note, row_id),
    )
    conn.commit()
    conn.close()


def reset_to_pending(db_path, row_id):
    """Undo a verdict. Refuses rows already exported into a retraining manifest, since their
    label is already baked into a built .h5."""
    conn = init_db(db_path)
    cur = conn.execute(
        "UPDATE predictions SET status='pending', corrected_label=NULL, reviewed_at=NULL, "
        "reviewer_note=NULL WHERE id=? AND exported_at IS NULL",
        (row_id,),
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def get_exportable(db_path):
    """Rows a human has reviewed (confirmed or corrected) and that haven't been exported into
    a retraining manifest yet."""
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions WHERE status IN ('confirmed','corrected') "
        "AND exported_at IS NULL ORDER BY session_label, rat, window_start"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_exported(db_path, row_ids):
    if not row_ids:
        return
    conn = init_db(db_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.executemany(
        "UPDATE predictions SET exported_at=? WHERE id=?", [(now, rid) for rid in row_ids]
    )
    conn.commit()
    conn.close()


def exclude_sessions(db_path, session_labels, reason="excluded: eval split"):
    """Keep these sessions out of every retraining export, including rows reviewed later, while
    keeping their verdicts. Uses exported_at (which get_exportable filters on) as the marker."""
    conn = init_db(db_path)
    cur = conn.executemany(
        "UPDATE predictions SET exported_at=? WHERE session_label=? AND exported_at IS NULL",
        [(reason, s) for s in session_labels],
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def set_subdirs(db_path, session_label, subdirs):
    """Record the subfolder names for a session's rows logged before they were recorded."""
    conn = init_db(db_path)
    cur = conn.execute("UPDATE predictions SET subdirs=? WHERE session_label=? AND subdirs IS NULL",
                       (json.dumps(normalize_subdirs(subdirs)), session_label))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def stats(db_path):
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT status, COUNT(*) as n, "
        "SUM(CASE WHEN exported_at IS NOT NULL THEN 1 ELSE 0 END) as n_exported "
        "FROM predictions GROUP BY status"
    ).fetchall()
    conn.close()
    return {r["status"]: {"count": r["n"], "exported": r["n_exported"]} for r in rows}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--set-subdirs", nargs=4, metavar=("SESSION", "RGB", "MASK1", "MASK2"),
                    help="record the camera subfolder names for a session logged before they were "
                         "recorded (older rows assume cutout_all_objects / mask_object_1 / mask_object_2)")
    args = ap.parse_args()
    if args.set_subdirs:
        session, rgb, m1, m2 = args.set_subdirs
        n = set_subdirs(args.db, session, {"rgb": rgb, "mask1": m1, "mask2": m2})
        print(f"set subfolder names on {n} rows of session '{session}'")
    if args.stats:
        s = stats(args.db)
        print(f"correction log: {args.db}")
        if not s:
            print("  (empty)")
        for status, info in s.items():
            print(f"  {status}: {info['count']} rows ({info['exported']} already exported)")


if __name__ == "__main__":
    main()
