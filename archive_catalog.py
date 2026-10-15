"""A local SQLite index of frozen closing-price snapshots.

The gzip CSV and manifest remain the observations and their provenance.
SQLite stores only searchable metadata and an explicit local path. Every
registration and load validates the raw snapshot through ``cached_closes``;
this module never downloads, refreshes, or modifies snapshot files.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from data_cache import cached_closes


_COLUMNS = (
    "snapshot_key", "snapshot_path", "source", "interval",
    "actual_first_timestamp", "actual_last_timestamp", "index_timezone",
    "symbol_count", "row_count", "sha256", "snapshot_saved_at_utc",
    "request_json",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_key TEXT PRIMARY KEY,
    snapshot_path TEXT NOT NULL,
    source TEXT NOT NULL,
    interval TEXT NOT NULL,
    actual_first_timestamp TEXT NOT NULL,
    actual_last_timestamp TEXT NOT NULL,
    index_timezone TEXT,
    symbol_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    snapshot_saved_at_utc TEXT NOT NULL,
    request_json TEXT NOT NULL
)
"""


@contextmanager
def _catalog_connection(db_path, *, create=False):
    """Close connections reliably; queries never create a missing database."""
    path = Path(db_path).resolve()
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute(_SCHEMA)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"No local snapshot catalog: {path}")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _validated_snapshot(snapshot_dir):
    """Load the exact named cache folder and check its declared dimensions."""
    path = Path(snapshot_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"No snapshot directory: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    request = manifest["request"]
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(request_json.encode()).hexdigest()
    if path.name != key:
        raise ValueError("Keep the request-key folder name when copying a snapshot.")
    prices, manifest, hit = cached_closes(request, fetch=None, cache_dir=path.parent)
    if not hit:
        raise AssertionError("An offline snapshot load must be a cache hit.")

    actual = {
        "rows": len(prices),
        "actual_first_timestamp": prices.index[0].isoformat(),
        "actual_last_timestamp": prices.index[-1].isoformat(),
        "index_timezone": str(prices.index.tz) if prices.index.tz is not None else None,
        "actual_symbols": prices.columns.tolist(),
    }
    if any(manifest.get(field) != value for field, value in actual.items()):
        raise ValueError("Snapshot dimensions, bounds, timezone or symbols disagree with its manifest.")

    metadata = {
        "snapshot_key": key,
        "snapshot_path": str(path),
        "source": request["source"],
        "interval": request["interval"],
        "actual_first_timestamp": actual["actual_first_timestamp"],
        "actual_last_timestamp": actual["actual_last_timestamp"],
        "index_timezone": actual["index_timezone"],
        "symbol_count": len(prices.columns),
        "row_count": len(prices),
        "sha256": manifest["sha256"],
        "snapshot_saved_at_utc": manifest["snapshot_saved_at_utc"],
        "request_json": request_json,
    }
    return prices, manifest, metadata


def _assert_same_snapshot(recorded, current):
    """A request key cannot silently acquire new contents or provenance."""
    changed = [
        field for field in _COLUMNS
        if field != "snapshot_path" and recorded[field] != current[field]
    ]
    if changed:
        raise ValueError(f"Registered snapshot metadata changed: {', '.join(changed)}")


def register_snapshot(snapshot_dir, db_path, *, relocate=False):
    """Validate a frozen snapshot, index it, and return its metadata dictionary.

    ``snapshot_dir`` is the 64-character request-key directory containing
    ``prices.csv.gz`` and ``manifest.json``. ``db_path`` is a local SQLite
    filename. Repeating the same registration is idempotent. A copied/moved
    snapshot may replace the catalog's path only with ``relocate=True`` and
    identical contents/provenance. A different snapshot needs a different
    request key; this function does not replace an archive's raw files.
    """
    _, _, metadata = _validated_snapshot(snapshot_dir)
    with _catalog_connection(db_path, create=True) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_key = ?",
                (metadata["snapshot_key"],),
            ).fetchone()
            if row is None:
                placeholders = ", ".join("?" for _ in _COLUMNS)
                connection.execute(
                    f"INSERT INTO snapshots ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                    tuple(metadata[field] for field in _COLUMNS),
                )
            else:
                recorded = dict(row)
                _assert_same_snapshot(recorded, metadata)
                if recorded["snapshot_path"] != metadata["snapshot_path"]:
                    if not relocate:
                        raise ValueError("Snapshot path changed; use relocate=True to update it explicitly.")
                    connection.execute(
                        "UPDATE snapshots SET snapshot_path = ? WHERE snapshot_key = ?",
                        (metadata["snapshot_path"], metadata["snapshot_key"]),
                    )
    return metadata


def list_snapshots(db_path, *, source=None, interval=None):
    """Return metadata dictionaries, optionally filtered by exact source/interval.

    These are catalog entries, not a fresh integrity check of every raw file.
    Use ``load_snapshot`` before analyzing observations. Paths may need an
    explicit relocation after moving an archive to another computer.
    """
    conditions, values = [], []
    for field, value in (("source", source), ("interval", interval)):
        if value is not None:
            conditions.append(f"{field} = ?")
            values.append(value)
    query = "SELECT * FROM snapshots"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY source, interval, actual_first_timestamp, snapshot_key"
    with _catalog_connection(db_path) as connection:
        return [dict(row) for row in connection.execute(query, values)]


def load_snapshot(key_or_path, db_path=None):
    """Return ``(prices, manifest)`` from a validated frozen snapshot, offline.

    With ``db_path``, ``key_or_path`` is a registered snapshot key; its current
    raw metadata must still match the catalog. Without ``db_path``, pass the
    request-key snapshot directory directly. Missing/corrupt files raise an
    error; no downloader or alternative source is substituted.
    """
    if db_path is None:
        prices, manifest, _ = _validated_snapshot(key_or_path)
        return prices, manifest
    with _catalog_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM snapshots WHERE snapshot_key = ?", (str(key_or_path),)
        ).fetchone()
    if row is None:
        raise KeyError(f"Snapshot key is not registered: {key_or_path}")
    recorded = dict(row)
    prices, manifest, current = _validated_snapshot(recorded["snapshot_path"])
    _assert_same_snapshot(recorded, current)
    return prices, manifest
