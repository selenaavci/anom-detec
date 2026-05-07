"""Persistent feedback store.

Saves user feedback (true anomaly / false alarm) per dataset across sessions
so that the model selection logic and presets can adapt over time.

Storage layout (single JSON file at ``~/.anomaly_detection_feedback.json``):

{
  "datasets": {
      "<dataset_signature>": {
          "name": str,
          "rows": int,
          "columns_selected": [...],
          "preset": str,
          "first_seen": iso8601,
          "entries": [
              {
                  "ts": iso8601,
                  "row_idx": int,
                  "label": 0|1,
                  "scores": {"isolation_forest": .., "lof": .., "ocsvm": ..,
                              "ensemble": ..},
                  "note": str,
              }
          ]
      }
  },
  "model_stats": {
      "<preset_name>": {
          "isolation_forest": {"true_positive": .., "false_positive": ..},
          ...
      }
  }
}
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path(os.path.expanduser("~/.anomaly_detection_feedback.json"))


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _load_store(path: Path = DEFAULT_PATH) -> dict:
    if not path.exists():
        return {"datasets": {}, "model_stats": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"datasets": {}, "model_stats": {}}


def dataset_signature(filename: str, n_rows: int, columns: list[str]) -> str:
    """Build a stable hash that identifies a dataset + column set."""
    h = hashlib.sha1()
    h.update(filename.encode("utf-8"))
    h.update(str(n_rows).encode("utf-8"))
    for c in sorted(columns):
        h.update(b"|")
        h.update(c.encode("utf-8"))
    return h.hexdigest()[:16]


def save_feedback(
    signature: str,
    name: str,
    n_rows: int,
    columns_selected: list[str],
    preset: str,
    entries: list[dict],
    path: Path = DEFAULT_PATH,
) -> None:
    """Persist a batch of feedback entries for a dataset.

    ``entries`` items: {row_idx, label (0/1), scores: dict, note: str}
    """
    store = _load_store(path)
    datasets = store.setdefault("datasets", {})
    bucket = datasets.setdefault(signature, {
        "name": name,
        "rows": n_rows,
        "columns_selected": columns_selected,
        "preset": preset,
        "first_seen": datetime.utcnow().isoformat(),
        "entries": [],
    })
    bucket["preset"] = preset
    bucket["columns_selected"] = columns_selected
    bucket["last_updated"] = datetime.utcnow().isoformat()

    seen = {(e.get("row_idx"), e.get("label")) for e in bucket["entries"]}
    for e in entries:
        key = (e.get("row_idx"), e.get("label"))
        if key in seen:
            continue
        e_record = {
            "ts": datetime.utcnow().isoformat(),
            "row_idx": e.get("row_idx"),
            "label": int(e.get("label", 0)),
            "scores": e.get("scores", {}),
            "note": e.get("note", ""),
        }
        bucket["entries"].append(e_record)

    _update_model_stats(store, preset, entries)
    _atomic_write(path, store)


def _update_model_stats(store: dict, preset: str, entries: list[dict]) -> None:
    stats = store.setdefault("model_stats", {})
    p = stats.setdefault(preset, {})
    for e in entries:
        label = int(e.get("label", 0))
        scores = e.get("scores", {}) or {}
        for model_key, score in scores.items():
            if model_key == "ensemble":
                continue
            row = p.setdefault(model_key, {"true_positive": 0, "false_positive": 0,
                                          "true_negative": 0, "false_negative": 0,
                                          "score_sum_tp": 0.0, "score_sum_fp": 0.0})
            try:
                # Heuristic: did the model rank this in its top region (score > 0.7 percentile)?
                model_flagged = float(score) >= 0.7
            except Exception:
                model_flagged = False
            if label == 1 and model_flagged:
                row["true_positive"] += 1
                row["score_sum_tp"] += float(score)
            elif label == 1 and not model_flagged:
                row["false_negative"] += 1
            elif label == 0 and model_flagged:
                row["false_positive"] += 1
                row["score_sum_fp"] += float(score)
            else:
                row["true_negative"] += 1


def load_feedback_for(
    signature: str,
    path: Path = DEFAULT_PATH,
) -> list[dict]:
    """Return previously-saved feedback entries for this dataset signature."""
    store = _load_store(path)
    return store.get("datasets", {}).get(signature, {}).get("entries", [])


def get_model_stats(preset: str, path: Path = DEFAULT_PATH) -> dict:
    """Aggregate true/false positive counts per model for a preset."""
    store = _load_store(path)
    return store.get("model_stats", {}).get(preset, {})


def derive_model_weights(
    preset: str,
    base_weights: dict,
    min_observations: int = 10,
    path: Path = DEFAULT_PATH,
) -> dict:
    """Adjust base preset weights based on accumulated feedback.

    Computes a precision-like score per model; models with low precision get
    their weight reduced, high precision raised. Falls back to ``base_weights``
    when too little data.
    """
    stats = get_model_stats(preset, path)
    if not stats:
        return dict(base_weights)
    weights = dict(base_weights)
    for model_key, row in stats.items():
        tp = row.get("true_positive", 0)
        fp = row.get("false_positive", 0)
        observations = tp + fp
        if observations < min_observations:
            continue
        precision = tp / observations if observations else 0
        # Map precision (0..1) into a multiplier (0.5..1.5)
        multiplier = 0.5 + precision
        weights[model_key] = round(float(weights.get(model_key, 1.0)) * multiplier, 3)
    return weights
