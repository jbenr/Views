"""Infer display groups from recent execution activity, not strategy ownership."""
from datetime import datetime, timezone


def group_positions(positions, fills, window_seconds=120):
    latest = {}
    for fill in fills:
        try:
            stamp = datetime.fromisoformat(fill["time"])
        except (KeyError, TypeError, ValueError):
            continue
        # Unzoned timestamps cannot safely be compared across broker sessions.
        if stamp.tzinfo is None or not fill.get("con_id"):
            continue
        stamp = stamp.astimezone(timezone.utc)
        key = fill["con_id"]
        latest[key] = max(stamp, latest.get(key, stamp))
    timed, unknown = [], []
    for position in positions:
        stamp = latest.get(position.get("con_id"))
        if stamp is None:
            unknown.append(position)
        else:
            timed.append((stamp, position))
    groups = []
    for stamp, position in sorted(timed, key=lambda item: item[0], reverse=True):
        if not groups or (groups[-1][0] - stamp).total_seconds() > window_seconds:
            groups.append((stamp, []))
        groups[-1][1].append(position)
    if unknown:
        groups.append((None, unknown))
    return groups
