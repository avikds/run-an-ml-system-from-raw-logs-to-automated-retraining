"""
Run an ML System: From Raw Logs to Automated Retraining

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - validate_events
import numpy as np

DAY = 1440  # minutes

def load_history(days=28, users=400, per_day=500, seed=0, update_day=14):
    """Four weeks of push notifications, as three logs.

    events:          {"event_id", "user_id", "ts", "hour", "device"}, one per notification sent
    feature_log:     {"user_id", "ts", "sessions_7d", "opens_7d"}, a user's activity counts, updated now and then
    outcomes:        {"event_id", "opened", "label_ts"}, whether it was opened, known only at label_ts
    Times are minutes since the start. From update_day on, an app update changes how Android users respond.
    A few events are malformed, the way real logs are.
    """
    rs = np.random.RandomState(seed)
    devices = np.array(["ios", "android", "web"])
    user_device = devices[rs.choice(3, size=users, p=[0.45, 0.4, 0.15])]
    engagement = rs.gamma(2.0, 1.5, size=users)
    feature_log, events, outcomes = [], [], []

    for u in range(users):
        t = int(rs.randint(0, DAY))
        while t < days * DAY:
            sessions = int(rs.poisson(2 + 3 * engagement[u]))
            opens = int(rs.binomial(sessions, min(0.9, 0.15 + 0.1 * engagement[u])))
            feature_log.append({
                "user_id": u,
                "ts": t,
                "sessions_7d": sessions,
                "opens_7d": opens
            })
            t += int(rs.randint(DAY // 2, 3 * DAY))

    latest = {}
    for row in sorted(feature_log, key=lambda r: r["ts"]):
        latest.setdefault(row["user_id"], []).append(row)

    eid = 0
    for d in range(days):
        for _ in range(per_day):
            u = int(rs.randint(users))
            ts = d * DAY + int(rs.randint(DAY))
            hour = (ts % DAY) // 60
            history = [r for r in latest.get(u, []) if r["ts"] <= ts]
            opens = history[-1]["opens_7d"] if history else 0
            sessions = history[-1]["sessions_7d"] if history else 0
            android = user_device[u] == "android"

            if android and d >= update_day:
                # After the update, Android users open in the morning and past opens matter much less.
                z = -1.4 + 0.1 * opens - 0.04 * sessions + 2.0 * (7 <= hour <= 10) - 1.0 * (18 <= hour <= 22)
            else:
                z = (
                    -2.2 + 0.35 * opens - 0.04 * sessions
                    + 0.6 * (18 <= hour <= 22)
                    - 0.5 * (user_device[u] == "web")
                    + 0.3 * android
                )

            opened = int(rs.rand() < 1.0 / (1.0 + np.exp(-z)))

            events.append({
                "event_id": eid,
                "user_id": u,
                "ts": ts,
                "hour": int(hour),
                "device": str(user_device[u])
            })

            outcomes.append({
                "event_id": eid,
                "opened": opened,
                "label_ts": ts + int(rs.randint(DAY // 4, 2 * DAY))
            })

            eid += 1

    for i in rs.choice(len(events), size=len(events) // 50, replace=False):
        bad = dict(events[i])
        kind = i % 3

        if kind == 0:
            bad["device"] = "fridge"
        elif kind == 1:
            bad["hour"] = 25
        else:
            bad["user_id"] = None

        events[i] = bad

    rs.shuffle(feature_log)
    return events, feature_log, outcomes

# ---- end provided


DEVICES = ("ios", "android", "web")


def validate_events(events):
    """Split raw events into valid and rejected events.

    Each event is checked in the following order:
    1. user_id must be an integer (not a boolean).
    2. ts must be a non-negative integer (not a boolean).
    3. hour must be an integer between 0 and 23 (not a boolean).
    4. device must be one of the supported devices.

    Returns:
        clean: Valid events in their original order.
        rejected: (event_id, reason) pairs in their original order.
    """

    clean = []
    rejected = []

    for event in events:

        # Check 1: Validate user_id.
        user_id = event.get("user_id")

        if not isinstance(user_id, int) or isinstance(user_id, bool):
            rejected.append((event.get("event_id"), "bad user_id"))
            continue

        # Check 2: Validate timestamp.
        ts = event.get("ts")

        if not isinstance(ts, int) or isinstance(ts, bool) or ts < 0:
            rejected.append((event.get("event_id"), "bad ts"))
            continue

        # Check 3: Validate notification hour.
        hour = event.get("hour")

        if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23:
            rejected.append((event.get("event_id"), "bad hour"))
            continue

        # Check 4: Validate device type.
        device = event.get("device")

        if device not in DEVICES:
            rejected.append((event.get("event_id"), "bad device"))
            continue

        # All validation checks passed. Keep the original event unchanged.
        clean.append(event)

    return clean, rejected

# Step 2 - dataset_version
import hashlib
import json


def dataset_version(rows, key="event_id"):
    """First 12 hex chars of sha256 over the rows sorted by key, canonical JSON."""

    # Sort rows by the specified key to ensure order-independent versioning.
    sorted_rows = sorted(rows, key=lambda row: row[key])

    # Serialize rows into canonical JSON for consistent hashing.
    serialized = json.dumps(
        sorted_rows,
        sort_keys=True,
        separators=(",", ":")
    )

    # Encode the serialized JSON as UTF-8 bytes.
    encoded = serialized.encode("utf-8")

    # Generate the SHA-256 hash and return its first 12 hexadecimal characters.
    version_id = hashlib.sha256(encoded).hexdigest()[:12]

    return version_id

# Step 3 - point_in_time_features
import bisect

FEATURE_NAMES = [
    "sessions_7d",
    "opens_7d",
    "morning",
    "evening",
    "android",
    "web",
    "android_morning",
    "android_evening",
    "android_opens",
    "has_history"
]


def point_in_time_features(events, feature_log):
    """(n_events, 10) array from the latest log row at or before each event."""

    # Group feature-log entries by user.
    user_history = {}

    for row in feature_log:
        user_id = row["user_id"]
        user_history.setdefault(user_id, []).append(row)

    # Sort each user's history chronologically and store timestamps separately.
    user_timestamps = {}

    for user_id, history in user_history.items():
        history.sort(key=lambda row: row["ts"])
        user_timestamps[user_id] = [row["ts"] for row in history]

    features = []

    # Generate one feature row for every notification event.
    for event in events:
        user_id = event["user_id"]
        ts = event["ts"]
        hour = event["hour"]
        device = event["device"]

        # Find the most recent feature-log entry at or before the event timestamp.
        history = user_history.get(user_id, [])
        timestamps = user_timestamps.get(user_id, [])

        index = bisect.bisect_right(timestamps, ts) - 1

        # Use zero activity counts when no historical record is available.
        if index >= 0:
            log_row = history[index]
            sessions_7d = log_row["sessions_7d"]
            opens_7d = log_row["opens_7d"]
            has_history = 1
        else:
            sessions_7d = 0
            opens_7d = 0
            has_history = 0

        # Create time-of-day indicator features.
        morning = int(7 <= hour <= 10)
        evening = int(18 <= hour <= 22)

        # Create device indicator features.
        android = int(device == "android")
        web = int(device == "web")

        # Create interaction features.
        android_morning = android * morning
        android_evening = android * evening
        android_opens = android * opens_7d

        # Assemble features in the exact order specified by FEATURE_NAMES.
        feature_row = [
            sessions_7d,
            opens_7d,
            morning,
            evening,
            android,
            web,
            android_morning,
            android_evening,
            android_opens,
            has_history
        ]

        features.append(feature_row)

    # Return a NumPy floating-point array with shape (n_events, 10).
    return np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES))

# Step 4 - matured_labels
def matured_labels(outcomes, as_of):
    """{event_id: opened} for outcomes with label_ts <= as_of."""

    # Keep only outcomes whose labels were available by the cutoff time.
    matured = {
        outcome["event_id"]: outcome["opened"]
        for outcome in outcomes
        if outcome["label_ts"] <= as_of
    }

    return matured

# Step 5 - train_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y, C=1.0):
    """Fitted StandardScaler + LogisticRegression(C, max_iter=2000, tol=1e-8)."""

    # Create a pipeline with feature standardization followed by logistic regression.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=2000,
            tol=1e-8
        )
    )

    # Fit the complete pipeline using the training features and labels.
    model.fit(X, y)

    # Return the fitted pipeline.
    return model

# Step 6 - ExperimentTracker
class ExperimentTracker:
    """Logs runs (params, data version, metrics) and finds the best run per data version."""

    def __init__(self):
        # Store all experiment runs in chronological order.
        self.runs = []

    def log_run(self, params, data_version, metrics):
        """Store a training run and return its unique run ID."""

        # Generate sequential run IDs starting from run-001.
        run_id = f"run-{len(self.runs) + 1:03d}"

        # Store copies of parameters and metrics to prevent external
        # dictionary modifications from changing the recorded run.
        run_record = {
            "run_id": run_id,
            "params": dict(params),
            "data_version": data_version,
            "metrics": dict(metrics)
        }

        # Append the run record to the experiment history.
        self.runs.append(run_record)

        return run_id

    def best_run(self, metric, data_version, higher_is_better=True):
        """Return the ID of the best run for a metric and dataset version."""

        best = None

        # Examine runs belonging to the requested dataset version.
        for run in self.runs:
            if run["data_version"] != data_version:
                continue

            # Skip runs that do not contain the requested metric.
            if metric not in run["metrics"]:
                continue

            current_value = run["metrics"][metric]

            # Select the first eligible run as the initial best run.
            if best is None:
                best = run
                continue

            best_value = best["metrics"][metric]

            # Update only when a strictly better metric value is found.
            # Keeping strict comparisons ensures earlier runs win ties.
            if higher_is_better:
                if current_value > best_value:
                    best = run
            else:
                if current_value < best_value:
                    best = run

        # Return the best run ID, or None if no eligible run exists.
        return best["run_id"] if best is not None else None

# Step 7 - slice_report
from sklearn.metrics import roc_auc_score


def slice_report(y_true, p, slices):
    """Overall and per-slice counts and AUC (rounded to 4, None for a one-class group)."""

    # Convert inputs into NumPy arrays for efficient indexing.
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(p).ravel()
    slices = np.asarray(slices).ravel()

    def calculate_auc(y, probabilities):
        """Calculate AUC, returning None when only one class is present."""

        # AUC is undefined when the group contains fewer than two classes.
        if len(np.unique(y)) < 2:
            return None

        # Calculate ROC AUC and round the result to four decimal places.
        return round(float(roc_auc_score(y, probabilities)), 4)

    # Calculate overall sample count and AUC.
    overall = {
        "n": len(y_true),
        "auc": calculate_auc(y_true, p)
    }

    # Initialize the dictionary containing per-slice evaluation results.
    slice_results = {}

    # Process each unique slice in sorted order.
    for name in sorted(np.unique(slices).tolist()):

        # Select the samples belonging to the current slice.
        mask = slices == name

        y_group = y_true[mask]
        p_group = p[mask]

        # Calculate sample count and AUC for the current slice.
        slice_results[name] = {
            "n": len(y_group),
            "auc": calculate_auc(y_group, p_group)
        }

    # Return the overall and per-slice evaluation report.
    return {
        "overall": overall,
        "slices": slice_results
    }

# Step 8 - ci_gate
def ci_gate(candidate, baseline, tolerance=0.015, min_count=100):
    """(passed, reasons) from overall and per-slice AUC rules."""

    reasons = []

    # Extract overall AUC values from the candidate and baseline reports.
    candidate_auc = candidate["overall"]["auc"]
    baseline_auc = baseline["overall"]["auc"]

    # Rule 1: Reject the candidate if its overall AUC is lower.
    if (
        candidate_auc is not None
        and baseline_auc is not None
        and candidate_auc < baseline_auc
    ):
        reasons.append("overall auc dropped")

    # Retrieve the device-slice evaluation reports.
    candidate_slices = candidate["slices"]
    baseline_slices = baseline["slices"]

    # Evaluate every baseline slice in sorted name order.
    for name in sorted(baseline_slices):

        # Rule 2: Reject the candidate if a baseline slice is missing.
        if name not in candidate_slices:
            reasons.append(f"slice missing: {name}")
            continue

        candidate_slice = candidate_slices[name]
        baseline_slice = baseline_slices[name]

        candidate_slice_auc = candidate_slice["auc"]
        baseline_slice_auc = baseline_slice["auc"]

        # Rule 3: Check regression only when the candidate slice
        # has enough samples and both AUC values are available.
        if (
            candidate_slice["n"] >= min_count
            and candidate_slice_auc is not None
            and baseline_slice_auc is not None
        ):

            # Reject if the candidate AUC is more than the allowed
            # tolerance below the corresponding baseline AUC.
            if candidate_slice_auc < baseline_slice_auc - tolerance:
                reasons.append(f"slice regressed: {name}")

    # The candidate passes only when no release rules have failed.
    passed = len(reasons) == 0

    return passed, reasons

# Step 9 - assign_variant
def assign_variant(user_id, canary_percent, salt="release-1"):
    """'canary' if sha256('salt:user_id') bucket (first 8 hex mod 100) < canary_percent, else 'stable'."""

    # Construct a unique string using the release salt and user ID.
    user_key = f"{salt}:{user_id}"

    # Generate a SHA-256 hash of the user key.
    hash_value = hashlib.sha256(user_key.encode("utf-8")).hexdigest()

    # Convert the first 8 hexadecimal characters into an integer.
    hash_integer = int(hash_value[:8], 16)

    # Map the hash into a deterministic bucket between 0 and 99.
    bucket = hash_integer % 100

    # Assign the user to the canary group if the bucket is below the threshold.
    if bucket < canary_percent:
        return "canary"

    return "stable"

# Step 10 - canary_check (not yet solved)
# TODO: implement

# Step 11 - daily_auc (not yet solved)
# TODO: implement

# Step 12 - retrain_decision (not yet solved)
# TODO: implement

