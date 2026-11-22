"""
Run an ML System: From Raw Logs to Automated Retraining scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Run an ML system: four weeks of a notification-open model, from raw logs to an automated retrain.

Story: clean and version the data, build features as they were known at each moment, train only on
labels that have arrived, track the runs, gate the release on every device slice, roll it out to a
canary, then watch it decay after an app update and retrain when the matured labels say so.
"""
import numpy as np


def subset(events, first_day, last_day):
    return [e for e in events if first_day * 1440 <= e["ts"] < last_day * 1440]


def main():
    raw, feature_log, outcomes = load_history()
    events, rejected = validate_events(raw)
    print("events:", len(raw), "| rejected:", len(rejected), "| first rejections:", rejected[:3])

    # ---- Train v1 on the first week, with the labels known on day 8 ----
    week1 = subset(events, 0, 7)
    labels = matured_labels(outcomes, as_of=8 * 1440)
    train = [e for e in week1 if e["event_id"] in labels]
    version1 = dataset_version(train)
    X = point_in_time_features(train, feature_log)
    y = np.array([labels[e["event_id"]] for e in train])
    tracker = ExperimentTracker()
    models = {}
    holdout = subset(events, 7, 9)
    hold_labels = matured_labels(outcomes, as_of=11 * 1440)
    holdout = [e for e in holdout if e["event_id"] in hold_labels]
    Xh = point_in_time_features(holdout, feature_log)
    yh = np.array([hold_labels[e["event_id"]] for e in holdout])
    for C in (0.01, 0.1, 1.0):
        model = train_model(X, y, C=C)
        report = slice_report(yh, model.predict_proba(Xh)[:, 1], [e["device"] for e in holdout])
        run = tracker.log_run({"C": C}, version1, {"auc": report["overall"]["auc"]})
        models[run] = (model, report)
    best = tracker.best_run("auc", version1)
    v1, v1_report = models[best]
    print("data version", version1, "| runs:", [(r["run_id"], r["params"]["C"], r["metrics"]["auc"]) for r in tracker.runs], "| best:", best)
    print("v1 on held-out days:", v1_report)

    # ---- Watch it in production, judging each day only on labels that have arrived ----
    live = subset(events, 9, 28)
    probs = v1.predict_proba(point_in_time_features(live, feature_log))[:, 1]
    for today in (14, 18, 22):
        daily = daily_auc(live, probs, matured_labels(outcomes, as_of=today * 1440))
        due = retrain_decision(daily)
        print("day", today, "| matured daily AUC:", daily, "| retrain due on day:", due)
        if due is not None:
            break

    # ---- Retrain, gate every candidate on each device, canary the one that passes ----
    check_labels = matured_labels(outcomes, as_of=(due + 6) * 1440)
    check = [e for e in subset(events, due + 1, due + 4) if e["event_id"] in check_labels]
    Xc = point_in_time_features(check, feature_log)
    yc = np.array([check_labels[e["event_id"]] for e in check])
    devices = [e["device"] for e in check]
    base = slice_report(yc, v1.predict_proba(Xc)[:, 1], devices)
    print("v1 after the update:", base)
    known_now = matured_labels(outcomes, as_of=(due + 1) * 1440)
    for window in (3, 7):
        recent = [e for e in subset(events, due + 1 - window, due + 1) if e["event_id"] in known_now]
        version = dataset_version(recent)
        model = train_model(point_in_time_features(recent, feature_log), np.array([known_now[e["event_id"]] for e in recent]))
        cand = slice_report(yc, model.predict_proba(Xc)[:, 1], devices)
        run = tracker.log_run({"C": 1.0, "window_days": window}, version, {"auc": cand["overall"]["auc"]})
        passed, reasons = ci_gate(cand, base)
        print(run, "| last", window, "days, data version", version, "| overall", base["overall"]["auc"], "->", cand["overall"]["auc"],
              "| by device:", {d: (base["slices"][d]["auc"], cand["slices"][d]["auc"]) for d in base["slices"]})
        print("   CI gate passed:", passed, reasons)
        if passed:
            break
    users = sorted({e["user_id"] for e in check})
    canary_users = [u for u in users if assign_variant(u, 10) == "canary"]
    print("canary gets", len(canary_users), "of", len(users), "users")
    print("canary with 5 errors in 600 requests against 40 in 5000:", canary_check({"requests": 5000, "errors": 40}, {"requests": 600, "errors": 5}))
    print("canary with 15 errors in 600 requests against 40 in 5000:", canary_check({"requests": 5000, "errors": 40}, {"requests": 600, "errors": 15}))


if __name__ == "__main__":
    main()

