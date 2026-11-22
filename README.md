# Run an ML System: From Raw Logs to Automated Retraining

Operate four weeks of a notification-open model the way a production team does. Reject malformed events, give every training set a content-hash version, and build features exactly as they were known when each notification was sent. Train only on outcomes that have arrived, track every run, and gate releases on every device slice. Roll out to a deterministic canary. Then watch daily quality on matured labels as an app update breaks the model for Android users, decide when to retrain, and see the gate block one retrained model and pass another.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** validate_events
- [x] **2.** dataset_version
- [ ] **3.** point_in_time_features
- [ ] **4.** matured_labels
- [ ] **5.** train_model
- [ ] **6.** ExperimentTracker
- [ ] **7.** slice_report
- [ ] **8.** ci_gate
- [ ] **9.** assign_variant
- [ ] **10.** canary_check
- [ ] **11.** daily_auc
- [ ] **12.** retrain_decision

---

Built on Deep-ML.
