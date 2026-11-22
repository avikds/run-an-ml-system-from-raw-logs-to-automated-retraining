```
events: 14000 | rejected: 280 | first rejections: [(121, 'bad hour'), (192, 'bad device'), (208, 'bad hour')]
data version 6bd65340f448 | runs: [('run-001', 0.01, 0.7858), ('run-002', 0.1, 0.7916), ('run-003', 1.0, 0.7947)] | best: run-003
v1 on held-out days: {'overall': {'n': 975, 'auc': 0.7947}, 'slices': {'android': {'n': 394, 'auc': 0.7858}, 'ios': {'n': 448, 'auc': 0.813}, 'web': {'n': 133, 'auc': 0.7166}}}
day 14 | matured daily AUC: {9: 0.832, 10: 0.8334, 11: 0.7937, 12: 0.806} | retrain due on day: None
day 18 | matured daily AUC: {9: 0.832, 10: 0.8334, 11: 0.7937, 12: 0.8212, 13: 0.7833, 14: 0.7091, 15: 0.7014, 16: 0.6684} | retrain due on day: 15
v1 after the update: {'overall': {'n': 1477, 'auc': 0.7107}, 'slices': {'android': {'n': 594, 'auc': 0.5115}, 'ios': {'n': 648, 'auc': 0.8303}, 'web': {'n': 235, 'auc': 0.8265}}}
run-004 | last 3 days, data version 56de03e04f0a | overall 0.7107 -> 0.7695 | by device: {'android': (0.5115, 0.6601), 'ios': (0.8303, 0.8298), 'web': (0.8265, 0.8062)}
   CI gate passed: False ['slice regressed: web']
run-005 | last 7 days, data version bb5f7234c5e8 | overall 0.7107 -> 0.7447 | by device: {'android': (0.5115, 0.5857), 'ios': (0.8303, 0.8323), 'web': (0.8265, 0.8175)}
   CI gate passed: True []
canary gets 35 of 391 users
canary with 5 errors in 600 requests against 40 in 5000: promote
canary with 15 errors in 600 requests against 40 in 5000: rollback
```
