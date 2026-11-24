#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التقييم والضبط: كيف تعرف أن نموذجك جيد؟
Professional TensorFlow — Evaluation

Book slug : tensorflow-professional-ar
Chapter   : ch-08-ch08-evaluation (lesson 8)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-08-ch08-evaluation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التقييم والضبط: كيف تعرف أن نموذجك جيد؟
Professional TensorFlow — Evaluation

Book slug : tensorflow-professional-ar
Chapter   : ch-08-ch08-evaluation (lesson 8)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch08_evaluation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# 1 تعني حالة تستلزم مراجعة بشرية.
y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
prob = np.array([0.91, 0.62, 0.38, 0.81, 0.44, 0.31, 0.20, 0.05])
threshold = 0.60  # يختارها الفريق على التحقق وفق سعة المراجعة والكلفة
y_pred = (prob >= threshold).astype(int)

print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)
print(f"precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}")
# المتوقع: [[4, 1], [1, 2]] ثم precision=0.67 وrecall=0.67 وF1=0.67

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch08_evaluation.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
threshold = 0.60
y_pred = (validation_prob >= threshold).astype("int32")
# Do not tune a threshold on the test set.

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch08_evaluation.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, zero_division=0))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-08-ch08-evaluation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# 1 تعني حالة تستلزم مراجعة بشرية.
y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
prob = np.array([0.91, 0.62, 0.38, 0.81, 0.44, 0.31, 0.20, 0.05])
threshold = 0.60  # يختارها الفريق على التحقق وفق سعة المراجعة والكلفة
y_pred = (prob >= threshold).astype(int)

print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)
print(f"precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}")
# المتوقع: [[4, 1], [1, 2]] ثم precision=0.67 وrecall=0.67 وF1=0.67

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-08-ch08-evaluation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
threshold = 0.60
y_pred = (validation_prob >= threshold).astype("int32")
# Do not tune a threshold on the test set.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-08-ch08-evaluation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, zero_division=0))
