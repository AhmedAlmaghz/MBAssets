#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التثبيت والإعداد على Python 3.14
Mastering scikit-learn 1.9 — Python

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-03-ch03 (lesson 3)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-03-ch03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — التثبيت والإعداد على Python 3.14
Mastering scikit-learn 1.9 — Python

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-03-ch03 (lesson 3)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
import platform
import sys
from pathlib import Path

exe = Path(sys.executable)
print("sys.executable :", exe)
print("base_prefix    :", sys.base_prefix)
print("prefix         :", sys.prefix)
print("inside a venv? :", sys.prefix != sys.base_prefix)
print("python version :", sys.version.split()[0])
print("platform       :", platform.system(), platform.machine())
print("sys.path entries:", len(sys.path))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
# على ويندوز (PowerShell)
py -3.14 -m venv .venv

# تفعيلها
.\.venv\Scripts\Activate.ps1

# على لينكس وماك
python3.14 -m venv .venv
source .venv/bin/activate

# التحقّق
python -c "import sys; print(sys.prefix != sys.base_prefix)"

# ────────────────────────────────────────────────────────────────────────
# [003] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
from importlib.metadata import requires, version

from sklearn import get_config

CORE = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
        "joblib", "threadpoolctl", "narwhals"]

print("== الإصدارات ==")
for pkg in CORE:
    try:
        print(f"  {pkg:<16} {version(pkg)}")
    except Exception:
        print(f"  {pkg:<16} غير مثبّت")

print("\n== ما يطلبه scikit-learn إجبارياً ==")
print("  ", requires("scikit-learn"))

print("\n== إعدادات المكتبة الافتراضية ==")
for key in ("display", "transform_output", "enable_metadata_routing",
            "assume_finite", "working_memory"):
    print(f"  {key:<26} {get_config()[key]}")

print("\n== خيوط الحساب ==")
import threadpoolctl
for d in threadpoolctl.threadpool_info():
    print(f"  {d['user_api']:<8} threads={d['num_threads']}  {d.get('internal_api', '')}")

# ────────────────────────────────────────────────────────────────────────
# [004] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
import hashlib
from importlib.metadata import version

CORE = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "joblib"]


def environment_fingerprint(names=CORE) -> dict:
    """Versions + a short hash, for pinning in a run log."""
    pins = {}
    for name in names:
        try:
            pins[name] = version(name)
        except Exception:
            pins[name] = None
    blob = "|".join(f"{k}=={v}" for k, v in sorted(pins.items()))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return {"pins": pins, "hash": digest}


fp = environment_fingerprint()
for name, ver in fp["pins"].items():
    print(f"  {name:<14} {ver}")
print("\nfingerprint:", fp["hash"])

# نفس البيئة تُعطي نفس البصمة دائماً
print("stable:", environment_fingerprint()["hash"] == fp["hash"])
print("\n# requirements.txt لهذا الكتاب:")
print("\n".join(f"{k}=={v}" for k, v in fp["pins"].items() if v))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=400, n_features=5, random_state=11)

runs = []
for _ in range(2):
    lr = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    kn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    runs.append((lr.coef_.copy(), kn.predict(X)))

print("LogisticRegression  identical:", np.array_equal(runs[0][0], runs[1][0]))
print("KNeighborsClassifier identical:", np.array_equal(runs[0][1], runs[1][1]))
print("coef checksum:", round(float(np.abs(runs[0][0]).sum()), 10))

print("\nمصدرا الاختلاف الحقيقيان:")
print("  1. نسخة الحزمة:      تغيّر scipy قد يغيّر المحلّل عددياً")
print("  2. عدد خيوط BLAS:    الجمع المتوازي غير حتمي الترتيب")
print("\nالحلّ عند تسجيل نتيجة حرجة:")
print("  OMP_NUM_THREADS=1  python train.py")

# ────────────────────────────────────────────────────────────────────────
# [006] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context, get_config

print("الحالة الراهنة عند بناء هذا الكتاب:")
for key in ("display", "transform_output", "enable_metadata_routing",
            "assume_finite", "working_memory", "print_changed_only"):
    print(f"  {key:<26} {get_config()[key]}")

from sklearn.preprocessing import StandardScaler

with config_context(display="text"):
    print("\nداخل config_context(display='text'):", get_config()["display"])
print("بعد الخروج من الكتلة      :", get_config()["display"])

with config_context(transform_output="default"):
    out_default = StandardScaler().fit_transform(X)
print("تحت transform_output='default' :", type(out_default).__name__)
with config_context(transform_output="pandas"):
    out_pandas = StandardScaler().fit_transform(X)
print("تحت transform_output='pandas'  :", type(out_pandas).__name__)
print("القيم متطابقة:", np.allclose(np.asarray(out_default, dtype=float),
                                  out_pandas.to_numpy()))

# ────────────────────────────────────────────────────────────────────────
# [007] content/part01/ch03.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification

SCHEMA = {"age": (18.0, 90.0), "income": (0.0, 5e6), "tenure_months": (0.0, 720.0)}


def validate(batch: dict) -> tuple[list[str], dict]:
    """Check a raw payload against the schema the model was trained on."""
    problems: list[str] = []
    for col in SCHEMA:
        if col not in batch:
            problems.append(f"عمود ناقص: {col}")
    for col, (lo, hi) in SCHEMA.items():
        if col in batch:
            v = float(batch[col])
            if np.isnan(v):
                problems.append(f"قيمة مفقودة: {col}")
            elif not (lo <= v <= hi):
                problems.append(f"خارج النطاق: {col}={v} (المسموح {lo}..{hi})")
    return problems, batch


cases = [
    {"age": 41, "income": 82000, "tenure_months": 36},
    {"age": 12, "income": 82000, "tenure_months": 36},
    {"age": 41, "income": np.nan, "tenure_months": 36},
    {"age": 41, "income": 82000},
]
for case in cases:
    problems, _ = validate(case)
    print(f"{str(case)[:58]:<60} -> {problems or 'سليم'}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import platform
import sys
from pathlib import Path

exe = Path(sys.executable)
print("sys.executable :", exe)
print("base_prefix    :", sys.base_prefix)
print("prefix         :", sys.prefix)
print("inside a venv? :", sys.prefix != sys.base_prefix)
print("python version :", sys.version.split()[0])
print("platform       :", platform.system(), platform.machine())
print("sys.path entries:", len(sys.path))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# على ويندوز (PowerShell)
py -3.14 -m venv .venv

# تفعيلها
.\.venv\Scripts\Activate.ps1

# على لينكس وماك
python3.14 -m venv .venv
source .venv/bin/activate

# التحقّق
python -c "import sys; print(sys.prefix != sys.base_prefix)"

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from importlib.metadata import requires, version

from sklearn import get_config

CORE = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
        "joblib", "threadpoolctl", "narwhals"]

print("== الإصدارات ==")
for pkg in CORE:
    try:
        print(f"  {pkg:<16} {version(pkg)}")
    except Exception:
        print(f"  {pkg:<16} غير مثبّت")

print("\n== ما يطلبه scikit-learn إجبارياً ==")
print("  ", requires("scikit-learn"))

print("\n== إعدادات المكتبة الافتراضية ==")
for key in ("display", "transform_output", "enable_metadata_routing",
            "assume_finite", "working_memory"):
    print(f"  {key:<26} {get_config()[key]}")

print("\n== خيوط الحساب ==")
import threadpoolctl
for d in threadpoolctl.threadpool_info():
    print(f"  {d['user_api']:<8} threads={d['num_threads']}  {d.get('internal_api', '')}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import hashlib
from importlib.metadata import version

CORE = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "joblib"]


def environment_fingerprint(names=CORE) -> dict:
    """Versions + a short hash, for pinning in a run log."""
    pins = {}
    for name in names:
        try:
            pins[name] = version(name)
        except Exception:
            pins[name] = None
    blob = "|".join(f"{k}=={v}" for k, v in sorted(pins.items()))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return {"pins": pins, "hash": digest}


fp = environment_fingerprint()
for name, ver in fp["pins"].items():
    print(f"  {name:<14} {ver}")
print("\nfingerprint:", fp["hash"])

# نفس البيئة تُعطي نفس البصمة دائماً
print("stable:", environment_fingerprint()["hash"] == fp["hash"])
print("\n# requirements.txt لهذا الكتاب:")
print("\n".join(f"{k}=={v}" for k, v in fp["pins"].items() if v))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=400, n_features=5, random_state=11)

runs = []
for _ in range(2):
    lr = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    kn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    runs.append((lr.coef_.copy(), kn.predict(X)))

print("LogisticRegression  identical:", np.array_equal(runs[0][0], runs[1][0]))
print("KNeighborsClassifier identical:", np.array_equal(runs[0][1], runs[1][1]))
print("coef checksum:", round(float(np.abs(runs[0][0]).sum()), 10))

print("\nمصدرا الاختلاف الحقيقيان:")
print("  1. نسخة الحزمة:      تغيّر scipy قد يغيّر المحلّل عددياً")
print("  2. عدد خيوط BLAS:    الجمع المتوازي غير حتمي الترتيب")
print("\nالحلّ عند تسجيل نتيجة حرجة:")
print("  OMP_NUM_THREADS=1  python train.py")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn import config_context, get_config

print("الحالة الراهنة عند بناء هذا الكتاب:")
for key in ("display", "transform_output", "enable_metadata_routing",
            "assume_finite", "working_memory", "print_changed_only"):
    print(f"  {key:<26} {get_config()[key]}")

from sklearn.preprocessing import StandardScaler

with config_context(display="text"):
    print("\nداخل config_context(display='text'):", get_config()["display"])
print("بعد الخروج من الكتلة      :", get_config()["display"])

with config_context(transform_output="default"):
    out_default = StandardScaler().fit_transform(X)
print("تحت transform_output='default' :", type(out_default).__name__)
with config_context(transform_output="pandas"):
    out_pandas = StandardScaler().fit_transform(X)
print("تحت transform_output='pandas'  :", type(out_pandas).__name__)
print("القيم متطابقة:", np.allclose(np.asarray(out_default, dtype=float),
                                  out_pandas.to_numpy()))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-03-ch03.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import make_classification

SCHEMA = {"age": (18.0, 90.0), "income": (0.0, 5e6), "tenure_months": (0.0, 720.0)}


def validate(batch: dict) -> tuple[list[str], dict]:
    """Check a raw payload against the schema the model was trained on."""
    problems: list[str] = []
    for col in SCHEMA:
        if col not in batch:
            problems.append(f"عمود ناقص: {col}")
    for col, (lo, hi) in SCHEMA.items():
        if col in batch:
            v = float(batch[col])
            if np.isnan(v):
                problems.append(f"قيمة مفقودة: {col}")
            elif not (lo <= v <= hi):
                problems.append(f"خارج النطاق: {col}={v} (المسموح {lo}..{hi})")
    return problems, batch


cases = [
    {"age": 41, "income": 82000, "tenure_months": 36},
    {"age": 12, "income": 82000, "tenure_months": 36},
    {"age": 41, "income": np.nan, "tenure_months": 36},
    {"age": 41, "income": 82000},
]
for case in cases:
    problems, _ = validate(case)
    print(f"{str(case)[:58]:<60} -> {problems or 'سليم'}")
