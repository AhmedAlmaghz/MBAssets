#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — بنك المعادلات والخلاصات التشغيلية
PyTorch and Keras: The Complete Deep Learning Guide — Appendix C — بنك المعادلات والخلاصات التشغيلية

Book slug : pytorch-keras-deep-learning
Chapter   : app-c (lesson c)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/app-c.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — بنك المعادلات والخلاصات التشغيلية
PyTorch and Keras: The Complete Deep Learning Guide — Appendix C — بنك المعادلات والخلاصات التشغيلية

Book slug : pytorch-keras-deep-learning
Chapter   : app-c (lesson c)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/appC.html
# ────────────────────────────────────────────────────────────────────────
import sympy as sp

x, w = sp.symbols("x w", real=True)
f = w * x

# 1) اشتقاق: كم تتغيّر f عند تغيير x؟
df_dx = sp.diff(f, x)
print(df_dx)          # w

    # 2) التوزيع: أين أكبر أثر للتغيير؟
print(sp.diff(f, x, 2))   # 0 -> خطية

# 3) حلّ المعادلة: ما قيمة w التي تجعل المشتقّة صفراً؟
print(sp.solve(sp.Eq(df_dx, 0), w))   # [0]

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/app-c.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import sympy as sp

x, w = sp.symbols("x w", real=True)
f = w * x

# 1) اشتقاق: كم تتغيّر f عند تغيير x؟
df_dx = sp.diff(f, x)
print(df_dx)          # w

    # 2) التوزيع: أين أكبر أثر للتغيير؟
print(sp.diff(f, x, 2))   # 0 -> خطية

# 3) حلّ المعادلة: ما قيمة w التي تجعل المشتقّة صفراً؟
print(sp.solve(sp.Eq(df_dx, 0), w))   # [0]
