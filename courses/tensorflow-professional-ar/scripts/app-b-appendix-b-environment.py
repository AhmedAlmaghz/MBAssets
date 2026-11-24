#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الملحق B: دليل البيئة والأعطال
Professional TensorFlow — B Environment

Book slug : tensorflow-professional-ar
Chapter   : app-b-appendix-b-environment (lesson b)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/app-b-appendix-b-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الملحق B: دليل البيئة والأعطال
Professional TensorFlow — B Environment

Book slug : tensorflow-professional-ar
Chapter   : app-b-appendix-b-environment (lesson b)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/appendix/appendix_b_environment.md
# ────────────────────────────────────────────────────────────────────────
import sys
import keras
import tensorflow as tf

print(sys.executable)
print(tf.__version__, keras.__version__)
print(tf.config.list_physical_devices())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/app-b-appendix-b-environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import sys
import keras
import tensorflow as tf

print(sys.executable)
print(tf.__version__, keras.__version__)
print(tf.config.list_physical_devices())
