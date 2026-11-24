#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — النشر: LiteRT وTF.js وTF Serving
Professional TensorFlow — Deployment

Book slug : tensorflow-professional-ar
Chapter   : ch-19-ch19-deployment (lesson 19)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-19-ch19-deployment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — النشر: LiteRT وTF.js وTF Serving
Professional TensorFlow — Deployment

Book slug : tensorflow-professional-ar
Chapter   : ch-19-ch19-deployment (lesson 19)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch19_deployment.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(lite_model)
print(len(lite_model))
# المتوقع: حجم ملف بالبايت؛ افحص الدقة على مجموعة تحقق بعد التحويل.

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch19_deployment.md
# ────────────────────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print(interpreter.get_input_details())

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch19_deployment.md
# ────────────────────────────────────────────────────────────────────────
np.testing.assert_allclose(reference, converted, rtol=0.05, atol=0.05)

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch19_deployment.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="classifier.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
batch = np.zeros((1, 224, 224, 3), dtype=inp["dtype"])
interpreter.set_tensor(inp["index"], batch)
interpreter.invoke()
scores = interpreter.get_tensor(out["index"])
print(scores.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-19-ch19-deployment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(lite_model)
print(len(lite_model))
# المتوقع: حجم ملف بالبايت؛ افحص الدقة على مجموعة تحقق بعد التحويل.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-19-ch19-deployment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print(interpreter.get_input_details())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-19-ch19-deployment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
np.testing.assert_allclose(reference, converted, rtol=0.05, atol=0.05)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-19-ch19-deployment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="classifier.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
batch = np.zeros((1, 224, 224, 3), dtype=inp["dtype"])
interpreter.set_tensor(inp["index"], batch)
interpreter.invoke()
scores = interpreter.get_tensor(out["index"])
print(scores.shape)
