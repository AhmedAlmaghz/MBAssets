#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — البيئة: من الصفر إلى أول استيراد ناجح
Keras 3 in Arabic — Setup Environment

Book slug : keras-3-arabic-professional
Chapter   : ch-02-ch02-setup-environment (lesson 2)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-02-ch02-setup-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — البيئة: من الصفر إلى أول استيراد ناجح
Keras 3 in Arabic — Setup Environment

Book slug : keras-3-arabic-professional
Chapter   : ch-02-ch02-setup-environment (lesson 2)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-02-ch02-setup-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — البيئة: من الصفر إلى أول استيراد ناجح
Keras 3 in Arabic — Setup Environment

Book slug : keras-3-arabic-professional
Chapter   : ch-02-ch02-setup-environment (lesson 2)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch02_setup_environment/009_inside.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : من داخل الكود
# المصدر  : src/chapters/ch02_setup_environment.md:127
# الترتيب : 9/13
# ──────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch02_setup_environment/010_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : التحقّق
# المصدر  : src/chapters/ch02_setup_environment.md:136
# الترتيب : 10/13
# ──────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch02_setup_environment/011_s2_5_does_see_backend_gpu.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.5 هل يرى المحرّك معالجك الرسومي؟
# المصدر  : src/chapters/ch02_setup_environment.md:150
# الترتيب : 11/13
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch02_setup_environment/012_s2_6_colab_kaggle_paths_quick.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.6 المسارات السريعة: Colab وKaggle
# المصدر  : src/chapters/ch02_setup_environment.md:198
# الترتيب : 12/13
# ──────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch02_setup_environment/013_s2_7_script_validation_ready.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.7 سكربت التحقّق الجاهز
# المصدر  : src/chapters/ch02_setup_environment.md:219
# الترتيب : 13/13
# ──────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : من داخل الكود
# المصدر  : src/chapters/ch02_setup_environment.md:127
# الترتيب : 9/13
# ──────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : التحقّق
# المصدر  : src/chapters/ch02_setup_environment.md:136
# الترتيب : 10/13
# ──────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.5 هل يرى المحرّك معالجك الرسومي؟
# المصدر  : src/chapters/ch02_setup_environment.md:150
# الترتيب : 11/13
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.6 المسارات السريعة: Colab وKaggle
# المصدر  : src/chapters/ch02_setup_environment.md:198
# الترتيب : 12/13
# ──────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.7 سكربت التحقّق الجاهز
# المصدر  : src/chapters/ch02_setup_environment.md:219
# الترتيب : 13/13
# ──────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — البيئة: من الصفر إلى أول استيراد ناجح
Keras 3 in Arabic — Setup Environment

Book slug : keras-3-arabic-professional
Chapter   : ch-02-ch02-setup-environment (lesson 2)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch02_setup_environment/009_inside.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : من داخل الكود
# المصدر  : src/chapters/ch02_setup_environment.md:127
# الترتيب : 9/13
# ──────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch02_setup_environment/010_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : التحقّق
# المصدر  : src/chapters/ch02_setup_environment.md:136
# الترتيب : 10/13
# ──────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch02_setup_environment/011_s2_5_does_see_backend_gpu.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.5 هل يرى المحرّك معالجك الرسومي؟
# المصدر  : src/chapters/ch02_setup_environment.md:150
# الترتيب : 11/13
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch02_setup_environment/012_s2_6_colab_kaggle_paths_quick.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.6 المسارات السريعة: Colab وKaggle
# المصدر  : src/chapters/ch02_setup_environment.md:198
# الترتيب : 12/13
# ──────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch02_setup_environment/013_s2_7_script_validation_ready.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.7 سكربت التحقّق الجاهز
# المصدر  : src/chapters/ch02_setup_environment.md:219
# الترتيب : 13/13
# ──────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch02_setup_environment.md
# ────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : من داخل الكود
# المصدر  : src/chapters/ch02_setup_environment.md:127
# الترتيب : 9/13
# ──────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : التحقّق
# المصدر  : src/chapters/ch02_setup_environment.md:136
# الترتيب : 10/13
# ──────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.5 هل يرى المحرّك معالجك الرسومي؟
# المصدر  : src/chapters/ch02_setup_environment.md:150
# الترتيب : 11/13
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.6 المسارات السريعة: Colab وKaggle
# المصدر  : src/chapters/ch02_setup_environment.md:198
# الترتيب : 12/13
# ──────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : 2.7 سكربت التحقّق الجاهز
# المصدر  : src/chapters/ch02_setup_environment.md:219
# الترتيب : 13/13
# ──────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
keras.config.set_backend("jax")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

print("Keras :", keras.__version__)
print("Backend:", keras.backend.backend())

# اختبار تمرير أمامي فعلي
x = np.random.rand(64, 784).astype("float32")
model = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
y = model(x, training=False)
print("Output shape:", y.shape)

# هل الأوزان على معالج؟
try:
    dev = next(iter(model.trainable_variables)).device
    print("Device:", dev)
except Exception as exc:
    print("تعذّر تحديد الجهاز:", exc)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# في خلية Colab أو Kaggle
# !pip install -q "keras[torch]==3.15.1"

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-02-ch02-setup-environment.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
"""فحص بيئة Keras — احفظه باسم smoke_test.py واستدعِه بعد أي ترقية."""
import sys
import numpy as np
import keras

def main() -> int:
    print("=" * 46)
    print("Keras        :", keras.__version__)
    print("Backend      :", keras.backend.backend())
    print("Python       :", sys.version.split()[0])
    print("=" * 46)

    # 1) بناء نموذج
    model = keras.Sequential([
        keras.layers.Input((32,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    # 2) تجميع
    model.compile(optimizer="adam", loss="mse")

    # 3) تدريب
    x = np.random.rand(128, 32).astype("float32")
    y = np.random.rand(128, 1).astype("float32")
    hist = model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    assert "loss" in hist.history, "لا يوجد سجل خسارة"

    # 4) تنبّؤ
    out = model.predict(x[:8], verbose=0)
    assert out.shape == (8, 1), f"شكل غير متوقّع: {out.shape}"

    # 5) حفظ وتحميل
    model.save("_smoke.keras")
    reloaded = keras.saving.load_model("_smoke.keras")
    assert reloaded.count_params() == model.count_params()

    print("المعاملات    :", model.count_params())
    print("الخسارة النهائية:", float(hist.history["loss"][-1]))
    print("\n✓ كل الفحوص ناجحة — البيئة سليمة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
