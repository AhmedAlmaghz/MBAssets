#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — بنية النموذج: ثلاث طرق لتحديد الشبكة
Keras 3 in Arabic — Model Architecture

Book slug : keras-3-arabic-professional
Chapter   : ch-04-ch04-model-architecture (lesson 4)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-04-ch04-model-architecture.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — بنية النموذج: ثلاث طرق لتحديد الشبكة
Keras 3 in Arabic — Model Architecture

Book slug : keras-3-arabic-professional
Chapter   : ch-04-ch04-model-architecture (lesson 4)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-04-ch04-model-architecture.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — بنية النموذج: ثلاث طرق لتحديد الشبكة
Keras 3 in Arabic — Model Architecture

Book slug : keras-3-arabic-professional
Chapter   : ch-04-ch04-model-architecture (lesson 4)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch04_model_architecture/001_s4_2_sequential_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.2 الواجهة الأولى: Sequential
# المصدر  : src/chapters/ch04_model_architecture.md:37
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch04_model_architecture/002_s4_3_functional_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.3 الواجهة الثانية: Functional — الأصل الحقيقي
# المصدر  : src/chapters/ch04_model_architecture.md:58
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch04_model_architecture/003_sequential_branches_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : الفرعين: حالة لا يستطيع Sequential التعبير عنها
# المصدر  : src/chapters/ch04_model_architecture.md:78
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch04_model_architecture/004_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فرع واحد بمخرجين
# المصدر  : src/chapters/ch04_model_architecture.md:99
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch04_model_architecture/005_trap_sharing_weights.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فخّ مشاركة الأوزان
# المصدر  : src/chapters/ch04_model_architecture.md:115
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch04_model_architecture/006_s4_4_subclassing_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.4 الواجهة الثالثة: Subclassing
# المصدر  : src/chapters/ch04_model_architecture.md:135
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch04_model_architecture/007_init_call_rule_golden.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : القاعدة الذهبية: أنشئ الطبقات في `__init__` لا في `call`
# المصدر  : src/chapters/ch04_model_architecture.md:154
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch04_model_architecture/008_build_compute_output_shape_structure_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : بنية النموذج: `build` و`compute_output_shape`
# المصدر  : src/chapters/ch04_model_architecture.md:170
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch04_model_architecture/009_s4_6_example_solved_model_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:221
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch04_model_architecture/010_s4_6_example_solved_model_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:246
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.2 الواجهة الأولى: Sequential
# المصدر  : src/chapters/ch04_model_architecture.md:37
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.3 الواجهة الثانية: Functional — الأصل الحقيقي
# المصدر  : src/chapters/ch04_model_architecture.md:58
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : الفرعين: حالة لا يستطيع Sequential التعبير عنها
# المصدر  : src/chapters/ch04_model_architecture.md:78
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فرع واحد بمخرجين
# المصدر  : src/chapters/ch04_model_architecture.md:99
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فخّ مشاركة الأوزان
# المصدر  : src/chapters/ch04_model_architecture.md:115
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.4 الواجهة الثالثة: Subclassing
# المصدر  : src/chapters/ch04_model_architecture.md:135
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : القاعدة الذهبية: أنشئ الطبقات في `__init__` لا في `call`
# المصدر  : src/chapters/ch04_model_architecture.md:154
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : بنية النموذج: `build` و`compute_output_shape`
# المصدر  : src/chapters/ch04_model_architecture.md:170
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:221
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:246
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — بنية النموذج: ثلاث طرق لتحديد الشبكة
Keras 3 in Arabic — Model Architecture

Book slug : keras-3-arabic-professional
Chapter   : ch-04-ch04-model-architecture (lesson 4)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch04_model_architecture/001_s4_2_sequential_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.2 الواجهة الأولى: Sequential
# المصدر  : src/chapters/ch04_model_architecture.md:37
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch04_model_architecture/002_s4_3_functional_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.3 الواجهة الثانية: Functional — الأصل الحقيقي
# المصدر  : src/chapters/ch04_model_architecture.md:58
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch04_model_architecture/003_sequential_branches_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : الفرعين: حالة لا يستطيع Sequential التعبير عنها
# المصدر  : src/chapters/ch04_model_architecture.md:78
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch04_model_architecture/004_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فرع واحد بمخرجين
# المصدر  : src/chapters/ch04_model_architecture.md:99
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch04_model_architecture/005_trap_sharing_weights.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فخّ مشاركة الأوزان
# المصدر  : src/chapters/ch04_model_architecture.md:115
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch04_model_architecture/006_s4_4_subclassing_api.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.4 الواجهة الثالثة: Subclassing
# المصدر  : src/chapters/ch04_model_architecture.md:135
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch04_model_architecture/007_init_call_rule_golden.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : القاعدة الذهبية: أنشئ الطبقات في `__init__` لا في `call`
# المصدر  : src/chapters/ch04_model_architecture.md:154
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch04_model_architecture/008_build_compute_output_shape_structure_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : بنية النموذج: `build` و`compute_output_shape`
# المصدر  : src/chapters/ch04_model_architecture.md:170
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch04_model_architecture/009_s4_6_example_solved_model_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:221
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch04_model_architecture/010_s4_6_example_solved_model_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:246
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch04_model_architecture.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.2 الواجهة الأولى: Sequential
# المصدر  : src/chapters/ch04_model_architecture.md:37
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.3 الواجهة الثانية: Functional — الأصل الحقيقي
# المصدر  : src/chapters/ch04_model_architecture.md:58
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : الفرعين: حالة لا يستطيع Sequential التعبير عنها
# المصدر  : src/chapters/ch04_model_architecture.md:78
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فرع واحد بمخرجين
# المصدر  : src/chapters/ch04_model_architecture.md:99
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فخّ مشاركة الأوزان
# المصدر  : src/chapters/ch04_model_architecture.md:115
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.4 الواجهة الثالثة: Subclassing
# المصدر  : src/chapters/ch04_model_architecture.md:135
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : القاعدة الذهبية: أنشئ الطبقات في `__init__` لا في `call`
# المصدر  : src/chapters/ch04_model_architecture.md:154
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : بنية النموذج: `build` و`compute_output_shape`
# المصدر  : src/chapters/ch04_model_architecture.md:170
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:221
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:246
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(10,))
layer = keras.layers.Dense(8, activation="relu")

a = layer(inputs)   # نداء أول
b = layer(inputs)   # نداء ثانٍ — نفس الطبقة!

print(layer.weights[0].shape)   # (10, 8) — مصفوفة واحدة فقط

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ✗ خطأ: إنشاء طبقة داخل call
def call(self, x):
    d = keras.layers.Dense(64)   # خطأ: طبقة جديدة في كل استدعاء!
    return d(x)

# ✓ صح: أنشئها مرة واحدة في __init__
def __init__(self):
    super().__init__()
    self.d = keras.layers.Dense(64)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class ScaledDense(keras.layers.Layer):
    def __init__(self, units, scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = scale

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "scale": self.scale})
        return config

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-04-ch04-model-architecture.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)
