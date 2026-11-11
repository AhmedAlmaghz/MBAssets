# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.5 نموذج LSTM
# المصدر  : src/chapters/ch42_project_timeseries.md:107
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج LSTM للانحدار
inp = keras.Input(shape=(WINDOW, 1))
x = keras.layers.LSTM(32)(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1)(x)          # بلا تنشيط: انحدار
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])

# ─── 2. التدريب
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=32, verbose=0)

# ─── 3. التقييم على الاختبار
model_mae = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"model MAE = {model_mae:.4f}, naive MAE = {naive_mae:.4f}")
# النتيجة المتوقعة: model MAE أقلّ من naive MAE (شرط التقبّل).
