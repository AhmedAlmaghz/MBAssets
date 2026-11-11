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
