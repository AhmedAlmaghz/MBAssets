# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : كشف تنوّع المولّد (كشف انهيار الأنماط)
# المصدر  : src/chapters/ch32_gan.md:219
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.
