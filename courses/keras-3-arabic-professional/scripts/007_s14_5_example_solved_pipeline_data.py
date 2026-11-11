# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:266
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية
