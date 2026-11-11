# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : حساب عدد الرقع وكلفة الانتباه
# المصدر  : src/chapters/ch23_vision_transformer.md:153
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.
