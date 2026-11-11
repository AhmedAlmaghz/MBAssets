# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.4 التوليد من ضجيج خالص
# المصدر  : src/chapters/ch33_diffusion.md:121
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)
