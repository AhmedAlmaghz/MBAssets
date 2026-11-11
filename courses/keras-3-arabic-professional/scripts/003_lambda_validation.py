# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : تقاطع التحقّق: اختر $\lambda$ بالأدلة
# المصدر  : src/chapters/ch13_regularization.md:181
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)
