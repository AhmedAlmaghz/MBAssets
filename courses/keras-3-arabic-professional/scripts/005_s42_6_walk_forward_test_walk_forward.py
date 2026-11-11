# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.6 الاختبار العكسيّ (Walk-Forward)
# المصدر  : src/chapters/ch42_project_timeseries.md:132
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
def walk_forward(series, window, model_fn, n_folds=4):
    """تقييم عبر عدّة نوافذ زمنيّة متتالية."""
    maes = []
    n = len(series)
    fold_size = (n - window) // (n_folds + 1)
    for i in range(n_folds):
        end_train = window + fold_size * (i + 1)
        train = series[:end_train]
        test = series[end_train:end_train + fold_size]
        if len(test) < window + 1:
            break
        x_tr, y_tr = make_windows(train, window)
        x_te, y_te = make_windows(test, window)
        m, s = x_tr.mean(), x_tr.std()
        x_tr_n = (x_tr - m) / s
        y_tr_n = (y_tr - m) / s
        x_te_n = (x_te - m) / s
        y_te_n = (y_te - m) / s
        m_ = model_fn()
        m_.fit(x_tr_n, y_tr_n, epochs=10, batch_size=32, verbose=0)
        mae = m_.evaluate(x_te_n, y_te_n, verbose=0)[1]
        maes.append(mae)
        print(f"fold {i+1}: MAE = {mae:.4f}")
    return np.mean(maes)

def build():
    m = keras.Sequential([keras.Input(shape=(WINDOW, 1)),
                          keras.layers.LSTM(32),
                          keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m

avg_mae = walk_forward(sales, WINDOW, build, n_folds=4)
# النتيجة المتوقعة: متوسّط MAE عبر أربع فترات.
