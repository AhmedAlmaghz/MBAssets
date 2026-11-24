#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار تحليل البيانات (pandas, numpy, matplotlib)
Python 3.14 — From Zero to Pro — pandas

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 75 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-20-chapter-20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار تحليل البيانات (pandas, numpy, matplotlib)
Python 3.14 — From Zero to Pro — pandas

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 74 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["pandas", "numpy", "matplotlib", "openpyxl", "arabic_reshaper", "bidi"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q pandas numpy matplotlib openpyxl arabic-reshaper python-bidi

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# مجلدات الإخراج التي تكتب فيها الأمثلة
pathlib.Path('data').mkdir(exist_ok=True)

# إعداد خاص بهذا الفصل
import pandas as _pd
_sales = _pd.DataFrame({
    'المنتج': ['لابتوب', 'ماوس', 'شاشة', 'لابتوب', 'ماوس'],
    'المدينة': ['الرياض', 'جدة', 'الرياض', 'جدة', 'الرياض'],
    'السعر': [4500, 120, 900, 4700, 110],
    'الكمية': [2, 10, 3, 1, 8],
})
if not pathlib.Path('sales.csv').exists():
    _sales.to_csv('sales.csv', index=False, encoding='utf-8')
    _sales.to_json('sales.json', orient='records', force_ascii=False)
    _sales.to_excel('sales.xlsx', index=False, sheet_name='2024')
    print('أُنشئت ملفات المبيعات النموذجية: sales.csv / sales.json / sales.xlsx')

# --- عرض العربية في رسوم Matplotlib بشكل متصل وصحيح الاتجاه ---
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _shape = lambda s: get_display(arabic_reshaper.reshape(s))
    import matplotlib.text as _mt
    _orig = _mt.Text.set_text
    def _set_text(self, s):
        if isinstance(s, str) and any('\u0600' <= c <= '\u06FF' for c in s):
            s = _shape(s)
        return _orig(self, s)
    _mt.Text.set_text = _set_text
    print("دعم العربية في الرسوم: مُفعّل")
except ImportError:
    print("تنبيه: ثبّت arabic-reshaper وpython-bidi لعرض العربية في الرسوم")

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# محاولة جمع قائمتين من الأرقام عنصرًا بعنصر
prices = [10, 20, 30]
discounts = [1, 2, 3]

# هذا لن يعمل كما تتوقع!
result = prices + discounts
print(result)  # [10, 20, 30, 1, 2, 3] -- دمج وليس جمع!

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
result = [p - d for p, d in zip(prices, discounts)]
print(result)  # [9, 18, 27]

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

prices = np.array([10, 20, 30])
discounts = np.array([1, 2, 3])

result = prices - discounts
print(result)  # [9 18 27]
print(type(result))  # <class 'numpy.ndarray'>

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# من قائمة
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.dtype)    # int64 (نوع البيانات الموحّد)

# مصفوفة ثنائية الأبعاد (مصفوفة من مصفوفات = جدول)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(matrix.shape)  # (2, 3) -- صفّان، 3 أعمدة

# دوال إنشاء جاهزة
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))        # مصفوفة 2x3 من الآحاد
range_arr = np.arange(0, 10, 2)   # [0 2 4 6 8]  (مثل range لكن تُرجع array)
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ] -- 5 قيم متباعدة بالتساوي
random_arr = np.random.rand(3)    # 3 أرقام عشوائية بين 0 و1

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
prices = np.array([100, 200, 300, 400])

# عملية على كل عنصر دفعة واحدة
with_tax = prices * 1.15
print(with_tax)  # [115. 230. 345. 460.]

# مقارنة تُرجع مصفوفة من Boolean
expensive = prices > 250
print(expensive)  # [False False  True  True]

# استخدام هذه المصفوفة Boolean للتصفية! (Boolean Indexing)
print(prices[expensive])  # [300 400]

# أو في سطر واحد
print(prices[prices > 250])  # [300 400]

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-20.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
sales = np.array([120, 340, 210, 450, 90, 380])

print(sales.sum())     # 1590 -- المجموع
print(sales.mean())    # 265.0 -- المتوسط
print(sales.max())     # 450 -- الأكبر
print(sales.min())     # 90 -- الأصغر
print(sales.std())     # الانحراف المعياري
print(sales.argmax())  # 3 -- فهرس العنصر الأكبر
print(np.median(sales))  # الوسيط

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-20.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

temperatures = pd.Series([22, 25, 19, 30], index=["الأحد", "الاثنين", "الثلاثاء", "الأربعاء"])
print(temperatures)

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-20.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
data = {
    "اسم المنتج": ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة"],
    "السعر": [3500, 50, 120, 900],
    "الكمية": [10, 50, 30, 15],
}

df = pd.DataFrame(data)
print(df)

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-20.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# من CSV
df = pd.read_csv("sales.csv")

# من Excel (يتطلب: uv add openpyxl)
df = pd.read_excel("sales.xlsx", sheet_name="2024")

# من JSON
df = pd.read_json("sales.json")

# تحديد الترميز إن كانت البيانات عربية
df = pd.read_csv("sales.csv", encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-20.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
df.head()       # أول 5 صفوف (الافتراضي)
df.head(10)     # أول 10 صفوف
df.tail(3)      # آخر 3 صفوف

df.info()       # معلومات عامة: الأعمدة، الأنواع، القيم غير الفارغة
df.describe()   # إحصائيات (متوسط، انحراف معياري، أدنى، أقصى...) للأعمدة الرقمية

df.shape        # (عدد الصفوف, عدد الأعمدة) -- خاصية وليست دالة!
df.columns      # أسماء الأعمدة
df.dtypes       # نوع بيانات كل عمود
df.index        # معلومات الفهرس

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-20.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس"],
    "السعر": [3500, 50, 900, 3600, 55],
    "الكمية": [2, 10, 3, 1, 8],
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام", "جدة"],
})

print(df.info())

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-20.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# اختيار عمود واحد -> يُرجع Series
print(df["المنتج"])

# اختيار عدة أعمدة -> يُرجع DataFrame (لاحظ قائمة داخل قوسين)
print(df[["المنتج", "السعر"]])

# loc: الاختيار بالاسم/التسمية
print(df.loc[0])              # الصف الأول (حسب التسمية، وهنا تتطابق مع الرقم)
print(df.loc[0, "المنتج"])    # قيمة محددة
print(df.loc[0:2, "السعر"])   # نطاق من الصفوف، عمود محدد

# iloc: الاختيار بالموضع الرقمي (مثل القوائم تمامًا)
print(df.iloc[0])        # أول صف بغض النظر عن تسميته
print(df.iloc[0, 1])     # الصف 0، العمود 1
print(df.iloc[-1])       # آخر صف

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-20.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# كل المنتجات التي سعرها أكبر من 500
expensive = df[df["السعر"] > 500]
print(expensive)

# شروط متعددة: استخدم & و | (وليس and/or) مع أقواس إلزامية
cheap_riyadh = df[(df["السعر"] < 1000) & (df["المدينة"] == "الرياض")]
print(cheap_riyadh)

# التحقق من انتماء القيمة لقائمة
target_cities = df[df["المدينة"].isin(["جدة", "الدمام"])]

# استبعاد قيم
not_riyadh = df[df["المدينة"] != "الرياض"]

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-20.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# إضافة عمود جديد محسوب
df["الإجمالي"] = df["السعر"] * df["الكمية"]
print(df)

# إضافة عمود بشرط باستخدام numpy.where
df["الفئة"] = np.where(df["السعر"] > 500, "غالي", "رخيص")

# تعديل عمود موجود
df["السعر"] = df["السعر"] * 1.15  # إضافة ضريبة 15%

# حذف عمود
df = df.drop(columns=["الفئة"])

# إعادة تسمية الأعمدة
df = df.rename(columns={"السعر": "سعر_الوحدة"})

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-20.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
data = {
    "اسم": ["أحمد", "سارة", None, "خالد"],
    "العمر": [25, None, 30, 28],
}
df = pd.DataFrame(data)

print(df.isna())          # جدول Boolean يوضح أماكن القيم المفقودة
print(df.isna().sum())    # عدد القيم المفقودة في كل عمود

# حذف الصفوف التي تحتوي أي قيمة مفقودة
clean_df = df.dropna()

# حذف فقط إذا كان عمود معين فارغًا
clean_df = df.dropna(subset=["اسم"])

# تعويض القيم المفقودة بقيمة محددة
df["العمر"] = df["العمر"].fillna(df["العمر"].mean())  # بالمتوسط
df["اسم"] = df["اسم"].fillna("غير معروف")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-20.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# 🔄 نعيد تعريف df ببيانات المبيعات حتى يعمل المثال كما في الكتاب
df = pd.DataFrame({
    'المنتج': ['لابتوب', 'ماوس', 'شاشة', 'لابتوب', 'ماوس'],
    'المدينة': ['الرياض', 'جدة', 'الرياض', 'جدة', 'الرياض'],
    'السعر': [4500, 120, 900, 4700, 110],
    'الكمية': [2, 10, 3, 1, 8],
})

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-20.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
df_sorted = df.sort_values("السعر")                    # تصاعدي
df_sorted = df.sort_values("السعر", ascending=False)    # تنازلي
df_sorted = df.sort_values(["المدينة", "السعر"])         # فرز متعدد المستويات

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-20.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس", "شاشة"],
    "المدينة": ["الرياض", "جدة", "الرياض", "جدة", "الرياض", "جدة"],
    "الإجمالي": [7000, 500, 2700, 3600, 450, 1800],
})

# مجموع المبيعات لكل مدينة
by_city = df.groupby("المدينة")["الإجمالي"].sum()
print(by_city)

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-20.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# عدة إحصائيات دفعة واحدة
summary = df.groupby("المدينة")["الإجمالي"].agg(["sum", "mean", "count", "max"])
print(summary)

# التجميع حسب أكثر من عمود
by_city_product = df.groupby(["المدينة", "المنتج"])["الإجمالي"].sum()
print(by_city_product)

# تحويل النتيجة إلى DataFrame عادي مرة أخرى
result = df.groupby("المدينة")["الإجمالي"].sum().reset_index()

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-20.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
pivot = df.pivot_table(
    values="الإجمالي",
    index="المدينة",
    columns="المنتج",
    aggfunc="sum",
    fill_value=0,
)
print(pivot)

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-20.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "اسم": ["أحمد", "سارة", "خالد"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "المبلغ": [200, 450, 120, 300],
})

# الدمج (مثل JOIN في SQL)
merged = pd.merge(orders, customers, on="customer_id", how="left")
print(merged)

# دمج/تكديس DataFrames فوق بعضها (مثل إلحاق ملفات شهر بشهر)
jan_sales = pd.DataFrame({"المنتج": ["لابتوب"], "الإجمالي": [3500]})
feb_sales = pd.DataFrame({"المنتج": ["ماوس"], "الإجمالي": [50]})
all_sales = pd.concat([jan_sales, feb_sales], ignore_index=True)

# ────────────────────────────────────────────────────────────────────────
# [023] dist/notebooks/chapter-20.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "التاريخ": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "المبيعات": [1000, 1500, 1200],
})

# تحويل النص إلى تاريخ حقيقي
df["التاريخ"] = pd.to_datetime(df["التاريخ"])

# استخراج معلومات من التاريخ
df["الشهر"] = df["التاريخ"].dt.month
df["اسم_اليوم"] = df["التاريخ"].dt.day_name()

print(df)

# ────────────────────────────────────────────────────────────────────────
# [024] dist/notebooks/chapter-20.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
df.to_excel("output.xlsx", index=False)
df.to_json("output.json", orient="records", force_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [025] dist/notebooks/chapter-20.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

months = ["يناير", "فبراير", "مارس", "أبريل", "مايو"]
sales = [1200, 1900, 1500, 2200, 2800]

plt.plot(months, sales)
plt.title("المبيعات الشهرية")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.show()  # يفتح نافذة لعرض الرسم

# ────────────────────────────────────────────────────────────────────────
# [026] dist/notebooks/chapter-20.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

categories = ["لابتوب", "ماوس", "شاشة", "لوحة مفاتيح"]
values = [45, 120, 30, 80]

# رسم بياني خطي -- مناسب للاتجاهات عبر الزمن
plt.plot(categories, values, marker="o")
plt.show()

# رسم بياني شريطي (Bar Chart) -- مناسب للمقارنة بين فئات
plt.bar(categories, values, color="steelblue")
plt.show()

# رسم دائري (Pie Chart) -- مناسب لإظهار النسب من الكل
plt.pie(values, labels=categories, autopct="%1.1f%%")
plt.show()

# مخطط التشتت (Scatter Plot) -- مناسب لإظهار العلاقة بين متغيرين
prices = [3500, 50, 900, 120]
plt.scatter(prices, values)
plt.xlabel("السعر")
plt.ylabel("الكمية المباعة")
plt.show()

# المدرج التكراري (Histogram) -- مناسب لتوزيع البيانات
import numpy as np
ages = np.random.normal(30, 5, 1000)  # 1000 عمر عشوائي حول متوسط 30
plt.hist(ages, bins=20, color="coral", edgecolor="black")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [027] dist/notebooks/chapter-20.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))  # حجم الرسم (عرض, ارتفاع) بالبوصة

plt.plot(months, sales, marker="o", linestyle="--", color="green", linewidth=2, label="2024")
plt.title("المبيعات الشهرية", fontsize=16, fontweight="bold")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.legend()        # إظهار الـ label الموجود أعلاه
plt.grid(True, alpha=0.3)   # شبكة خفيفة في الخلفية
plt.tight_layout()  # ترتيب تلقائي لمنع تداخل العناصر

plt.savefig("sales_chart.png", dpi=300)  # حفظ كصورة (بدلاً من/بالإضافة إلى show)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [028] dist/notebooks/chapter-20.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # صف واحد، عمودان

axes[0].plot(months, sales, marker="o")
axes[0].set_title("خطي")

axes[1].bar(months, sales, color="orange")
axes[1].set_title("شريطي")

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [029] dist/notebooks/chapter-20.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "الشهر": ["يناير", "فبراير", "مارس"],
    "المبيعات": [1200, 1900, 1500],
})

df.plot(x="الشهر", y="المبيعات", kind="bar", title="المبيعات الشهرية")
plt.show()

# أو مباشرة من نتيجة groupby
df.groupby("الشهر")["المبيعات"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [030] dist/notebooks/chapter-20.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
"""سكربت مساعد لتوليد بيانات مبيعات تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)  # لضمان نتائج قابلة لإعادة الإنتاج

n_records = 1000

products = ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة", "سماعات", "طابعة"]
product_prices = {
    "لابتوب": 3500, "ماوس": 50, "لوحة مفاتيح": 120,
    "شاشة": 900, "سماعات": 150, "طابعة": 600,
}
cities = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]

data = {
    "order_id": range(1, n_records + 1),
    "التاريخ": pd.date_range("2024-01-01", "2024-12-31", periods=n_records),
    "المنتج": np.random.choice(products, n_records),
    "المدينة": np.random.choice(cities, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "الكمية": np.random.randint(1, 6, n_records),
}

df = pd.DataFrame(data)
df["سعر_الوحدة"] = df["المنتج"].map(product_prices)
df["الإجمالي"] = df["سعر_الوحدة"] * df["الكمية"]

# إدخال بعض القيم المفقودة عمدًا (كما يحدث في البيانات الحقيقية)
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, "المدينة"] = None

df.to_csv("data/sales_data.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n_records} سجل مبيعات في data/sales_data.csv")

# ────────────────────────────────────────────────────────────────────────
# [031] dist/notebooks/chapter-20.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
students = pd.DataFrame({
    "الاسم": ["أحمد", "سارة", "خالد", "منى", "يوسف"],
    "الرياضيات": [85, 92, 78, 95, 88],
    "العلوم": [90, 85, 82, 91, 79],
})
print(students["الرياضيات"].mean())
print(students["العلوم"].mean())

# 2
numbers = np.random.randint(1, 101, 10)
print(numbers)
print("الأكبر:", numbers.max())
print("الأصغر:", numbers.min())
print("المتوسط:", numbers.mean())

# 3
days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء"]
hours = [2, 3, 1.5, 4, 2.5]
plt.bar(days, hours, color="purple")
plt.title("ساعات الدراسة")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [032] dist/notebooks/chapter-20.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

# 4
employees = pd.DataFrame({
    "اسم": ["علي", "فاطمة", "سعد", "ليلى", "عمر"],
    "القسم": ["مبيعات", "تقنية", "مبيعات", "تقنية", "مالية"],
    "الراتب": [8000, 12000, 8500, 13000, 9500],
})
avg_salary = employees.groupby("القسم")["الراتب"].mean()
print(avg_salary)
avg_salary.plot(kind="bar", color="teal")
plt.title("متوسط الراتب حسب القسم")
plt.ylabel("الراتب")
plt.tight_layout()
plt.show()

# 5
data = pd.DataFrame({"العمر": [25, 30, None, 45, 28, None, 90]})
# الوسيط أفضل من المتوسط هنا لأن القيمة 90 (قيمة متطرفة/Outlier)
# تؤثر بشدة على المتوسط لكن تأثيرها على الوسيط محدود جدًا
data["العمر"] = data["العمر"].fillna(data["العمر"].median())
print(data)

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch20/20-001-prices.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-001-prices.py
"""

# محاولة جمع قائمتين من الأرقام عنصرًا بعنصر
prices = [10, 20, 30]
discounts = [1, 2, 3]

# هذا لن يعمل كما تتوقع!
result = prices + discounts
print(result)  # [10, 20, 30, 1, 2, 3] -- دمج وليس جمع!

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch20/20-002-result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-002-result.py
"""

result = [p - d for p, d in zip(prices, discounts)]
print(result)  # [9, 18, 27]

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch20/20-003-prices.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-003-prices.py
"""

import numpy as np

prices = np.array([10, 20, 30])
discounts = np.array([1, 2, 3])

result = prices - discounts
print(result)  # [9 18 27]
print(type(result))  # <class 'numpy.ndarray'>

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch20/20-004-ex20_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-004-ex20_3.py
"""

import numpy as np

# من قائمة
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.dtype)    # int64 (نوع البيانات الموحّد)

# مصفوفة ثنائية الأبعاد (مصفوفة من مصفوفات = جدول)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(matrix.shape)  # (2, 3) -- صفّان، 3 أعمدة

# دوال إنشاء جاهزة
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))        # مصفوفة 2x3 من الآحاد
range_arr = np.arange(0, 10, 2)   # [0 2 4 6 8]  (مثل range لكن تُرجع array)
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ] -- 5 قيم متباعدة بالتساوي
random_arr = np.random.rand(3)    # 3 أرقام عشوائية بين 0 و1

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch20/20-005-prices.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-005-prices.py
"""

prices = np.array([100, 200, 300, 400])

# عملية على كل عنصر دفعة واحدة
with_tax = prices * 1.15
print(with_tax)  # [115. 230. 345. 460.]

# مقارنة تُرجع مصفوفة من Boolean
expensive = prices > 250
print(expensive)  # [False False  True  True]

# استخدام هذه المصفوفة Boolean للتصفية! (Boolean Indexing)
print(prices[expensive])  # [300 400]

# أو في سطر واحد
print(prices[prices > 250])  # [300 400]

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch20/20-006-sales.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-006-sales.py
"""

sales = np.array([120, 340, 210, 450, 90, 380])

print(sales.sum())     # 1590 -- المجموع
print(sales.mean())    # 265.0 -- المتوسط
print(sales.max())     # 450 -- الأكبر
print(sales.min())     # 90 -- الأصغر
print(sales.std())     # الانحراف المعياري
print(sales.argmax())  # 3 -- فهرس العنصر الأكبر
print(np.median(sales))  # الوسيط

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch20/20-007-temperatures.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-007-temperatures.py
"""

import pandas as pd

temperatures = pd.Series([22, 25, 19, 30], index=["الأحد", "الاثنين", "الثلاثاء", "الأربعاء"])
print(temperatures)

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch20/20-008-data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-008-data.py
"""

data = {
    "اسم المنتج": ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة"],
    "السعر": [3500, 50, 120, 900],
    "الكمية": [10, 50, 30, 15],
}

df = pd.DataFrame(data)
print(df)

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch20/20-009-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-009-ex20_4.py
"""

# من CSV
df = pd.read_csv("sales.csv")

# من Excel (يتطلب: uv add openpyxl)
df = pd.read_excel("sales.xlsx", sheet_name="2024")

# من JSON
df = pd.read_json("sales.json")

# تحديد الترميز إن كانت البيانات عربية
df = pd.read_csv("sales.csv", encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch20/20-010-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-010-ex20_4.py
"""

df.head()       # أول 5 صفوف (الافتراضي)
df.head(10)     # أول 10 صفوف
df.tail(3)      # آخر 3 صفوف

df.info()       # معلومات عامة: الأعمدة، الأنواع، القيم غير الفارغة
df.describe()   # إحصائيات (متوسط، انحراف معياري، أدنى، أقصى...) للأعمدة الرقمية

df.shape        # (عدد الصفوف, عدد الأعمدة) -- خاصية وليست دالة!
df.columns      # أسماء الأعمدة
df.dtypes       # نوع بيانات كل عمود
df.index        # معلومات الفهرس

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch20/20-011-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-011-ex20_4.py
"""

import pandas as pd

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس"],
    "السعر": [3500, 50, 900, 3600, 55],
    "الكمية": [2, 10, 3, 1, 8],
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام", "جدة"],
})

print(df.info())

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch20/20-012-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-012-ex20_4.py
"""

# اختيار عمود واحد -> يُرجع Series
print(df["المنتج"])

# اختيار عدة أعمدة -> يُرجع DataFrame (لاحظ قائمة داخل قوسين)
print(df[["المنتج", "السعر"]])

# loc: الاختيار بالاسم/التسمية
print(df.loc[0])              # الصف الأول (حسب التسمية، وهنا تتطابق مع الرقم)
print(df.loc[0, "المنتج"])    # قيمة محددة
print(df.loc[0:2, "السعر"])   # نطاق من الصفوف، عمود محدد

# iloc: الاختيار بالموضع الرقمي (مثل القوائم تمامًا)
print(df.iloc[0])        # أول صف بغض النظر عن تسميته
print(df.iloc[0, 1])     # الصف 0، العمود 1
print(df.iloc[-1])       # آخر صف

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch20/20-013-expensive.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-013-expensive.py
"""

# كل المنتجات التي سعرها أكبر من 500
expensive = df[df["السعر"] > 500]
print(expensive)

# شروط متعددة: استخدم & و | (وليس and/or) مع أقواس إلزامية
cheap_riyadh = df[(df["السعر"] < 1000) & (df["المدينة"] == "الرياض")]
print(cheap_riyadh)

# التحقق من انتماء القيمة لقائمة
target_cities = df[df["المدينة"].isin(["جدة", "الدمام"])]

# استبعاد قيم
not_riyadh = df[df["المدينة"] != "الرياض"]

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch20/20-014-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-014-ex20_4.py
"""

# إضافة عمود جديد محسوب
df["الإجمالي"] = df["السعر"] * df["الكمية"]
print(df)

# إضافة عمود بشرط باستخدام numpy.where
df["الفئة"] = np.where(df["السعر"] > 500, "غالي", "رخيص")

# تعديل عمود موجود
df["السعر"] = df["السعر"] * 1.15  # إضافة ضريبة 15%

# حذف عمود
df = df.drop(columns=["الفئة"])

# إعادة تسمية الأعمدة
df = df.rename(columns={"السعر": "سعر_الوحدة"})

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch20/20-015-data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-015-data.py
"""

data = {
    "اسم": ["أحمد", "سارة", None, "خالد"],
    "العمر": [25, None, 30, 28],
}
df = pd.DataFrame(data)

print(df.isna())          # جدول Boolean يوضح أماكن القيم المفقودة
print(df.isna().sum())    # عدد القيم المفقودة في كل عمود

# حذف الصفوف التي تحتوي أي قيمة مفقودة
clean_df = df.dropna()

# حذف فقط إذا كان عمود معين فارغًا
clean_df = df.dropna(subset=["اسم"])

# تعويض القيم المفقودة بقيمة محددة
df["العمر"] = df["العمر"].fillna(df["العمر"].mean())  # بالمتوسط
df["اسم"] = df["اسم"].fillna("غير معروف")

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch20/20-016-df_sorted.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-016-df_sorted.py
"""

df_sorted = df.sort_values("السعر")                    # تصاعدي
df_sorted = df.sort_values("السعر", ascending=False)    # تنازلي
df_sorted = df.sort_values(["المدينة", "السعر"])         # فرز متعدد المستويات

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch20/20-017-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-017-ex20_4.py
"""

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس", "شاشة"],
    "المدينة": ["الرياض", "جدة", "الرياض", "جدة", "الرياض", "جدة"],
    "الإجمالي": [7000, 500, 2700, 3600, 450, 1800],
})

# مجموع المبيعات لكل مدينة
by_city = df.groupby("المدينة")["الإجمالي"].sum()
print(by_city)

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch20/20-018-summary.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-018-summary.py
"""

# عدة إحصائيات دفعة واحدة
summary = df.groupby("المدينة")["الإجمالي"].agg(["sum", "mean", "count", "max"])
print(summary)

# التجميع حسب أكثر من عمود
by_city_product = df.groupby(["المدينة", "المنتج"])["الإجمالي"].sum()
print(by_city_product)

# تحويل النتيجة إلى DataFrame عادي مرة أخرى
result = df.groupby("المدينة")["الإجمالي"].sum().reset_index()

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch20/20-019-pivot.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-019-pivot.py
"""

pivot = df.pivot_table(
    values="الإجمالي",
    index="المدينة",
    columns="المنتج",
    aggfunc="sum",
    fill_value=0,
)
print(pivot)

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch20/20-020-customers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-020-customers.py
"""

customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "اسم": ["أحمد", "سارة", "خالد"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "المبلغ": [200, 450, 120, 300],
})

# الدمج (مثل JOIN في SQL)
merged = pd.merge(orders, customers, on="customer_id", how="left")
print(merged)

# دمج/تكديس DataFrames فوق بعضها (مثل إلحاق ملفات شهر بشهر)
jan_sales = pd.DataFrame({"المنتج": ["لابتوب"], "الإجمالي": [3500]})
feb_sales = pd.DataFrame({"المنتج": ["ماوس"], "الإجمالي": [50]})
all_sales = pd.concat([jan_sales, feb_sales], ignore_index=True)

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch20/20-021-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-021-ex20_4.py
"""

df = pd.DataFrame({
    "التاريخ": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "المبيعات": [1000, 1500, 1200],
})

# تحويل النص إلى تاريخ حقيقي
df["التاريخ"] = pd.to_datetime(df["التاريخ"])

# استخراج معلومات من التاريخ
df["الشهر"] = df["التاريخ"].dt.month
df["اسم_اليوم"] = df["التاريخ"].dt.day_name()

print(df)

# ────────────────────────────────────────────────────────────────────────
# [054] code/chapters/ch20/20-022-ex20_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-022-ex20_4.py
"""

df.to_csv("output.csv", index=False, encoding="utf-8-sig")
df.to_excel("output.xlsx", index=False)
df.to_json("output.json", orient="records", force_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [055] code/chapters/ch20/20-023-months.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-023-months.py
"""

import matplotlib.pyplot as plt

months = ["يناير", "فبراير", "مارس", "أبريل", "مايو"]
sales = [1200, 1900, 1500, 2200, 2800]

plt.plot(months, sales)
plt.title("المبيعات الشهرية")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.show()  # يفتح نافذة لعرض الرسم

# ────────────────────────────────────────────────────────────────────────
# [056] code/chapters/ch20/20-024-categories.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-024-categories.py
"""

import matplotlib.pyplot as plt

categories = ["لابتوب", "ماوس", "شاشة", "لوحة مفاتيح"]
values = [45, 120, 30, 80]

# رسم بياني خطي -- مناسب للاتجاهات عبر الزمن
plt.plot(categories, values, marker="o")
plt.show()

# رسم بياني شريطي (Bar Chart) -- مناسب للمقارنة بين فئات
plt.bar(categories, values, color="steelblue")
plt.show()

# رسم دائري (Pie Chart) -- مناسب لإظهار النسب من الكل
plt.pie(values, labels=categories, autopct="%1.1f%%")
plt.show()

# مخطط التشتت (Scatter Plot) -- مناسب لإظهار العلاقة بين متغيرين
prices = [3500, 50, 900, 120]
plt.scatter(prices, values)
plt.xlabel("السعر")
plt.ylabel("الكمية المباعة")
plt.show()

# المدرج التكراري (Histogram) -- مناسب لتوزيع البيانات
import numpy as np
ages = np.random.normal(30, 5, 1000)  # 1000 عمر عشوائي حول متوسط 30
plt.hist(ages, bins=20, color="coral", edgecolor="black")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [057] code/chapters/ch20/20-025-ex20_5.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-025-ex20_5.py
"""

plt.figure(figsize=(10, 6))  # حجم الرسم (عرض, ارتفاع) بالبوصة

plt.plot(months, sales, marker="o", linestyle="--", color="green", linewidth=2, label="2024")
plt.title("المبيعات الشهرية", fontsize=16, fontweight="bold")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.legend()        # إظهار الـ label الموجود أعلاه
plt.grid(True, alpha=0.3)   # شبكة خفيفة في الخلفية
plt.tight_layout()  # ترتيب تلقائي لمنع تداخل العناصر

plt.savefig("sales_chart.png", dpi=300)  # حفظ كصورة (بدلاً من/بالإضافة إلى show)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [058] code/chapters/ch20/20-026-ex20_5.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-026-ex20_5.py
"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # صف واحد، عمودان

axes[0].plot(months, sales, marker="o")
axes[0].set_title("خطي")

axes[1].bar(months, sales, color="orange")
axes[1].set_title("شريطي")

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [059] code/chapters/ch20/20-027-ex20_5.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-027-ex20_5.py
"""

import pandas as pd

df = pd.DataFrame({
    "الشهر": ["يناير", "فبراير", "مارس"],
    "المبيعات": [1200, 1900, 1500],
})

df.plot(x="الشهر", y="المبيعات", kind="bar", title="المبيعات الشهرية")
plt.show()

# أو مباشرة من نتيجة groupby
df.groupby("الشهر")["المبيعات"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [060] code/chapters/ch20/20-028-n_records.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-028-n_records.py
"""

"""سكربت مساعد لتوليد بيانات مبيعات تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)  # لضمان نتائج قابلة لإعادة الإنتاج

n_records = 1000

products = ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة", "سماعات", "طابعة"]
product_prices = {
    "لابتوب": 3500, "ماوس": 50, "لوحة مفاتيح": 120,
    "شاشة": 900, "سماعات": 150, "طابعة": 600,
}
cities = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]

data = {
    "order_id": range(1, n_records + 1),
    "التاريخ": pd.date_range("2024-01-01", "2024-12-31", periods=n_records),
    "المنتج": np.random.choice(products, n_records),
    "المدينة": np.random.choice(cities, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "الكمية": np.random.randint(1, 6, n_records),
}

df = pd.DataFrame(data)
df["سعر_الوحدة"] = df["المنتج"].map(product_prices)
df["الإجمالي"] = df["سعر_الوحدة"] * df["الكمية"]

# إدخال بعض القيم المفقودة عمدًا (كما يحدث في البيانات الحقيقية)
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, "المدينة"] = None

df.to_csv("data/sales_data.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n_records} سجل مبيعات في data/sales_data.csv")

# ────────────────────────────────────────────────────────────────────────
# [061] code/chapters/ch20/20-029-load_sales_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-029-load_sales_data.py
"""

"""وحدة تحميل وتنظيف بيانات المبيعات."""

from pathlib import Path

import pandas as pd

def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """تحميل بيانات المبيعات من ملف CSV.

    Args:
        file_path: مسار ملف CSV.

    Returns:
        DataFrame يحتوي بيانات المبيعات الخام.
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig", parse_dates=["التاريخ"])
    return df

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف بيانات المبيعات: معالجة القيم المفقودة والتكرارات.

    Args:
        df: DataFrame الخام.

    Returns:
        DataFrame نظيف جاهز للتحليل.
    """
    clean_df = df.copy()

    # تعويض المدن المفقودة بقيمة واضحة بدلاً من حذف الصفوف (نحتفظ بالبيانات)
    clean_df["المدينة"] = clean_df["المدينة"].fillna("غير محدد")

    # حذف أي صفوف مكررة بالكامل إن وُجدت
    clean_df = clean_df.drop_duplicates()

    # إضافة أعمدة مشتقة مفيدة للتحليل
    clean_df["الشهر"] = clean_df["التاريخ"].dt.month
    clean_df["اسم_الشهر"] = clean_df["التاريخ"].dt.month_name()
    clean_df["ربع_السنة"] = clean_df["التاريخ"].dt.quarter

    return clean_df

# ────────────────────────────────────────────────────────────────────────
# [062] code/chapters/ch20/20-030-total_revenue.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-030-total_revenue.py
"""

"""وحدة تحليل بيانات المبيعات."""

import pandas as pd

def total_revenue(df: pd.DataFrame) -> float:
    """حساب إجمالي الإيرادات."""
    return float(df["الإجمالي"].sum())

def revenue_by_product(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل منتج، مرتبة تنازليًا."""
    return df.groupby("المنتج")["الإجمالي"].sum().sort_values(ascending=False)

def revenue_by_city(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل مدينة، مرتبة تنازليًا."""
    return df.groupby("المدينة")["الإجمالي"].sum().sort_values(ascending=False)

def monthly_trend(df: pd.DataFrame) -> pd.Series:
    """اتجاه الإيرادات الشهري عبر العام."""
    return df.groupby("الشهر")["الإجمالي"].sum().sort_index()

def top_n_products_by_quantity(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """أكثر n منتجات مبيعًا من حيث الكمية."""
    return df.groupby("المنتج")["الكمية"].sum().sort_values(ascending=False).head(n)

def average_order_value(df: pd.DataFrame) -> float:
    """متوسط قيمة الطلب الواحد."""
    return float(df["الإجمالي"].mean())

def city_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """جدول محوري: الإيرادات لكل منتج في كل مدينة."""
    return df.pivot_table(
        values="الإجمالي",
        index="المدينة",
        columns="المنتج",
        aggfunc="sum",
        fill_value=0,
    )

def generate_summary_report(df: pd.DataFrame) -> dict[str, float | int]:
    """توليد ملخص إحصائي شامل للبيانات."""
    return {
        "إجمالي_الإيرادات": total_revenue(df),
        "عدد_الطلبات": len(df),
        "متوسط_قيمة_الطلب": average_order_value(df),
        "أعلى_منتج_مبيعًا": str(revenue_by_product(df).index[0]),
        "أعلى_مدينة_مبيعًا": str(revenue_by_city(df).index[0]),
    }

# ────────────────────────────────────────────────────────────────────────
# [063] code/chapters/ch20/20-031-_ensure_output_dir.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-031-_ensure_output_dir.py
"""

"""وحدة توليد الرسوم البيانية لتحليل المبيعات."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("output")

def _ensure_output_dir() -> None:
    """التأكد من وجود مجلد الإخراج."""
    OUTPUT_DIR.mkdir(exist_ok=True)

def plot_revenue_by_product(revenue: pd.Series) -> None:
    """رسم بياني شريطي لإيرادات المنتجات."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    revenue.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Revenue by Product", fontsize=14, fontweight="bold")
    plt.xlabel("Product")
    plt.ylabel("Revenue (SAR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_product.png", dpi=200)
    plt.close()

def plot_revenue_by_city_pie(revenue: pd.Series) -> None:
    """رسم دائري لتوزيع الإيرادات حسب المدينة."""
    _ensure_output_dir()
    plt.figure(figsize=(8, 8))
    plt.pie(revenue.values, labels=revenue.index, autopct="%1.1f%%", startangle=90)
    plt.title("Revenue Distribution by City", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_city.png", dpi=200)
    plt.close()

def plot_monthly_trend(trend: pd.Series) -> None:
    """رسم خطي لاتجاه المبيعات الشهري."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(trend.index, trend.values, marker="o", linewidth=2, color="darkgreen")
    plt.title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Revenue (SAR)")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_trend.png", dpi=200)
    plt.close()

def plot_city_product_heatmap(matrix: pd.DataFrame) -> None:
    """رسم Heatmap لمصفوفة المدينة/المنتج."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="Revenue (SAR)")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title("Revenue Heatmap: City vs Product", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "city_product_heatmap.png", dpi=200)
    plt.close()

# ────────────────────────────────────────────────────────────────────────
# [064] code/chapters/ch20/20-032-print_report.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-032-print_report.py
"""

"""نقطة الدخول الرئيسية لتحليل بيانات المبيعات."""

from sales_analysis.analysis import (
    city_product_matrix,
    generate_summary_report,
    monthly_trend,
    revenue_by_city,
    revenue_by_product,
    top_n_products_by_quantity,
)
from sales_analysis.data_loader import clean_sales_data, load_sales_data
from sales_analysis.visualization import (
    plot_city_product_heatmap,
    plot_monthly_trend,
    plot_revenue_by_city_pie,
    plot_revenue_by_product,
)

def print_report(summary: dict) -> None:
    """طباعة تقرير نصي منسّق في الطرفية."""
    print("=" * 50)
    print(" " * 15 + "تقرير تحليل المبيعات")
    print("=" * 50)
    for key, value in summary.items():
        label = key.replace("_", " ")
        if isinstance(value, float):
            print(f"{label}: {value:,.2f}")
        else:
            print(f"{label}: {value}")
    print("=" * 50)

def main() -> None:
    """تشغيل خط التحليل الكامل."""
    print("📊 تحميل البيانات...")
    raw_df = load_sales_data("data/sales_data.csv")

    print("🧹 تنظيف البيانات...")
    df = clean_sales_data(raw_df)
    print(f"   تم تحميل {len(df)} سجل بنجاح.\n")

    print("📈 تحليل البيانات...")
    summary = generate_summary_report(df)
    print_report(summary)

    print("\n🏆 أفضل 5 منتجات من حيث الكمية المباعة:")
    print(top_n_products_by_quantity(df, 5))

    print("\n🎨 توليد الرسوم البيانية...")
    plot_revenue_by_product(revenue_by_product(df))
    plot_revenue_by_city_pie(revenue_by_city(df))
    plot_monthly_trend(monthly_trend(df))
    plot_city_product_heatmap(city_product_matrix(df))

    print("✅ تم حفظ كل الرسوم البيانية في مجلد output/")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [065] code/chapters/ch20/20-033-students.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-033-students.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
students = pd.DataFrame({
    "الاسم": ["أحمد", "سارة", "خالد", "منى", "يوسف"],
    "الرياضيات": [85, 92, 78, 95, 88],
    "العلوم": [90, 85, 82, 91, 79],
})
print(students["الرياضيات"].mean())
print(students["العلوم"].mean())

# 2
numbers = np.random.randint(1, 101, 10)
print(numbers)
print("الأكبر:", numbers.max())
print("الأصغر:", numbers.min())
print("المتوسط:", numbers.mean())

# 3
days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء"]
hours = [2, 3, 1.5, 4, 2.5]
plt.bar(days, hours, color="purple")
plt.title("ساعات الدراسة")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [066] code/chapters/ch20/20-034-employees.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-034-employees.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# 4
employees = pd.DataFrame({
    "اسم": ["علي", "فاطمة", "سعد", "ليلى", "عمر"],
    "القسم": ["مبيعات", "تقنية", "مبيعات", "تقنية", "مالية"],
    "الراتب": [8000, 12000, 8500, 13000, 9500],
})
avg_salary = employees.groupby("القسم")["الراتب"].mean()
print(avg_salary)
avg_salary.plot(kind="bar", color="teal")
plt.title("متوسط الراتب حسب القسم")
plt.ylabel("الراتب")
plt.tight_layout()
plt.show()

# 5
data = pd.DataFrame({"العمر": [25, 30, None, 45, 28, None, 90]})
# الوسيط أفضل من المتوسط هنا لأن القيمة 90 (قيمة متطرفة/Outlier)
# تؤثر بشدة على المتوسط لكن تأثيرها على الوسيط محدود جدًا
data["العمر"] = data["العمر"].fillna(data["العمر"].median())
print(data)

# ────────────────────────────────────────────────────────────────────────
# [067] code/chapters/ch20/20-035-best_selling_month_per_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-035-best_selling_month_per_product.py
"""

# إضافة إلى analysis.py
def best_selling_month_per_product(df: pd.DataFrame) -> pd.Series:
    """الشهر الأفضل مبيعًا لكل منتج."""
    grouped = df.groupby(["المنتج", "الشهر"])["الإجمالي"].sum()
    return grouped.groupby(level="المنتج").idxmax()

# إضافة إلى visualization.py
def plot_quarterly_comparison(df: pd.DataFrame) -> None:
    """مقارنة مبيعات الأرباع الأربعة لكل مدينة."""
    _ensure_output_dir()
    pivot = df.pivot_table(
        values="الإجمالي", index="ربع_السنة", columns="المدينة",
        aggfunc="sum", fill_value=0,
    )
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Quarterly Revenue Comparison by City")
    plt.xlabel("Quarter")
    plt.ylabel("Revenue (SAR)")
    plt.legend(title="City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quarterly_comparison.png", dpi=200)
    plt.close()

# ────────────────────────────────────────────────────────────────────────
# [068] code/chapters/ch20/20-036-arr.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-036-arr.py
"""

# ===== NumPy =====
import numpy as np
arr = np.array([1, 2, 3])
arr.sum(); arr.mean(); arr.max(); arr.min(); arr.std()
filtered = arr[arr > 1]                      # Boolean Indexing

# ===== pandas: القراءة والاستكشاف =====
import pandas as pd
df = pd.read_csv("file.csv", encoding="utf-8")
df.head(); df.tail(); df.info(); df.describe()
df.shape; df.columns; df.dtypes
df.isna().sum()                              # القيم المفقودة

# ===== pandas: الاختيار والتصفية =====
df["col"]                                    # عمود -> Series
df[["col1", "col2"]]                         # أعمدة -> DataFrame
df.loc[0, "col"]                             # بالتسمية
df.iloc[0, 1]                                # بالموضع
df[(df["a"] > 5) & (df["b"] < 10)]           # تصفية بشروط (& | وليس and or)

# ===== pandas: التعديل =====
df["new_col"] = df["a"] * df["b"]
df = df.drop(columns=["col"])
df = df.rename(columns={"old": "new"})
df["col"] = df["col"].fillna(df["col"].mean())
df = df.dropna(subset=["col"])
df = df.sort_values("col", ascending=False)

# ===== pandas: التجميع والدمج =====
df.groupby("col")["val"].sum()
df.groupby("col")["val"].agg(["sum", "mean", "count"])
df.pivot_table(values="val", index="r", columns="c", aggfunc="sum")
pd.merge(df1, df2, on="key", how="left")
pd.concat([df1, df2], ignore_index=True)

# ===== pandas: التصدير =====
df.to_csv("out.csv", index=False, encoding="utf-8-sig")

# ===== Matplotlib =====
import matplotlib.pyplot as plt
plt.plot(x, y); plt.bar(x, y); plt.pie(values, labels=labels)
plt.scatter(x, y); plt.hist(data, bins=20)
plt.title(""); plt.xlabel(""); plt.ylabel(""); plt.legend()
plt.savefig("chart.png", dpi=200); plt.show(); plt.close()

# ────────────────────────────────────────────────────────────────────────
# [069] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> ```

---

## 20.3 NumPy: أساس الحوسبة الرقمية

### لماذا لا نستخدم القوائم العادية؟

في الفصل 8، تعلمنا القوائم (`list`). لماذا نحتاج شيئًا آخر؟

# ────────────────────────────────────────────────────────────────────────
# [070] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: Broadcasting قد يُربك المبتدئين**
>
> عندما تجري عملية بين مصفوفة وعدد مفرد (scalar)، يطبّق NumPy العملية على كل عنصر تلقائيًا — هذا يُسمى **Broadcasting**:
> ```python
> arr = np.array([1, 2, 3])
> print(arr + 10)  # [11 12 13] -- وليس خطأ!
> ```
> هذا سلوك **مقصود ومفيد جدًا**، لكن يجب أن تفهمه حتى لا يفاجئك.

> **💡 نصيحة خبير:** NumPy هو الأساس الذي تُبنى عليه pandas وscikit-learn وTensorFlow. لن تحتاج غالبًا لاستخدامه مباشرة بشكل مكثف في تحليل البيانات اليومي (ستستخدم pandas بدلاً منه)، لكن **فهمه ضروري** لأن pandas مبني عليه بالكامل، وستقابله باستمرار.

---

## 20.4 pandas: العمود الفقري لتحليل البيانات

إذا كان NumPy يتعامل مع أرقام في مصفوفات، فإن **pandas** يتعامل مع بيانات **جدولية** حقيقية: صفوف وأعمدة لها أسماء، أنواع بيانات مختلفة (نصوص، أرقام، تواريخ)، وقيم مفقودة — تمامًا كملف Excel أو جدول في قاعدة بيانات.

### هيكلا البيانات الأساسيان

**1. `Series`**: عمود واحد من البيانات (كـ«قائمة» لكن أذكى، بفهارس مسمّاة).

# ────────────────────────────────────────────────────────────────────────
# [071] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: دعم اللغة العربية في matplotlib**
> matplotlib لا يدعم اتجاه الكتابة من اليمين لليسار (RTL) أو ربط الحروف العربية بشكل جيد افتراضيًا — قد تظهر الحروف منفصلة أو معكوسة. للحصول على نص عربي صحيح، استخدم مكتبة `arabic-reshaper` مع `python-bidi`:
> ```python
> import arabic_reshaper
> from bidi.algorithm import get_display
>
> def fix_arabic(text: str) -> str:
>     reshaped = arabic_reshaper.reshape(text)
>     return get_display(reshaped)
>
> plt.title(fix_arabic(«المبيعات الشهرية»))
> ```
> في الأمثلة التالية سنستخدم أحيانًا تسميات إنجليزية للتبسيط، لكن اعتمد هذا الحل في مشاريعك الحقيقية عند الحاجة لنصوص عربية في الرسوم.

### أنواع الرسوم الشائعة

# ────────────────────────────────────────────────────────────────────────
# [072] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
> **جدول مرجعي: أي رسم تختار؟**
>
> | نوع البيانات | السؤال | الرسم المناسب |
> |---|---|---|
> | تطور عبر الزمن | هل المبيعات تزداد؟ | `plt.plot()` (خطي) |
> | مقارنة فئات | أي منتج الأكثر مبيعًا؟ | `plt.bar()` (شريطي) |
> | نسب من كل | ما حصة كل قسم من الميزانية؟ | `plt.pie()` (دائري) |
> | علاقة بين متغيرين | هل السعر يؤثر على الكمية المباعة؟ | `plt.scatter()` (تشتت) |
> | توزيع قيم | كيف تتوزع أعمار العملاء؟ | `plt.hist()` (مدرج تكراري) |

### تخصيص الرسوم

# ────────────────────────────────────────────────────────────────────────
# [073] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
> **💡 نصيحة خبير:** استخدمنا `plt.close()` بعد كل `savefig()` بدلاً من `plt.show()`. هذا ضروري في السكربتات التي تنتج عدة رسوم متتالية، لأنه يُحرر الذاكرة ويمنع تراكب الرسوم فوق بعضها بالخطأ.

### الخطوة 5: `main.py` — تجميع كل شيء

# ────────────────────────────────────────────────────────────────────────
# [074] dist/markdown/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
</details>

---

## 20.9 اختبر فهمك (Quiz)

<details>
<summary>1. ما الفرق الجوهري بين `list` العادية و `numpy.ndarray`؟</summary>

`ndarray` تتطلب أن تكون كل عناصرها من نفس النوع (موحدة)، وتدعم العمليات الحسابية مباشرة على كل العناصر دفعة واحدة (Vectorization) دون حلقات، وهي أسرع بكثير وأقل استهلاكًا للذاكرة لأنها منفذة بلغة C داخليًا. القوائم العادية مرنة (تقبل أنواعًا مختلطة) لكنها أبطأ في الحسابات الرقمية.
</details>

<details>
<summary>2. لماذا تفشل `df[(df["a"] > 5) and (df["b"] < 10)]` وما البديل الصحيح؟</summary>

لأن `and`/`or` في Python تتوقع قيمة Boolean مفردة، بينما الشرط `(df["a"] > 5)` يُرجع `Series` كاملة من قيم True/False وليس قيمة واحدة. البديل الصحيح هو استخدام `&` و`|` مع وضع أقواس حول كل شرط فرعي: `df[(df["a"] > 5) & (df["b"] < 10)]`.
</details>

<details>
<summary>3. متى تستخدم `loc` ومتى تستخدم `iloc`؟</summary>

`loc` يُستخدم للاختيار بناءً على **التسمية/الاسم** (label-based)، مثل اسم العمود أو قيمة الفهرس الفعلية. `iloc` يُستخدم للاختيار بناءً على **الموضع الرقمي** (position-based) تمامًا كالقوائم العادية (0، 1، 2...)، بغض النظر عن تسميات الفهرس.
</details>

<details>
<summary>4. ما وظيفة `groupby` ومتى تستخدمها؟</summary>

`groupby` يُستخدم لتقسيم البيانات إلى مجموعات بناءً على قيمة عمود (أو أكثر)، ثم تطبيق دالة تجميعية (sum, mean, count...) على كل مجموعة على حدة. يُستخدم للإجابة على أسئلة مثل «ما إجمالي المبيعات لكل مدينة؟» وهو مكافئ لـ `GROUP BY` في SQL.
</details>

<details>
<summary>5. لماذا نستخدم `.copy()` أحيانًا بعد تصفية DataFrame؟</summary>

لتجنب تحذير `SettingWithCopyWarning` والسلوك غير المتوقع الناتج عن أن الجزء المُصفّى قد يكون «عرضًا» (view) مرتبطًا بالبيانات الأصلية بدلاً من نسخة مستقلة. استخدام `.copy()` يضمن أننا نعمل على نسخة منفصلة تمامًا وآمنة للتعديل.
</details>

<details>
<summary>6. ما الفرق بين `plt.show()` و `plt.savefig()`؟</summary>

`plt.show()` يفتح نافذة تفاعلية لعرض الرسم على الشاشة مباشرة، بينما `plt.savefig()` يحفظ الرسم كملف صورة (PNG, JPG, PDF...) على القرص دون بالضرورة عرضه، وهذا مفيد جدًا في السكربتات الآلية التي تولّد عدة رسوم دون تدخل بشري.
</details>

---

## 20.10 ملخص الفصل (Cheat Sheet)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["pandas", "numpy", "matplotlib", "openpyxl", "arabic_reshaper", "bidi"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q pandas numpy matplotlib openpyxl arabic-reshaper python-bidi

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# مجلدات الإخراج التي تكتب فيها الأمثلة
pathlib.Path('data').mkdir(exist_ok=True)

# إعداد خاص بهذا الفصل
import pandas as _pd
_sales = _pd.DataFrame({
    'المنتج': ['لابتوب', 'ماوس', 'شاشة', 'لابتوب', 'ماوس'],
    'المدينة': ['الرياض', 'جدة', 'الرياض', 'جدة', 'الرياض'],
    'السعر': [4500, 120, 900, 4700, 110],
    'الكمية': [2, 10, 3, 1, 8],
})
if not pathlib.Path('sales.csv').exists():
    _sales.to_csv('sales.csv', index=False, encoding='utf-8')
    _sales.to_json('sales.json', orient='records', force_ascii=False)
    _sales.to_excel('sales.xlsx', index=False, sheet_name='2024')
    print('أُنشئت ملفات المبيعات النموذجية: sales.csv / sales.json / sales.xlsx')

# --- عرض العربية في رسوم Matplotlib بشكل متصل وصحيح الاتجاه ---
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _shape = lambda s: get_display(arabic_reshaper.reshape(s))
    import matplotlib.text as _mt
    _orig = _mt.Text.set_text
    def _set_text(self, s):
        if isinstance(s, str) and any('\u0600' <= c <= '\u06FF' for c in s):
            s = _shape(s)
        return _orig(self, s)
    _mt.Text.set_text = _set_text
    print("دعم العربية في الرسوم: مُفعّل")
except ImportError:
    print("تنبيه: ثبّت arabic-reshaper وpython-bidi لعرض العربية في الرسوم")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# محاولة جمع قائمتين من الأرقام عنصرًا بعنصر
prices = [10, 20, 30]
discounts = [1, 2, 3]

# هذا لن يعمل كما تتوقع!
result = prices + discounts
print(result)  # [10, 20, 30, 1, 2, 3] -- دمج وليس جمع!

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
result = [p - d for p, d in zip(prices, discounts)]
print(result)  # [9, 18, 27]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

prices = np.array([10, 20, 30])
discounts = np.array([1, 2, 3])

result = prices - discounts
print(result)  # [9 18 27]
print(type(result))  # <class 'numpy.ndarray'>

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# من قائمة
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.dtype)    # int64 (نوع البيانات الموحّد)

# مصفوفة ثنائية الأبعاد (مصفوفة من مصفوفات = جدول)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(matrix.shape)  # (2, 3) -- صفّان، 3 أعمدة

# دوال إنشاء جاهزة
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))        # مصفوفة 2x3 من الآحاد
range_arr = np.arange(0, 10, 2)   # [0 2 4 6 8]  (مثل range لكن تُرجع array)
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ] -- 5 قيم متباعدة بالتساوي
random_arr = np.random.rand(3)    # 3 أرقام عشوائية بين 0 و1

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
prices = np.array([100, 200, 300, 400])

# عملية على كل عنصر دفعة واحدة
with_tax = prices * 1.15
print(with_tax)  # [115. 230. 345. 460.]

# مقارنة تُرجع مصفوفة من Boolean
expensive = prices > 250
print(expensive)  # [False False  True  True]

# استخدام هذه المصفوفة Boolean للتصفية! (Boolean Indexing)
print(prices[expensive])  # [300 400]

# أو في سطر واحد
print(prices[prices > 250])  # [300 400]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
sales = np.array([120, 340, 210, 450, 90, 380])

print(sales.sum())     # 1590 -- المجموع
print(sales.mean())    # 265.0 -- المتوسط
print(sales.max())     # 450 -- الأكبر
print(sales.min())     # 90 -- الأصغر
print(sales.std())     # الانحراف المعياري
print(sales.argmax())  # 3 -- فهرس العنصر الأكبر
print(np.median(sales))  # الوسيط

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

temperatures = pd.Series([22, 25, 19, 30], index=["الأحد", "الاثنين", "الثلاثاء", "الأربعاء"])
print(temperatures)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
data = {
    "اسم المنتج": ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة"],
    "السعر": [3500, 50, 120, 900],
    "الكمية": [10, 50, 30, 15],
}

df = pd.DataFrame(data)
print(df)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# من CSV
df = pd.read_csv("sales.csv")

# من Excel (يتطلب: uv add openpyxl)
df = pd.read_excel("sales.xlsx", sheet_name="2024")

# من JSON
df = pd.read_json("sales.json")

# تحديد الترميز إن كانت البيانات عربية
df = pd.read_csv("sales.csv", encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
df.head()       # أول 5 صفوف (الافتراضي)
df.head(10)     # أول 10 صفوف
df.tail(3)      # آخر 3 صفوف

df.info()       # معلومات عامة: الأعمدة، الأنواع، القيم غير الفارغة
df.describe()   # إحصائيات (متوسط، انحراف معياري، أدنى، أقصى...) للأعمدة الرقمية

df.shape        # (عدد الصفوف, عدد الأعمدة) -- خاصية وليست دالة!
df.columns      # أسماء الأعمدة
df.dtypes       # نوع بيانات كل عمود
df.index        # معلومات الفهرس

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس"],
    "السعر": [3500, 50, 900, 3600, 55],
    "الكمية": [2, 10, 3, 1, 8],
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام", "جدة"],
})

print(df.info())

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# اختيار عمود واحد -> يُرجع Series
print(df["المنتج"])

# اختيار عدة أعمدة -> يُرجع DataFrame (لاحظ قائمة داخل قوسين)
print(df[["المنتج", "السعر"]])

# loc: الاختيار بالاسم/التسمية
print(df.loc[0])              # الصف الأول (حسب التسمية، وهنا تتطابق مع الرقم)
print(df.loc[0, "المنتج"])    # قيمة محددة
print(df.loc[0:2, "السعر"])   # نطاق من الصفوف، عمود محدد

# iloc: الاختيار بالموضع الرقمي (مثل القوائم تمامًا)
print(df.iloc[0])        # أول صف بغض النظر عن تسميته
print(df.iloc[0, 1])     # الصف 0، العمود 1
print(df.iloc[-1])       # آخر صف

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# كل المنتجات التي سعرها أكبر من 500
expensive = df[df["السعر"] > 500]
print(expensive)

# شروط متعددة: استخدم & و | (وليس and/or) مع أقواس إلزامية
cheap_riyadh = df[(df["السعر"] < 1000) & (df["المدينة"] == "الرياض")]
print(cheap_riyadh)

# التحقق من انتماء القيمة لقائمة
target_cities = df[df["المدينة"].isin(["جدة", "الدمام"])]

# استبعاد قيم
not_riyadh = df[df["المدينة"] != "الرياض"]

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# إضافة عمود جديد محسوب
df["الإجمالي"] = df["السعر"] * df["الكمية"]
print(df)

# إضافة عمود بشرط باستخدام numpy.where
df["الفئة"] = np.where(df["السعر"] > 500, "غالي", "رخيص")

# تعديل عمود موجود
df["السعر"] = df["السعر"] * 1.15  # إضافة ضريبة 15%

# حذف عمود
df = df.drop(columns=["الفئة"])

# إعادة تسمية الأعمدة
df = df.rename(columns={"السعر": "سعر_الوحدة"})

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
data = {
    "اسم": ["أحمد", "سارة", None, "خالد"],
    "العمر": [25, None, 30, 28],
}
df = pd.DataFrame(data)

print(df.isna())          # جدول Boolean يوضح أماكن القيم المفقودة
print(df.isna().sum())    # عدد القيم المفقودة في كل عمود

# حذف الصفوف التي تحتوي أي قيمة مفقودة
clean_df = df.dropna()

# حذف فقط إذا كان عمود معين فارغًا
clean_df = df.dropna(subset=["اسم"])

# تعويض القيم المفقودة بقيمة محددة
df["العمر"] = df["العمر"].fillna(df["العمر"].mean())  # بالمتوسط
df["اسم"] = df["اسم"].fillna("غير معروف")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# 🔄 نعيد تعريف df ببيانات المبيعات حتى يعمل المثال كما في الكتاب
df = pd.DataFrame({
    'المنتج': ['لابتوب', 'ماوس', 'شاشة', 'لابتوب', 'ماوس'],
    'المدينة': ['الرياض', 'جدة', 'الرياض', 'جدة', 'الرياض'],
    'السعر': [4500, 120, 900, 4700, 110],
    'الكمية': [2, 10, 3, 1, 8],
})

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
df_sorted = df.sort_values("السعر")                    # تصاعدي
df_sorted = df.sort_values("السعر", ascending=False)    # تنازلي
df_sorted = df.sort_values(["المدينة", "السعر"])         # فرز متعدد المستويات

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس", "شاشة"],
    "المدينة": ["الرياض", "جدة", "الرياض", "جدة", "الرياض", "جدة"],
    "الإجمالي": [7000, 500, 2700, 3600, 450, 1800],
})

# مجموع المبيعات لكل مدينة
by_city = df.groupby("المدينة")["الإجمالي"].sum()
print(by_city)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# عدة إحصائيات دفعة واحدة
summary = df.groupby("المدينة")["الإجمالي"].agg(["sum", "mean", "count", "max"])
print(summary)

# التجميع حسب أكثر من عمود
by_city_product = df.groupby(["المدينة", "المنتج"])["الإجمالي"].sum()
print(by_city_product)

# تحويل النتيجة إلى DataFrame عادي مرة أخرى
result = df.groupby("المدينة")["الإجمالي"].sum().reset_index()

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
pivot = df.pivot_table(
    values="الإجمالي",
    index="المدينة",
    columns="المنتج",
    aggfunc="sum",
    fill_value=0,
)
print(pivot)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "اسم": ["أحمد", "سارة", "خالد"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "المبلغ": [200, 450, 120, 300],
})

# الدمج (مثل JOIN في SQL)
merged = pd.merge(orders, customers, on="customer_id", how="left")
print(merged)

# دمج/تكديس DataFrames فوق بعضها (مثل إلحاق ملفات شهر بشهر)
jan_sales = pd.DataFrame({"المنتج": ["لابتوب"], "الإجمالي": [3500]})
feb_sales = pd.DataFrame({"المنتج": ["ماوس"], "الإجمالي": [50]})
all_sales = pd.concat([jan_sales, feb_sales], ignore_index=True)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "التاريخ": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "المبيعات": [1000, 1500, 1200],
})

# تحويل النص إلى تاريخ حقيقي
df["التاريخ"] = pd.to_datetime(df["التاريخ"])

# استخراج معلومات من التاريخ
df["الشهر"] = df["التاريخ"].dt.month
df["اسم_اليوم"] = df["التاريخ"].dt.day_name()

print(df)

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
df.to_excel("output.xlsx", index=False)
df.to_json("output.json", orient="records", force_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

months = ["يناير", "فبراير", "مارس", "أبريل", "مايو"]
sales = [1200, 1900, 1500, 2200, 2800]

plt.plot(months, sales)
plt.title("المبيعات الشهرية")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.show()  # يفتح نافذة لعرض الرسم

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

categories = ["لابتوب", "ماوس", "شاشة", "لوحة مفاتيح"]
values = [45, 120, 30, 80]

# رسم بياني خطي -- مناسب للاتجاهات عبر الزمن
plt.plot(categories, values, marker="o")
plt.show()

# رسم بياني شريطي (Bar Chart) -- مناسب للمقارنة بين فئات
plt.bar(categories, values, color="steelblue")
plt.show()

# رسم دائري (Pie Chart) -- مناسب لإظهار النسب من الكل
plt.pie(values, labels=categories, autopct="%1.1f%%")
plt.show()

# مخطط التشتت (Scatter Plot) -- مناسب لإظهار العلاقة بين متغيرين
prices = [3500, 50, 900, 120]
plt.scatter(prices, values)
plt.xlabel("السعر")
plt.ylabel("الكمية المباعة")
plt.show()

# المدرج التكراري (Histogram) -- مناسب لتوزيع البيانات
import numpy as np
ages = np.random.normal(30, 5, 1000)  # 1000 عمر عشوائي حول متوسط 30
plt.hist(ages, bins=20, color="coral", edgecolor="black")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))  # حجم الرسم (عرض, ارتفاع) بالبوصة

plt.plot(months, sales, marker="o", linestyle="--", color="green", linewidth=2, label="2024")
plt.title("المبيعات الشهرية", fontsize=16, fontweight="bold")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.legend()        # إظهار الـ label الموجود أعلاه
plt.grid(True, alpha=0.3)   # شبكة خفيفة في الخلفية
plt.tight_layout()  # ترتيب تلقائي لمنع تداخل العناصر

plt.savefig("sales_chart.png", dpi=300)  # حفظ كصورة (بدلاً من/بالإضافة إلى show)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # صف واحد، عمودان

axes[0].plot(months, sales, marker="o")
axes[0].set_title("خطي")

axes[1].bar(months, sales, color="orange")
axes[1].set_title("شريطي")

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame({
    "الشهر": ["يناير", "فبراير", "مارس"],
    "المبيعات": [1200, 1900, 1500],
})

df.plot(x="الشهر", y="المبيعات", kind="bar", title="المبيعات الشهرية")
plt.show()

# أو مباشرة من نتيجة groupby
df.groupby("الشهر")["المبيعات"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
"""سكربت مساعد لتوليد بيانات مبيعات تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)  # لضمان نتائج قابلة لإعادة الإنتاج

n_records = 1000

products = ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة", "سماعات", "طابعة"]
product_prices = {
    "لابتوب": 3500, "ماوس": 50, "لوحة مفاتيح": 120,
    "شاشة": 900, "سماعات": 150, "طابعة": 600,
}
cities = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]

data = {
    "order_id": range(1, n_records + 1),
    "التاريخ": pd.date_range("2024-01-01", "2024-12-31", periods=n_records),
    "المنتج": np.random.choice(products, n_records),
    "المدينة": np.random.choice(cities, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "الكمية": np.random.randint(1, 6, n_records),
}

df = pd.DataFrame(data)
df["سعر_الوحدة"] = df["المنتج"].map(product_prices)
df["الإجمالي"] = df["سعر_الوحدة"] * df["الكمية"]

# إدخال بعض القيم المفقودة عمدًا (كما يحدث في البيانات الحقيقية)
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, "المدينة"] = None

df.to_csv("data/sales_data.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n_records} سجل مبيعات في data/sales_data.csv")

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
students = pd.DataFrame({
    "الاسم": ["أحمد", "سارة", "خالد", "منى", "يوسف"],
    "الرياضيات": [85, 92, 78, 95, 88],
    "العلوم": [90, 85, 82, 91, 79],
})
print(students["الرياضيات"].mean())
print(students["العلوم"].mean())

# 2
numbers = np.random.randint(1, 101, 10)
print(numbers)
print("الأكبر:", numbers.max())
print("الأصغر:", numbers.min())
print("المتوسط:", numbers.mean())

# 3
days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء"]
hours = [2, 3, 1.5, 4, 2.5]
plt.bar(days, hours, color="purple")
plt.title("ساعات الدراسة")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

# 4
employees = pd.DataFrame({
    "اسم": ["علي", "فاطمة", "سعد", "ليلى", "عمر"],
    "القسم": ["مبيعات", "تقنية", "مبيعات", "تقنية", "مالية"],
    "الراتب": [8000, 12000, 8500, 13000, 9500],
})
avg_salary = employees.groupby("القسم")["الراتب"].mean()
print(avg_salary)
avg_salary.plot(kind="bar", color="teal")
plt.title("متوسط الراتب حسب القسم")
plt.ylabel("الراتب")
plt.tight_layout()
plt.show()

# 5
data = pd.DataFrame({"العمر": [25, 30, None, 45, 28, None, 90]})
# الوسيط أفضل من المتوسط هنا لأن القيمة 90 (قيمة متطرفة/Outlier)
# تؤثر بشدة على المتوسط لكن تأثيرها على الوسيط محدود جدًا
data["العمر"] = data["العمر"].fillna(data["العمر"].median())
print(data)

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-001-prices.py
"""

# محاولة جمع قائمتين من الأرقام عنصرًا بعنصر
prices = [10, 20, 30]
discounts = [1, 2, 3]

# هذا لن يعمل كما تتوقع!
result = prices + discounts
print(result)  # [10, 20, 30, 1, 2, 3] -- دمج وليس جمع!

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-002-result.py
"""

result = [p - d for p, d in zip(prices, discounts)]
print(result)  # [9, 18, 27]

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-003-prices.py
"""

import numpy as np

prices = np.array([10, 20, 30])
discounts = np.array([1, 2, 3])

result = prices - discounts
print(result)  # [9 18 27]
print(type(result))  # <class 'numpy.ndarray'>

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-004-ex20_3.py
"""

import numpy as np

# من قائمة
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.dtype)    # int64 (نوع البيانات الموحّد)

# مصفوفة ثنائية الأبعاد (مصفوفة من مصفوفات = جدول)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(matrix.shape)  # (2, 3) -- صفّان، 3 أعمدة

# دوال إنشاء جاهزة
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))        # مصفوفة 2x3 من الآحاد
range_arr = np.arange(0, 10, 2)   # [0 2 4 6 8]  (مثل range لكن تُرجع array)
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ] -- 5 قيم متباعدة بالتساوي
random_arr = np.random.rand(3)    # 3 أرقام عشوائية بين 0 و1

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-005-prices.py
"""

prices = np.array([100, 200, 300, 400])

# عملية على كل عنصر دفعة واحدة
with_tax = prices * 1.15
print(with_tax)  # [115. 230. 345. 460.]

# مقارنة تُرجع مصفوفة من Boolean
expensive = prices > 250
print(expensive)  # [False False  True  True]

# استخدام هذه المصفوفة Boolean للتصفية! (Boolean Indexing)
print(prices[expensive])  # [300 400]

# أو في سطر واحد
print(prices[prices > 250])  # [300 400]

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.3: NumPy: أساس الحوسبة الرقمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-006-sales.py
"""

sales = np.array([120, 340, 210, 450, 90, 380])

print(sales.sum())     # 1590 -- المجموع
print(sales.mean())    # 265.0 -- المتوسط
print(sales.max())     # 450 -- الأكبر
print(sales.min())     # 90 -- الأصغر
print(sales.std())     # الانحراف المعياري
print(sales.argmax())  # 3 -- فهرس العنصر الأكبر
print(np.median(sales))  # الوسيط

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-007-temperatures.py
"""

import pandas as pd

temperatures = pd.Series([22, 25, 19, 30], index=["الأحد", "الاثنين", "الثلاثاء", "الأربعاء"])
print(temperatures)

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-008-data.py
"""

data = {
    "اسم المنتج": ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة"],
    "السعر": [3500, 50, 120, 900],
    "الكمية": [10, 50, 30, 15],
}

df = pd.DataFrame(data)
print(df)

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-009-ex20_4.py
"""

# من CSV
df = pd.read_csv("sales.csv")

# من Excel (يتطلب: uv add openpyxl)
df = pd.read_excel("sales.xlsx", sheet_name="2024")

# من JSON
df = pd.read_json("sales.json")

# تحديد الترميز إن كانت البيانات عربية
df = pd.read_csv("sales.csv", encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-010-ex20_4.py
"""

df.head()       # أول 5 صفوف (الافتراضي)
df.head(10)     # أول 10 صفوف
df.tail(3)      # آخر 3 صفوف

df.info()       # معلومات عامة: الأعمدة، الأنواع، القيم غير الفارغة
df.describe()   # إحصائيات (متوسط، انحراف معياري، أدنى، أقصى...) للأعمدة الرقمية

df.shape        # (عدد الصفوف, عدد الأعمدة) -- خاصية وليست دالة!
df.columns      # أسماء الأعمدة
df.dtypes       # نوع بيانات كل عمود
df.index        # معلومات الفهرس

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-011-ex20_4.py
"""

import pandas as pd

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس"],
    "السعر": [3500, 50, 900, 3600, 55],
    "الكمية": [2, 10, 3, 1, 8],
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام", "جدة"],
})

print(df.info())

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-012-ex20_4.py
"""

# اختيار عمود واحد -> يُرجع Series
print(df["المنتج"])

# اختيار عدة أعمدة -> يُرجع DataFrame (لاحظ قائمة داخل قوسين)
print(df[["المنتج", "السعر"]])

# loc: الاختيار بالاسم/التسمية
print(df.loc[0])              # الصف الأول (حسب التسمية، وهنا تتطابق مع الرقم)
print(df.loc[0, "المنتج"])    # قيمة محددة
print(df.loc[0:2, "السعر"])   # نطاق من الصفوف، عمود محدد

# iloc: الاختيار بالموضع الرقمي (مثل القوائم تمامًا)
print(df.iloc[0])        # أول صف بغض النظر عن تسميته
print(df.iloc[0, 1])     # الصف 0، العمود 1
print(df.iloc[-1])       # آخر صف

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-013-expensive.py
"""

# كل المنتجات التي سعرها أكبر من 500
expensive = df[df["السعر"] > 500]
print(expensive)

# شروط متعددة: استخدم & و | (وليس and/or) مع أقواس إلزامية
cheap_riyadh = df[(df["السعر"] < 1000) & (df["المدينة"] == "الرياض")]
print(cheap_riyadh)

# التحقق من انتماء القيمة لقائمة
target_cities = df[df["المدينة"].isin(["جدة", "الدمام"])]

# استبعاد قيم
not_riyadh = df[df["المدينة"] != "الرياض"]

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-014-ex20_4.py
"""

# إضافة عمود جديد محسوب
df["الإجمالي"] = df["السعر"] * df["الكمية"]
print(df)

# إضافة عمود بشرط باستخدام numpy.where
df["الفئة"] = np.where(df["السعر"] > 500, "غالي", "رخيص")

# تعديل عمود موجود
df["السعر"] = df["السعر"] * 1.15  # إضافة ضريبة 15%

# حذف عمود
df = df.drop(columns=["الفئة"])

# إعادة تسمية الأعمدة
df = df.rename(columns={"السعر": "سعر_الوحدة"})

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-015-data.py
"""

data = {
    "اسم": ["أحمد", "سارة", None, "خالد"],
    "العمر": [25, None, 30, 28],
}
df = pd.DataFrame(data)

print(df.isna())          # جدول Boolean يوضح أماكن القيم المفقودة
print(df.isna().sum())    # عدد القيم المفقودة في كل عمود

# حذف الصفوف التي تحتوي أي قيمة مفقودة
clean_df = df.dropna()

# حذف فقط إذا كان عمود معين فارغًا
clean_df = df.dropna(subset=["اسم"])

# تعويض القيم المفقودة بقيمة محددة
df["العمر"] = df["العمر"].fillna(df["العمر"].mean())  # بالمتوسط
df["اسم"] = df["اسم"].fillna("غير معروف")

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-016-df_sorted.py
"""

df_sorted = df.sort_values("السعر")                    # تصاعدي
df_sorted = df.sort_values("السعر", ascending=False)    # تنازلي
df_sorted = df.sort_values(["المدينة", "السعر"])         # فرز متعدد المستويات

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-017-ex20_4.py
"""

df = pd.DataFrame({
    "المنتج": ["لابتوب", "ماوس", "شاشة", "لابتوب", "ماوس", "شاشة"],
    "المدينة": ["الرياض", "جدة", "الرياض", "جدة", "الرياض", "جدة"],
    "الإجمالي": [7000, 500, 2700, 3600, 450, 1800],
})

# مجموع المبيعات لكل مدينة
by_city = df.groupby("المدينة")["الإجمالي"].sum()
print(by_city)

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-018-summary.py
"""

# عدة إحصائيات دفعة واحدة
summary = df.groupby("المدينة")["الإجمالي"].agg(["sum", "mean", "count", "max"])
print(summary)

# التجميع حسب أكثر من عمود
by_city_product = df.groupby(["المدينة", "المنتج"])["الإجمالي"].sum()
print(by_city_product)

# تحويل النتيجة إلى DataFrame عادي مرة أخرى
result = df.groupby("المدينة")["الإجمالي"].sum().reset_index()

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-019-pivot.py
"""

pivot = df.pivot_table(
    values="الإجمالي",
    index="المدينة",
    columns="المنتج",
    aggfunc="sum",
    fill_value=0,
)
print(pivot)

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-020-customers.py
"""

customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "اسم": ["أحمد", "سارة", "خالد"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "المبلغ": [200, 450, 120, 300],
})

# الدمج (مثل JOIN في SQL)
merged = pd.merge(orders, customers, on="customer_id", how="left")
print(merged)

# دمج/تكديس DataFrames فوق بعضها (مثل إلحاق ملفات شهر بشهر)
jan_sales = pd.DataFrame({"المنتج": ["لابتوب"], "الإجمالي": [3500]})
feb_sales = pd.DataFrame({"المنتج": ["ماوس"], "الإجمالي": [50]})
all_sales = pd.concat([jan_sales, feb_sales], ignore_index=True)

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-021-ex20_4.py
"""

df = pd.DataFrame({
    "التاريخ": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "المبيعات": [1000, 1500, 1200],
})

# تحويل النص إلى تاريخ حقيقي
df["التاريخ"] = pd.to_datetime(df["التاريخ"])

# استخراج معلومات من التاريخ
df["الشهر"] = df["التاريخ"].dt.month
df["اسم_اليوم"] = df["التاريخ"].dt.day_name()

print(df)

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.4: pandas: العمود الفقري لتحليل البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-022-ex20_4.py
"""

df.to_csv("output.csv", index=False, encoding="utf-8-sig")
df.to_excel("output.xlsx", index=False)
df.to_json("output.json", orient="records", force_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-023-months.py
"""

import matplotlib.pyplot as plt

months = ["يناير", "فبراير", "مارس", "أبريل", "مايو"]
sales = [1200, 1900, 1500, 2200, 2800]

plt.plot(months, sales)
plt.title("المبيعات الشهرية")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.show()  # يفتح نافذة لعرض الرسم

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-024-categories.py
"""

import matplotlib.pyplot as plt

categories = ["لابتوب", "ماوس", "شاشة", "لوحة مفاتيح"]
values = [45, 120, 30, 80]

# رسم بياني خطي -- مناسب للاتجاهات عبر الزمن
plt.plot(categories, values, marker="o")
plt.show()

# رسم بياني شريطي (Bar Chart) -- مناسب للمقارنة بين فئات
plt.bar(categories, values, color="steelblue")
plt.show()

# رسم دائري (Pie Chart) -- مناسب لإظهار النسب من الكل
plt.pie(values, labels=categories, autopct="%1.1f%%")
plt.show()

# مخطط التشتت (Scatter Plot) -- مناسب لإظهار العلاقة بين متغيرين
prices = [3500, 50, 900, 120]
plt.scatter(prices, values)
plt.xlabel("السعر")
plt.ylabel("الكمية المباعة")
plt.show()

# المدرج التكراري (Histogram) -- مناسب لتوزيع البيانات
import numpy as np
ages = np.random.normal(30, 5, 1000)  # 1000 عمر عشوائي حول متوسط 30
plt.hist(ages, bins=20, color="coral", edgecolor="black")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-025-ex20_5.py
"""

plt.figure(figsize=(10, 6))  # حجم الرسم (عرض, ارتفاع) بالبوصة

plt.plot(months, sales, marker="o", linestyle="--", color="green", linewidth=2, label="2024")
plt.title("المبيعات الشهرية", fontsize=16, fontweight="bold")
plt.xlabel("الشهر")
plt.ylabel("المبيعات (ريال)")
plt.legend()        # إظهار الـ label الموجود أعلاه
plt.grid(True, alpha=0.3)   # شبكة خفيفة في الخلفية
plt.tight_layout()  # ترتيب تلقائي لمنع تداخل العناصر

plt.savefig("sales_chart.png", dpi=300)  # حفظ كصورة (بدلاً من/بالإضافة إلى show)
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-026-ex20_5.py
"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # صف واحد، عمودان

axes[0].plot(months, sales, marker="o")
axes[0].set_title("خطي")

axes[1].bar(months, sales, color="orange")
axes[1].set_title("شريطي")

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.5: Matplotlib: تحويل الأرقام إلى رسوم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-027-ex20_5.py
"""

import pandas as pd

df = pd.DataFrame({
    "الشهر": ["يناير", "فبراير", "مارس"],
    "المبيعات": [1200, 1900, 1500],
})

df.plot(x="الشهر", y="المبيعات", kind="bar", title="المبيعات الشهرية")
plt.show()

# أو مباشرة من نتيجة groupby
df.groupby("الشهر")["المبيعات"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-028-n_records.py
"""

"""سكربت مساعد لتوليد بيانات مبيعات تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)  # لضمان نتائج قابلة لإعادة الإنتاج

n_records = 1000

products = ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة", "سماعات", "طابعة"]
product_prices = {
    "لابتوب": 3500, "ماوس": 50, "لوحة مفاتيح": 120,
    "شاشة": 900, "سماعات": 150, "طابعة": 600,
}
cities = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]

data = {
    "order_id": range(1, n_records + 1),
    "التاريخ": pd.date_range("2024-01-01", "2024-12-31", periods=n_records),
    "المنتج": np.random.choice(products, n_records),
    "المدينة": np.random.choice(cities, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "الكمية": np.random.randint(1, 6, n_records),
}

df = pd.DataFrame(data)
df["سعر_الوحدة"] = df["المنتج"].map(product_prices)
df["الإجمالي"] = df["سعر_الوحدة"] * df["الكمية"]

# إدخال بعض القيم المفقودة عمدًا (كما يحدث في البيانات الحقيقية)
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, "المدينة"] = None

df.to_csv("data/sales_data.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n_records} سجل مبيعات في data/sales_data.csv")

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-029-load_sales_data.py
"""

"""وحدة تحميل وتنظيف بيانات المبيعات."""

from pathlib import Path

import pandas as pd

def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """تحميل بيانات المبيعات من ملف CSV.

    Args:
        file_path: مسار ملف CSV.

    Returns:
        DataFrame يحتوي بيانات المبيعات الخام.
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig", parse_dates=["التاريخ"])
    return df

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف بيانات المبيعات: معالجة القيم المفقودة والتكرارات.

    Args:
        df: DataFrame الخام.

    Returns:
        DataFrame نظيف جاهز للتحليل.
    """
    clean_df = df.copy()

    # تعويض المدن المفقودة بقيمة واضحة بدلاً من حذف الصفوف (نحتفظ بالبيانات)
    clean_df["المدينة"] = clean_df["المدينة"].fillna("غير محدد")

    # حذف أي صفوف مكررة بالكامل إن وُجدت
    clean_df = clean_df.drop_duplicates()

    # إضافة أعمدة مشتقة مفيدة للتحليل
    clean_df["الشهر"] = clean_df["التاريخ"].dt.month
    clean_df["اسم_الشهر"] = clean_df["التاريخ"].dt.month_name()
    clean_df["ربع_السنة"] = clean_df["التاريخ"].dt.quarter

    return clean_df

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-030-total_revenue.py
"""

"""وحدة تحليل بيانات المبيعات."""

import pandas as pd

def total_revenue(df: pd.DataFrame) -> float:
    """حساب إجمالي الإيرادات."""
    return float(df["الإجمالي"].sum())

def revenue_by_product(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل منتج، مرتبة تنازليًا."""
    return df.groupby("المنتج")["الإجمالي"].sum().sort_values(ascending=False)

def revenue_by_city(df: pd.DataFrame) -> pd.Series:
    """إجمالي الإيرادات لكل مدينة، مرتبة تنازليًا."""
    return df.groupby("المدينة")["الإجمالي"].sum().sort_values(ascending=False)

def monthly_trend(df: pd.DataFrame) -> pd.Series:
    """اتجاه الإيرادات الشهري عبر العام."""
    return df.groupby("الشهر")["الإجمالي"].sum().sort_index()

def top_n_products_by_quantity(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """أكثر n منتجات مبيعًا من حيث الكمية."""
    return df.groupby("المنتج")["الكمية"].sum().sort_values(ascending=False).head(n)

def average_order_value(df: pd.DataFrame) -> float:
    """متوسط قيمة الطلب الواحد."""
    return float(df["الإجمالي"].mean())

def city_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """جدول محوري: الإيرادات لكل منتج في كل مدينة."""
    return df.pivot_table(
        values="الإجمالي",
        index="المدينة",
        columns="المنتج",
        aggfunc="sum",
        fill_value=0,
    )

def generate_summary_report(df: pd.DataFrame) -> dict[str, float | int]:
    """توليد ملخص إحصائي شامل للبيانات."""
    return {
        "إجمالي_الإيرادات": total_revenue(df),
        "عدد_الطلبات": len(df),
        "متوسط_قيمة_الطلب": average_order_value(df),
        "أعلى_منتج_مبيعًا": str(revenue_by_product(df).index[0]),
        "أعلى_مدينة_مبيعًا": str(revenue_by_city(df).index[0]),
    }

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-031-_ensure_output_dir.py
"""

"""وحدة توليد الرسوم البيانية لتحليل المبيعات."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("output")

def _ensure_output_dir() -> None:
    """التأكد من وجود مجلد الإخراج."""
    OUTPUT_DIR.mkdir(exist_ok=True)

def plot_revenue_by_product(revenue: pd.Series) -> None:
    """رسم بياني شريطي لإيرادات المنتجات."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    revenue.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Revenue by Product", fontsize=14, fontweight="bold")
    plt.xlabel("Product")
    plt.ylabel("Revenue (SAR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_product.png", dpi=200)
    plt.close()

def plot_revenue_by_city_pie(revenue: pd.Series) -> None:
    """رسم دائري لتوزيع الإيرادات حسب المدينة."""
    _ensure_output_dir()
    plt.figure(figsize=(8, 8))
    plt.pie(revenue.values, labels=revenue.index, autopct="%1.1f%%", startangle=90)
    plt.title("Revenue Distribution by City", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_city.png", dpi=200)
    plt.close()

def plot_monthly_trend(trend: pd.Series) -> None:
    """رسم خطي لاتجاه المبيعات الشهري."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(trend.index, trend.values, marker="o", linewidth=2, color="darkgreen")
    plt.title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Revenue (SAR)")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_trend.png", dpi=200)
    plt.close()

def plot_city_product_heatmap(matrix: pd.DataFrame) -> None:
    """رسم Heatmap لمصفوفة المدينة/المنتج."""
    _ensure_output_dir()
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="Revenue (SAR)")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title("Revenue Heatmap: City vs Product", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "city_product_heatmap.png", dpi=200)
    plt.close()

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-032-print_report.py
"""

"""نقطة الدخول الرئيسية لتحليل بيانات المبيعات."""

from sales_analysis.analysis import (
    city_product_matrix,
    generate_summary_report,
    monthly_trend,
    revenue_by_city,
    revenue_by_product,
    top_n_products_by_quantity,
)
from sales_analysis.data_loader import clean_sales_data, load_sales_data
from sales_analysis.visualization import (
    plot_city_product_heatmap,
    plot_monthly_trend,
    plot_revenue_by_city_pie,
    plot_revenue_by_product,
)

def print_report(summary: dict) -> None:
    """طباعة تقرير نصي منسّق في الطرفية."""
    print("=" * 50)
    print(" " * 15 + "تقرير تحليل المبيعات")
    print("=" * 50)
    for key, value in summary.items():
        label = key.replace("_", " ")
        if isinstance(value, float):
            print(f"{label}: {value:,.2f}")
        else:
            print(f"{label}: {value}")
    print("=" * 50)

def main() -> None:
    """تشغيل خط التحليل الكامل."""
    print("📊 تحميل البيانات...")
    raw_df = load_sales_data("data/sales_data.csv")

    print("🧹 تنظيف البيانات...")
    df = clean_sales_data(raw_df)
    print(f"   تم تحميل {len(df)} سجل بنجاح.\n")

    print("📈 تحليل البيانات...")
    summary = generate_summary_report(df)
    print_report(summary)

    print("\n🏆 أفضل 5 منتجات من حيث الكمية المباعة:")
    print(top_n_products_by_quantity(df, 5))

    print("\n🎨 توليد الرسوم البيانية...")
    plot_revenue_by_product(revenue_by_product(df))
    plot_revenue_by_city_pie(revenue_by_city(df))
    plot_monthly_trend(monthly_trend(df))
    plot_city_product_heatmap(city_product_matrix(df))

    print("✅ تم حفظ كل الرسوم البيانية في مجلد output/")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-033-students.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
students = pd.DataFrame({
    "الاسم": ["أحمد", "سارة", "خالد", "منى", "يوسف"],
    "الرياضيات": [85, 92, 78, 95, 88],
    "العلوم": [90, 85, 82, 91, 79],
})
print(students["الرياضيات"].mean())
print(students["العلوم"].mean())

# 2
numbers = np.random.randint(1, 101, 10)
print(numbers)
print("الأكبر:", numbers.max())
print("الأصغر:", numbers.min())
print("المتوسط:", numbers.mean())

# 3
days = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء"]
hours = [2, 3, 1.5, 4, 2.5]
plt.bar(days, hours, color="purple")
plt.title("ساعات الدراسة")
plt.show()

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-034-employees.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# 4
employees = pd.DataFrame({
    "اسم": ["علي", "فاطمة", "سعد", "ليلى", "عمر"],
    "القسم": ["مبيعات", "تقنية", "مبيعات", "تقنية", "مالية"],
    "الراتب": [8000, 12000, 8500, 13000, 9500],
})
avg_salary = employees.groupby("القسم")["الراتب"].mean()
print(avg_salary)
avg_salary.plot(kind="bar", color="teal")
plt.title("متوسط الراتب حسب القسم")
plt.ylabel("الراتب")
plt.tight_layout()
plt.show()

# 5
data = pd.DataFrame({"العمر": [25, 30, None, 45, 28, None, 90]})
# الوسيط أفضل من المتوسط هنا لأن القيمة 90 (قيمة متطرفة/Outlier)
# تؤثر بشدة على المتوسط لكن تأثيرها على الوسيط محدود جدًا
data["العمر"] = data["العمر"].fillna(data["العمر"].median())
print(data)

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-035-best_selling_month_per_product.py
"""

# إضافة إلى analysis.py
def best_selling_month_per_product(df: pd.DataFrame) -> pd.Series:
    """الشهر الأفضل مبيعًا لكل منتج."""
    grouped = df.groupby(["المنتج", "الشهر"])["الإجمالي"].sum()
    return grouped.groupby(level="المنتج").idxmax()

# إضافة إلى visualization.py
def plot_quarterly_comparison(df: pd.DataFrame) -> None:
    """مقارنة مبيعات الأرباع الأربعة لكل مدينة."""
    _ensure_output_dir()
    pivot = df.pivot_table(
        values="الإجمالي", index="ربع_السنة", columns="المدينة",
        aggfunc="sum", fill_value=0,
    )
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Quarterly Revenue Comparison by City")
    plt.xlabel("Quarter")
    plt.ylabel("Revenue (SAR)")
    plt.legend(title="City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quarterly_comparison.png", dpi=200)
    plt.close()

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-036-arr.py
"""

# ===== NumPy =====
import numpy as np
arr = np.array([1, 2, 3])
arr.sum(); arr.mean(); arr.max(); arr.min(); arr.std()
filtered = arr[arr > 1]                      # Boolean Indexing

# ===== pandas: القراءة والاستكشاف =====
import pandas as pd
df = pd.read_csv("file.csv", encoding="utf-8")
df.head(); df.tail(); df.info(); df.describe()
df.shape; df.columns; df.dtypes
df.isna().sum()                              # القيم المفقودة

# ===== pandas: الاختيار والتصفية =====
df["col"]                                    # عمود -> Series
df[["col1", "col2"]]                         # أعمدة -> DataFrame
df.loc[0, "col"]                             # بالتسمية
df.iloc[0, 1]                                # بالموضع
df[(df["a"] > 5) & (df["b"] < 10)]           # تصفية بشروط (& | وليس and or)

# ===== pandas: التعديل =====
df["new_col"] = df["a"] * df["b"]
df = df.drop(columns=["col"])
df = df.rename(columns={"old": "new"})
df["col"] = df["col"].fillna(df["col"].mean())
df = df.dropna(subset=["col"])
df = df.sort_values("col", ascending=False)

# ===== pandas: التجميع والدمج =====
df.groupby("col")["val"].sum()
df.groupby("col")["val"].agg(["sum", "mean", "count"])
df.pivot_table(values="val", index="r", columns="c", aggfunc="sum")
pd.merge(df1, df2, on="key", how="left")
pd.concat([df1, df2], ignore_index=True)

# ===== pandas: التصدير =====
df.to_csv("out.csv", index=False, encoding="utf-8-sig")

# ===== Matplotlib =====
import matplotlib.pyplot as plt
plt.plot(x, y); plt.bar(x, y); plt.pie(values, labels=labels)
plt.scatter(x, y); plt.hist(data, bins=20)
plt.title(""); plt.xlabel(""); plt.ylabel(""); plt.legend()
plt.savefig("chart.png", dpi=200); plt.show(); plt.close()

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> ```

---

## 20.3 NumPy: أساس الحوسبة الرقمية

### لماذا لا نستخدم القوائم العادية؟

في الفصل 8، تعلمنا القوائم (`list`). لماذا نحتاج شيئًا آخر؟

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: Broadcasting قد يُربك المبتدئين**
>
> عندما تجري عملية بين مصفوفة وعدد مفرد (scalar)، يطبّق NumPy العملية على كل عنصر تلقائيًا — هذا يُسمى **Broadcasting**:
> ```python
> arr = np.array([1, 2, 3])
> print(arr + 10)  # [11 12 13] -- وليس خطأ!
> ```
> هذا سلوك **مقصود ومفيد جدًا**، لكن يجب أن تفهمه حتى لا يفاجئك.

> **💡 نصيحة خبير:** NumPy هو الأساس الذي تُبنى عليه pandas وscikit-learn وTensorFlow. لن تحتاج غالبًا لاستخدامه مباشرة بشكل مكثف في تحليل البيانات اليومي (ستستخدم pandas بدلاً منه)، لكن **فهمه ضروري** لأن pandas مبني عليه بالكامل، وستقابله باستمرار.

---

## 20.4 pandas: العمود الفقري لتحليل البيانات

إذا كان NumPy يتعامل مع أرقام في مصفوفات، فإن **pandas** يتعامل مع بيانات **جدولية** حقيقية: صفوف وأعمدة لها أسماء، أنواع بيانات مختلفة (نصوص، أرقام، تواريخ)، وقيم مفقودة — تمامًا كملف Excel أو جدول في قاعدة بيانات.

### هيكلا البيانات الأساسيان

**1. `Series`**: عمود واحد من البيانات (كـ«قائمة» لكن أذكى، بفهارس مسمّاة).

# ────────────────────────────────────────────────────────────────────────
# [072] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: دعم اللغة العربية في matplotlib**
> matplotlib لا يدعم اتجاه الكتابة من اليمين لليسار (RTL) أو ربط الحروف العربية بشكل جيد افتراضيًا — قد تظهر الحروف منفصلة أو معكوسة. للحصول على نص عربي صحيح، استخدم مكتبة `arabic-reshaper` مع `python-bidi`:
> ```python
> import arabic_reshaper
> from bidi.algorithm import get_display
>
> def fix_arabic(text: str) -> str:
>     reshaped = arabic_reshaper.reshape(text)
>     return get_display(reshaped)
>
> plt.title(fix_arabic(«المبيعات الشهرية»))
> ```
> في الأمثلة التالية سنستخدم أحيانًا تسميات إنجليزية للتبسيط، لكن اعتمد هذا الحل في مشاريعك الحقيقية عند الحاجة لنصوص عربية في الرسوم.

### أنواع الرسوم الشائعة

# ────────────────────────────────────────────────────────────────────────
# [073] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
> **جدول مرجعي: أي رسم تختار؟**
>
> | نوع البيانات | السؤال | الرسم المناسب |
> |---|---|---|
> | تطور عبر الزمن | هل المبيعات تزداد؟ | `plt.plot()` (خطي) |
> | مقارنة فئات | أي منتج الأكثر مبيعًا؟ | `plt.bar()` (شريطي) |
> | نسب من كل | ما حصة كل قسم من الميزانية؟ | `plt.pie()` (دائري) |
> | علاقة بين متغيرين | هل السعر يؤثر على الكمية المباعة؟ | `plt.scatter()` (تشتت) |
> | توزيع قيم | كيف تتوزع أعمار العملاء؟ | `plt.hist()` (مدرج تكراري) |

### تخصيص الرسوم

# ────────────────────────────────────────────────────────────────────────
# [074] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
> **💡 نصيحة خبير:** استخدمنا `plt.close()` بعد كل `savefig()` بدلاً من `plt.show()`. هذا ضروري في السكربتات التي تنتج عدة رسوم متتالية، لأنه يُحرر الذاكرة ويمنع تراكب الرسوم فوق بعضها بالخطأ.

### الخطوة 5: `main.py` — تجميع كل شيء

# ────────────────────────────────────────────────────────────────────────
# [075] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-20-chapter-20.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
</details>

---

## 20.9 اختبر فهمك (Quiz)

<details>
<summary>1. ما الفرق الجوهري بين `list` العادية و `numpy.ndarray`؟</summary>

`ndarray` تتطلب أن تكون كل عناصرها من نفس النوع (موحدة)، وتدعم العمليات الحسابية مباشرة على كل العناصر دفعة واحدة (Vectorization) دون حلقات، وهي أسرع بكثير وأقل استهلاكًا للذاكرة لأنها منفذة بلغة C داخليًا. القوائم العادية مرنة (تقبل أنواعًا مختلطة) لكنها أبطأ في الحسابات الرقمية.
</details>

<details>
<summary>2. لماذا تفشل `df[(df["a"] > 5) and (df["b"] < 10)]` وما البديل الصحيح؟</summary>

لأن `and`/`or` في Python تتوقع قيمة Boolean مفردة، بينما الشرط `(df["a"] > 5)` يُرجع `Series` كاملة من قيم True/False وليس قيمة واحدة. البديل الصحيح هو استخدام `&` و`|` مع وضع أقواس حول كل شرط فرعي: `df[(df["a"] > 5) & (df["b"] < 10)]`.
</details>

<details>
<summary>3. متى تستخدم `loc` ومتى تستخدم `iloc`؟</summary>

`loc` يُستخدم للاختيار بناءً على **التسمية/الاسم** (label-based)، مثل اسم العمود أو قيمة الفهرس الفعلية. `iloc` يُستخدم للاختيار بناءً على **الموضع الرقمي** (position-based) تمامًا كالقوائم العادية (0، 1، 2...)، بغض النظر عن تسميات الفهرس.
</details>

<details>
<summary>4. ما وظيفة `groupby` ومتى تستخدمها؟</summary>

`groupby` يُستخدم لتقسيم البيانات إلى مجموعات بناءً على قيمة عمود (أو أكثر)، ثم تطبيق دالة تجميعية (sum, mean, count...) على كل مجموعة على حدة. يُستخدم للإجابة على أسئلة مثل «ما إجمالي المبيعات لكل مدينة؟» وهو مكافئ لـ `GROUP BY` في SQL.
</details>

<details>
<summary>5. لماذا نستخدم `.copy()` أحيانًا بعد تصفية DataFrame؟</summary>

لتجنب تحذير `SettingWithCopyWarning` والسلوك غير المتوقع الناتج عن أن الجزء المُصفّى قد يكون «عرضًا» (view) مرتبطًا بالبيانات الأصلية بدلاً من نسخة مستقلة. استخدام `.copy()` يضمن أننا نعمل على نسخة منفصلة تمامًا وآمنة للتعديل.
</details>

<details>
<summary>6. ما الفرق بين `plt.show()` و `plt.savefig()`؟</summary>

`plt.show()` يفتح نافذة تفاعلية لعرض الرسم على الشاشة مباشرة، بينما `plt.savefig()` يحفظ الرسم كملف صورة (PNG, JPG, PDF...) على القرص دون بالضرورة عرضه، وهذا مفيد جدًا في السكربتات الآلية التي تولّد عدة رسوم دون تدخل بشري.
</details>

---

## 20.10 ملخص الفصل (Cheat Sheet)
