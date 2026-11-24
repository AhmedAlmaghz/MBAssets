#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كتابة كود نظيف (Clean Code)
Python 3.14 — From Zero to Pro — Clean

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 44 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-24-chapter-24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — كتابة كود نظيف (Clean Code)
Python 3.14 — From Zero to Pro — Clean

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 43 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["black", "ruff"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q black ruff

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# قبل black (ملف messy.py)
def   calculate(x,y) :
    result=x+y
    if result>10:
        return   'كبير'
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# بعد black -- تلقائيًا!
def calculate(x, y):
    result = x + y
    if result > 10:
        return "كبير"
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# problematic.py
import os  # مستورد لكن غير مستخدم
import sys

def process_data(data):
    unused_variable = 42  # معرّف لكن غير مستخدم
    result = []
    for item in data:
        if item == None:  # يجب استخدام "is None" وليس "== None"
            continue
        result.append(item)
    return result

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

# ✅ يتبع OCP: إضافة طريقة دفع جديدة لا تتطلب تعديل أي كود قديم
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...

class CreditCardPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر بطاقة الائتمان"

class PayPalPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر PayPal"

# لإضافة طريقة دفع جديدة، أضف صنفًا جديدًا فقط -- بدون لمس الكود القديم!
class CryptoPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر العملات الرقمية"

def checkout(method: PaymentMethod, amount: float) -> None:
    print(method.pay(amount))

checkout(CreditCardPayment(), 100)
checkout(CryptoPayment(), 50)

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-24.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ===== قبل: كود يعمل لكنه غير نظيف =====
def proc(l):
    r = []
    for i in l:
        if i['type'] == 'A':
            if i['amt'] > 100:
                r.append({'id': i['id'], 'final': i['amt'] * 0.9})
            else:
                r.append({'id': i['id'], 'final': i['amt']})
        elif i['type'] == 'B':
            r.append({'id': i['id'], 'final': i['amt'] * 0.95})
    return r

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-24.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ===== بعد: نظيف، واضح، وقابل للصيانة =====

TYPE_A = "A"
TYPE_B = "B"

TYPE_A_BULK_THRESHOLD = 100
TYPE_A_BULK_DISCOUNT = 0.9
TYPE_B_STANDARD_DISCOUNT = 0.95

def calculate_type_a_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع A (خصم إضافي للكميات الكبيرة)."""
    if amount > TYPE_A_BULK_THRESHOLD:
        return amount * TYPE_A_BULK_DISCOUNT
    return amount

def calculate_type_b_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع B (خصم ثابت)."""
    return amount * TYPE_B_STANDARD_DISCOUNT

def process_items(items: list[dict]) -> list[dict]:
    """معالجة قائمة عناصر وحساب السعر النهائي لكل منها حسب نوعه.

    Args:
        items: قائمة قواميس، كل منها يحتوي 'id', 'type', 'amt'.

    Returns:
        قائمة قواميس تحتوي 'id' و'final' (السعر النهائي بعد الخصم).
    """
    processed_items = []

    for item in items:
        if item["type"] == TYPE_A:
            final_price = calculate_type_a_discount(item["amt"])
        elif item["type"] == TYPE_B:
            final_price = calculate_type_b_discount(item["amt"])
        else:
            continue  # تخطَّ الأنواع غير المعروفة بدلاً من كسر البرنامج

        processed_items.append({"id": item["id"], "final": final_price})

    return processed_items

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-24.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def calculate_shipping(weight):
    if weight > 20:
        return weight * 5.5
    return weight * 3.2

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-24.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 2
# d -> user_data
# tmp -> temporary_file_path
# lst2 -> filtered_products
# flag1 -> is_user_verified

# 3
HEAVY_PACKAGE_THRESHOLD_KG = 20
HEAVY_PACKAGE_RATE = 5.5
STANDARD_RATE = 3.2

def calculate_shipping(weight: float) -> float:
    if weight > HEAVY_PACKAGE_THRESHOLD_KG:
        return weight * HEAVY_PACKAGE_RATE
    return weight * STANDARD_RATE

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-24.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def handle_user_signup(email, password):
    if "@" not in email:
        raise ValueError("بريد غير صالح")
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")
    hashed = password[::-1]  # تشفير وهمي للتبسيط
    print(f"حفظ المستخدم: {email}")
    print(f"إرسال بريد ترحيبي إلى: {email}")
    return {"email": email, "password": hashed}

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-24.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def get_shipping_cost(order):
    if order is not None:
        if order.weight > 0:
            if order.destination == "local":
                return order.weight * 2
            else:
                return order.weight * 5
    return 0

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-24.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# 4
def validate_email(email: str) -> None:
    if "@" not in email:
        raise ValueError("بريد غير صالح")

def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")

def hash_password(password: str) -> str:
    return password[::-1]  # تشفير وهمي للتبسيط

def save_user(email: str, hashed_password: str) -> None:
    print(f"حفظ المستخدم: {email}")

def send_welcome_email(email: str) -> None:
    print(f"إرسال بريد ترحيبي إلى: {email}")

def handle_user_signup(email: str, password: str) -> dict:
    validate_email(email)
    validate_password(password)
    hashed = hash_password(password)
    save_user(email, hashed)
    send_welcome_email(email)
    return {"email": email, "password": hashed}

# 5
def get_shipping_cost(order) -> float:
    if order is None:
        return 0
    if order.weight <= 0:
        return 0
    if order.destination == "local":
        return order.weight * 2
    return order.weight * 5

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-24.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class NotificationChannel(ABC):
    """تجريد (Abstraction) لأي قناة إشعار."""

    @abstractmethod
    def send(self, message: str, recipient: str) -> None:
        ...

class EmailNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📧 إرسال بريد إلى {recipient}: {message}")

class SMSNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📱 إرسال رسالة نصية إلى {recipient}: {message}")

# إضافة قناة جديدة -- بدون أي تعديل على الكود القديم (مبدأ Open/Closed)
class PushNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"🔔 إرسال إشعار دفع إلى {recipient}: {message}")

class NotificationService:
    """تعتمد على التجريد فقط -- مبدأ Dependency Inversion."""

    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels

    def notify_all(self, message: str, recipient: str) -> None:
        for channel in self.channels:
            channel.send(message, recipient)

# الاستخدام
service = NotificationService(
    channels=[EmailNotification(), SMSNotification(), PushNotification()]
)
service.notify_all("طلبك في الطريق!", "ahmad@example.com")

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-24.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ===== قواعد تسمية PEP 8 =====
# متغيرات/دوال   -> snake_case
# ثوابت          -> UPPER_SNAKE_CASE
# أصناف          -> PascalCase
# خاص            -> _single_underscore / __double_underscore

# ────────────────────────────────────────────────────────────────────────
# [015] code/chapters/ch24/24-001-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-001-greet.py
"""

# ✅ صحيح: مسافة واحدة حول عوامل التشغيل
total = price + tax

# ❌ خطأ: بدون مسافات، أو مسافات زائدة
total=price+tax
total  =  price  +  tax

# ✅ صحيح: بدون مسافة قبل الفاصلة، مسافة بعدها
def greet(name, age):
    pass

# ✅ صحيح: بدون مسافات حول = في المعاملات الافتراضية (دون type hint)
def greet(name="أحمد"):
    pass

# ✅ لكن مع type hint، نضيف مسافات حول =
def greet(name: str = "أحمد"):
    pass

# ────────────────────────────────────────────────────────────────────────
# [016] code/chapters/ch24/24-002-result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-002-result.py
"""

# ❌ سطر طويل جدًا، يصعب قراءته دفعة واحدة
result = calculate_total_price(unit_price, quantity, discount_percentage, tax_rate, shipping_cost)

# ✅ مقسّم بشكل مقروء
result = calculate_total_price(
    unit_price,
    quantity,
    discount_percentage,
    tax_rate,
    shipping_cost,
)

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch24/24-003-ex24_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-003-ex24_2.py
"""

# ✅ الترتيب الموصى به: مكتبات قياسية، ثم مكتبات خارجية، ثم وحدات المشروع
import json
import os
from pathlib import Path

import pandas as pd
import requests

from my_project.models import User
from my_project.utils import format_date

# ❌ تجنّب الاستيراد الشامل -- يُخفي مصدر الأسماء ويُسبب تعارضات
from os import *

# ✅ استورد ما تحتاجه فقط بوضوح
from os import getcwd, path

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch24/24-004-calculate.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.3: التنسيق الآلي: `black`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-004-calculate.py
"""

# قبل black (ملف messy.py)
def   calculate(x,y) :
    result=x+y
    if result>10:
        return   'كبير'
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch24/24-005-calculate.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.3: التنسيق الآلي: `black`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-005-calculate.py
"""

# بعد black -- تلقائيًا!
def calculate(x, y):
    result = x + y
    if result > 10:
        return "كبير"
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch24/24-006-process_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.4: الفحص الآلي للأخطاء والجودة: `ruff`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-006-process_data.py
"""

# problematic.py
import os  # مستورد لكن غير مستخدم
import sys

def process_data(data):
    unused_variable = 42  # معرّف لكن غير مستخدم
    result = []
    for item in data:
        if item == None:  # يجب استخدام "is None" وليس "== None"
            continue
        result.append(item)
    return result

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch24/24-007-process_order.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-007-process_order.py
"""

# ❌ سيء: دالة واحدة تفعل أشياء كثيرة جدًا
def process_order(order_data):
    # التحقق من الصحة
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر")

    # حساب الإجمالي
    total = sum(item["price"] * item["qty"] for item in order_data["items"])

    # تطبيق الخصم
    if order_data.get("coupon"):
        total *= 0.9

    # حفظ في قاعدة البيانات
    print(f"حفظ الطلب بإجمالي {total}")

    # إرسال بريد إلكتروني
    print("إرسال تأكيد عبر البريد...")

    return total

# ✅ جيد: كل دالة مسؤولة عن شيء واحد فقط
def validate_order(order_data: dict) -> None:
    """التحقق من صحة بيانات الطلب."""
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر في الطلب")

def calculate_total(items: list[dict]) -> float:
    """حساب إجمالي سعر العناصر."""
    return sum(item["price"] * item["qty"] for item in items)

def apply_discount(total: float, has_coupon: bool) -> float:
    """تطبيق خصم 10% إذا وُجد كوبون."""
    return total * 0.9 if has_coupon else total

def save_order(total: float) -> None:
    """حفظ الطلب في قاعدة البيانات."""
    print(f"حفظ الطلب بإجمالي {total}")

def send_confirmation_email() -> None:
    """إرسال بريد تأكيد الطلب."""
    print("إرسال تأكيد عبر البريد...")

def process_order(order_data: dict) -> float:
    """تنسيق عملية معالجة الطلب الكاملة."""
    validate_order(order_data)
    total = calculate_total(order_data["items"])
    total = apply_discount(total, bool(order_data.get("coupon")))
    save_order(total)
    send_confirmation_email()
    return total

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch24/24-008-calculate_price.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-008-calculate_price.py
"""

# ❌ سيء: ما معنى 0.15؟ ولماذا 3؟
def calculate_price(amount):
    if amount > 1000:
        return amount * 0.85
    return amount * 0.95

for attempt in range(3):
    try_connection()

# ✅ جيد: الثوابت المُسمّاة تشرح نفسها
BULK_DISCOUNT_THRESHOLD = 1000
BULK_DISCOUNT_RATE = 0.85
STANDARD_DISCOUNT_RATE = 0.95
MAX_CONNECTION_ATTEMPTS = 3

def calculate_price(amount: float) -> float:
    if amount > BULK_DISCOUNT_THRESHOLD:
        return amount * BULK_DISCOUNT_RATE
    return amount * STANDARD_DISCOUNT_RATE

for attempt in range(MAX_CONNECTION_ATTEMPTS):
    try_connection()

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch24/24-009-get_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-009-get_discount.py
"""

# ❌ سيء: تداخل عميق (Nested Ifs) صعب التتبع
def get_discount(user):
    if user is not None:
        if user.is_active:
            if user.membership == "premium":
                return 0.2
            else:
                return 0.1
        else:
            return 0
    else:
        return 0

# ✅ جيد: Guard Clauses -- خروج مبكر، أقل تداخلًا
def get_discount(user) -> float:
    if user is None:
        return 0
    if not user.is_active:
        return 0
    if user.membership == "premium":
        return 0.2
    return 0.1

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch24/24-010-ex24_5.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-010-ex24_5.py
"""

# ❌ تعليق زائد لا يُضيف أي معلومة
x = x + 1  # إضافة 1 إلى x

# ❌ تعليق يشرح "ماذا" بدلاً من "لماذا" (الكود نفسه يوضّح "ماذا")
# نحسب السعر الإجمالي بضرب السعر في الكمية
total = price * quantity

# ✅ تعليق مفيد يشرح "لماذا" -- معلومة لا يمكن استنتاجها من الكود نفسه
# نستخدم هامش أمان 5% لتعويض تقلبات أسعار الصرف اليومية
total = price * quantity * 1.05

# ✅ تعليق يشرح قرارًا غير بديهي
# ملاحظة: نستخدم format التاريخ القديم هنا للتوافق مع النظام القديم (legacy system)
date_string = date.strftime("%d/%m/%Y")

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch24/24-011-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-011-__init__.py
"""

# ❌ ينتهك SRP: صنف واحد يفعل أشياء كثيرة غير مرتبطة
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

    def save_to_file(self, path: str) -> None:  # مسؤولية مختلفة: التخزين!
        with open(path, "w") as f:
            f.write(str(self.data))

    def send_email(self, address: str) -> None:  # مسؤولية مختلفة: الاتصال!
        print(f"إرسال التقرير إلى {address}")

# ✅ يتبع SRP: كل صنف مسؤول عن شيء واحد
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

class ReportStorage:
    @staticmethod
    def save_to_file(report: Report, path: str) -> None:
        with open(path, "w") as f:
            f.write(str(report.data))

class ReportMailer:
    @staticmethod
    def send_email(report: Report, address: str) -> None:
        print(f"إرسال التقرير إلى {address}")

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch24/24-012-pay.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-012-pay.py
"""

from abc import ABC, abstractmethod

# ✅ يتبع OCP: إضافة طريقة دفع جديدة لا تتطلب تعديل أي كود قديم
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...

class CreditCardPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر بطاقة الائتمان"

class PayPalPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر PayPal"

# لإضافة طريقة دفع جديدة، أضف صنفًا جديدًا فقط -- بدون لمس الكود القديم!
class CryptoPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر العملات الرقمية"

def checkout(method: PaymentMethod, amount: float) -> None:
    print(method.pay(amount))

checkout(CreditCardPayment(), 100)
checkout(CryptoPayment(), 50)

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch24/24-013-fly.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-013-fly.py
"""

# ❌ ينتهك LSP: الصنف الفرعي يُغيّر السلوك المتوقع بشكل مفاجئ
class Bird:
    def fly(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("البطريق لا يطير!")  # مفاجأة سيئة لأي كود يتوقع fly()

# ✅ يتبع LSP: إعادة تصميم الهرمية لتعكس الواقع بدقة
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def move(self) -> str:
        return "أسبح في الماء"  # سلوك صحيح ومتوقع، بدون كسر توقعات المستخدم

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch24/24-014-work.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-014-work.py
"""

# ❌ ينتهك ISP: واجهة ضخمة تُجبر كل شيء على تطبيق كل شيء
class Worker(ABC):
    @abstractmethod
    def work(self) -> str: ...
    @abstractmethod
    def eat(self) -> str: ...

class RobotWorker(Worker):
    def work(self) -> str:
        return "أعمل 24 ساعة"

    def eat(self) -> str:
        raise NotImplementedError("الروبوت لا يأكل!")  # إجبار غير منطقي

# ✅ يتبع ISP: واجهات صغيرة ومتخصصة، يطبّق كل صنف ما يحتاجه فقط
class Workable(ABC):
    @abstractmethod
    def work(self) -> str: ...

class Eatable(ABC):
    @abstractmethod
    def eat(self) -> str: ...

class HumanWorker(Workable, Eatable):
    def work(self) -> str:
        return "أعمل 8 ساعات"

    def eat(self) -> str:
        return "أتناول الغداء"

class RobotWorker(Workable):  # لا حاجة لتطبيق eat() أبدًا
    def work(self) -> str:
        return "أعمل 24 ساعة"

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch24/24-015-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-015-__init__.py
"""

# ❌ ينتهك DIP: الصنف مرتبط مباشرة بتفصيل محدد (MySQL)
class UserRepository:
    def __init__(self):
        self.db = MySQLDatabase()  # اعتماد مباشر وصلب على تفصيل محدد

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# ✅ يتبع DIP: الاعتماد على تجريد (Protocol كما تعلمنا في الفصل 15)
from typing import Protocol

class Database(Protocol):
    def query(self, sql: str) -> list: ...

class UserRepository:
    def __init__(self, db: Database):  # نعتمد على أي شيء يطابق الواجهة
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# الآن يمكن تمرير MySQL أو PostgreSQL أو حتى قاعدة بيانات وهمية للاختبار
repo = UserRepository(db=MySQLDatabase())
repo_test = UserRepository(db=FakeDatabaseForTesting())

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch24/24-016-calculate_circle_area.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-016-calculate_circle_area.py
"""

# ❌ تكرار -- نفس المنطق مكرر 3 مرات
def calculate_circle_area(radius):
    return 3.14159 * radius ** 2

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

def calculate_sphere_volume(radius):
    return (4/3) * 3.14159 * radius ** 3

# ✅ استخراج الثابت المشترك
import math  # استخدام math.pi بدلاً من كتابة الرقم يدويًا في كل مكان

def calculate_circle_area(radius: float) -> float:
    return math.pi * radius ** 2

def calculate_circle_circumference(radius: float) -> float:
    return 2 * math.pi * radius

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch24/24-017-is_even.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-017-is_even.py
"""

# ❌ معقّد دون داعٍ
def is_even(n: int) -> bool:
    return True if n % 2 == 0 else False

# ✅ بسيط ومباشر
def is_even(n: int) -> bool:
    return n % 2 == 0

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch24/24-018-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-018-__init__.py
"""

# ❌ مبالغة: نظام إعدادات معقّد لميزة بسيطة قد لا تُستخدم أبدًا
class ConfigurableGreeter:
    def __init__(self, language="ar", formal=False, emoji_support=True,
                 max_length=100, custom_templates=None, plugins=None):
        # ... 20 سطرًا من المرونة غير المستخدمة فعليًا

# ✅ ابدأ بسيطًا، وأضف المرونة فقط عند الحاجة الفعلية
def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch24/24-019-proc.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.8: نموذج عملي: قبل وبعد (Refactoring كامل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-019-proc.py
"""

# ===== قبل: كود يعمل لكنه غير نظيف =====
def proc(l):
    r = []
    for i in l:
        if i['type'] == 'A':
            if i['amt'] > 100:
                r.append({'id': i['id'], 'final': i['amt'] * 0.9})
            else:
                r.append({'id': i['id'], 'final': i['amt']})
        elif i['type'] == 'B':
            r.append({'id': i['id'], 'final': i['amt'] * 0.95})
    return r

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch24/24-020-calculate_type_a_discount.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.8: نموذج عملي: قبل وبعد (Refactoring كامل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-020-calculate_type_a_discount.py
"""

# ===== بعد: نظيف، واضح، وقابل للصيانة =====

TYPE_A = "A"
TYPE_B = "B"

TYPE_A_BULK_THRESHOLD = 100
TYPE_A_BULK_DISCOUNT = 0.9
TYPE_B_STANDARD_DISCOUNT = 0.95

def calculate_type_a_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع A (خصم إضافي للكميات الكبيرة)."""
    if amount > TYPE_A_BULK_THRESHOLD:
        return amount * TYPE_A_BULK_DISCOUNT
    return amount

def calculate_type_b_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع B (خصم ثابت)."""
    return amount * TYPE_B_STANDARD_DISCOUNT

def process_items(items: list[dict]) -> list[dict]:
    """معالجة قائمة عناصر وحساب السعر النهائي لكل منها حسب نوعه.

    Args:
        items: قائمة قواميس، كل منها يحتوي 'id', 'type', 'amt'.

    Returns:
        قائمة قواميس تحتوي 'id' و'final' (السعر النهائي بعد الخصم).
    """
    processed_items = []

    for item in items:
        if item["type"] == TYPE_A:
            final_price = calculate_type_a_discount(item["amt"])
        elif item["type"] == TYPE_B:
            final_price = calculate_type_b_discount(item["amt"])
        else:
            continue  # تخطَّ الأنواع غير المعروفة بدلاً من كسر البرنامج

        processed_items.append({"id": item["id"], "final": final_price})

    return processed_items

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch24/24-021-calculate_shipping.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-021-calculate_shipping.py
"""

def calculate_shipping(weight):
    if weight > 20:
        return weight * 5.5
    return weight * 3.2

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch24/24-022-calculate_shipping.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-022-calculate_shipping.py
"""

# 2
# d -> user_data
# tmp -> temporary_file_path
# lst2 -> filtered_products
# flag1 -> is_user_verified

# 3
HEAVY_PACKAGE_THRESHOLD_KG = 20
HEAVY_PACKAGE_RATE = 5.5
STANDARD_RATE = 3.2

def calculate_shipping(weight: float) -> float:
    if weight > HEAVY_PACKAGE_THRESHOLD_KG:
        return weight * HEAVY_PACKAGE_RATE
    return weight * STANDARD_RATE

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch24/24-023-handle_user_signup.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-023-handle_user_signup.py
"""

def handle_user_signup(email, password):
    if "@" not in email:
        raise ValueError("بريد غير صالح")
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")
    hashed = password[::-1]  # تشفير وهمي للتبسيط
    print(f"حفظ المستخدم: {email}")
    print(f"إرسال بريد ترحيبي إلى: {email}")
    return {"email": email, "password": hashed}

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch24/24-024-get_shipping_cost.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-024-get_shipping_cost.py
"""

def get_shipping_cost(order):
    if order is not None:
        if order.weight > 0:
            if order.destination == "local":
                return order.weight * 2
            else:
                return order.weight * 5
    return 0

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch24/24-025-validate_email.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-025-validate_email.py
"""

# 4
def validate_email(email: str) -> None:
    if "@" not in email:
        raise ValueError("بريد غير صالح")

def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")

def hash_password(password: str) -> str:
    return password[::-1]  # تشفير وهمي للتبسيط

def save_user(email: str, hashed_password: str) -> None:
    print(f"حفظ المستخدم: {email}")

def send_welcome_email(email: str) -> None:
    print(f"إرسال بريد ترحيبي إلى: {email}")

def handle_user_signup(email: str, password: str) -> dict:
    validate_email(email)
    validate_password(password)
    hashed = hash_password(password)
    save_user(email, hashed)
    send_welcome_email(email)
    return {"email": email, "password": hashed}

# 5
def get_shipping_cost(order) -> float:
    if order is None:
        return 0
    if order.weight <= 0:
        return 0
    if order.destination == "local":
        return order.weight * 2
    return order.weight * 5

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch24/24-026-send.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-026-send.py
"""

from abc import ABC, abstractmethod

class NotificationChannel(ABC):
    """تجريد (Abstraction) لأي قناة إشعار."""

    @abstractmethod
    def send(self, message: str, recipient: str) -> None:
        ...

class EmailNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📧 إرسال بريد إلى {recipient}: {message}")

class SMSNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📱 إرسال رسالة نصية إلى {recipient}: {message}")

# إضافة قناة جديدة -- بدون أي تعديل على الكود القديم (مبدأ Open/Closed)
class PushNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"🔔 إرسال إشعار دفع إلى {recipient}: {message}")

class NotificationService:
    """تعتمد على التجريد فقط -- مبدأ Dependency Inversion."""

    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels

    def notify_all(self, message: str, recipient: str) -> None:
        for channel in self.channels:
            channel.send(message, recipient)

# الاستخدام
service = NotificationService(
    channels=[EmailNotification(), SMSNotification(), PushNotification()]
)
service.notify_all("طلبك في الطريق!", "ahmad@example.com")

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch24/24-027-ex24_11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.11: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-027-ex24_11.py
"""

# ===== قواعد تسمية PEP 8 =====
# متغيرات/دوال   -> snake_case
# ثوابت          -> UPPER_SNAKE_CASE
# أصناف          -> PascalCase
# خاص            -> _single_underscore / __double_underscore

# ────────────────────────────────────────────────────────────────────────
# [042] dist/markdown/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
> # ❌ سيء جدًا -- لا يخبرنا بشيء
> def calc(x, y, z):
>     return x * y - z
>
> # ✅ ممتاز -- الاسم نفسه يشرح الغرض
> def calculate_net_price(unit_price: float, quantity: int, discount: float) -> float:
>     return unit_price * quantity - discount
> ```
> الكود الجيد **يُقرأ كالنثر**. إذا احتجت تعليقًا لشرح ماذا تفعل دالة أو متغيّر، فكّر أولًا: هل يمكن أن يشرح الاسم نفسه هذا بدلاً من التعليق؟

### المسافات والتباعد

# ────────────────────────────────────────────────────────────────────────
# [043] dist/markdown/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: الذكاء المفرط عدو الوضوح**
> ```python
> # ❌ «ذكي» جدًا لكن غير مقروء
> result = [x for x in range(100) if not any(x % i == 0 for i in range(2, int(x**0.5)+1)) and x > 1]
>
> # ✅ أوضح، حتى لو أطول قليلاً
> def is_prime(n: int) -> bool:
>     if n < 2:
>         return False
>     return all(n % i != 0 for i in range(2, int(n**0.5) + 1))
>
> result = [x for x in range(100) if is_prime(x)]
> ```
> الكود «الذكي جدًا» الذي يتباهى بحل كل شيء في سطر واحد غالبًا ما يكون عبئًا على كل من يقرأه لاحقًا — بما فيهم أنت. **الوضوح دائمًا أهم من الإيجاز.**

### YAGNI: لن تحتاجه (You Aren't Gonna Need It)

لا تبني مرونة أو ميزات «قد تحتاجها مستقبلاً» دون حاجة فعلية حالية. هذا يُعقّد الكود دون فائدة حقيقية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["black", "ruff"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q black ruff

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# قبل black (ملف messy.py)
def   calculate(x,y) :
    result=x+y
    if result>10:
        return   'كبير'
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# بعد black -- تلقائيًا!
def calculate(x, y):
    result = x + y
    if result > 10:
        return "كبير"
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# problematic.py
import os  # مستورد لكن غير مستخدم
import sys

def process_data(data):
    unused_variable = 42  # معرّف لكن غير مستخدم
    result = []
    for item in data:
        if item == None:  # يجب استخدام "is None" وليس "== None"
            continue
        result.append(item)
    return result

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

# ✅ يتبع OCP: إضافة طريقة دفع جديدة لا تتطلب تعديل أي كود قديم
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...

class CreditCardPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر بطاقة الائتمان"

class PayPalPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر PayPal"

# لإضافة طريقة دفع جديدة، أضف صنفًا جديدًا فقط -- بدون لمس الكود القديم!
class CryptoPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر العملات الرقمية"

def checkout(method: PaymentMethod, amount: float) -> None:
    print(method.pay(amount))

checkout(CreditCardPayment(), 100)
checkout(CryptoPayment(), 50)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ===== قبل: كود يعمل لكنه غير نظيف =====
def proc(l):
    r = []
    for i in l:
        if i['type'] == 'A':
            if i['amt'] > 100:
                r.append({'id': i['id'], 'final': i['amt'] * 0.9})
            else:
                r.append({'id': i['id'], 'final': i['amt']})
        elif i['type'] == 'B':
            r.append({'id': i['id'], 'final': i['amt'] * 0.95})
    return r

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ===== بعد: نظيف، واضح، وقابل للصيانة =====

TYPE_A = "A"
TYPE_B = "B"

TYPE_A_BULK_THRESHOLD = 100
TYPE_A_BULK_DISCOUNT = 0.9
TYPE_B_STANDARD_DISCOUNT = 0.95

def calculate_type_a_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع A (خصم إضافي للكميات الكبيرة)."""
    if amount > TYPE_A_BULK_THRESHOLD:
        return amount * TYPE_A_BULK_DISCOUNT
    return amount

def calculate_type_b_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع B (خصم ثابت)."""
    return amount * TYPE_B_STANDARD_DISCOUNT

def process_items(items: list[dict]) -> list[dict]:
    """معالجة قائمة عناصر وحساب السعر النهائي لكل منها حسب نوعه.

    Args:
        items: قائمة قواميس، كل منها يحتوي 'id', 'type', 'amt'.

    Returns:
        قائمة قواميس تحتوي 'id' و'final' (السعر النهائي بعد الخصم).
    """
    processed_items = []

    for item in items:
        if item["type"] == TYPE_A:
            final_price = calculate_type_a_discount(item["amt"])
        elif item["type"] == TYPE_B:
            final_price = calculate_type_b_discount(item["amt"])
        else:
            continue  # تخطَّ الأنواع غير المعروفة بدلاً من كسر البرنامج

        processed_items.append({"id": item["id"], "final": final_price})

    return processed_items

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def calculate_shipping(weight):
    if weight > 20:
        return weight * 5.5
    return weight * 3.2

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 2
# d -> user_data
# tmp -> temporary_file_path
# lst2 -> filtered_products
# flag1 -> is_user_verified

# 3
HEAVY_PACKAGE_THRESHOLD_KG = 20
HEAVY_PACKAGE_RATE = 5.5
STANDARD_RATE = 3.2

def calculate_shipping(weight: float) -> float:
    if weight > HEAVY_PACKAGE_THRESHOLD_KG:
        return weight * HEAVY_PACKAGE_RATE
    return weight * STANDARD_RATE

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def handle_user_signup(email, password):
    if "@" not in email:
        raise ValueError("بريد غير صالح")
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")
    hashed = password[::-1]  # تشفير وهمي للتبسيط
    print(f"حفظ المستخدم: {email}")
    print(f"إرسال بريد ترحيبي إلى: {email}")
    return {"email": email, "password": hashed}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def get_shipping_cost(order):
    if order is not None:
        if order.weight > 0:
            if order.destination == "local":
                return order.weight * 2
            else:
                return order.weight * 5
    return 0

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# 4
def validate_email(email: str) -> None:
    if "@" not in email:
        raise ValueError("بريد غير صالح")

def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")

def hash_password(password: str) -> str:
    return password[::-1]  # تشفير وهمي للتبسيط

def save_user(email: str, hashed_password: str) -> None:
    print(f"حفظ المستخدم: {email}")

def send_welcome_email(email: str) -> None:
    print(f"إرسال بريد ترحيبي إلى: {email}")

def handle_user_signup(email: str, password: str) -> dict:
    validate_email(email)
    validate_password(password)
    hashed = hash_password(password)
    save_user(email, hashed)
    send_welcome_email(email)
    return {"email": email, "password": hashed}

# 5
def get_shipping_cost(order) -> float:
    if order is None:
        return 0
    if order.weight <= 0:
        return 0
    if order.destination == "local":
        return order.weight * 2
    return order.weight * 5

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod

class NotificationChannel(ABC):
    """تجريد (Abstraction) لأي قناة إشعار."""

    @abstractmethod
    def send(self, message: str, recipient: str) -> None:
        ...

class EmailNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📧 إرسال بريد إلى {recipient}: {message}")

class SMSNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📱 إرسال رسالة نصية إلى {recipient}: {message}")

# إضافة قناة جديدة -- بدون أي تعديل على الكود القديم (مبدأ Open/Closed)
class PushNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"🔔 إرسال إشعار دفع إلى {recipient}: {message}")

class NotificationService:
    """تعتمد على التجريد فقط -- مبدأ Dependency Inversion."""

    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels

    def notify_all(self, message: str, recipient: str) -> None:
        for channel in self.channels:
            channel.send(message, recipient)

# الاستخدام
service = NotificationService(
    channels=[EmailNotification(), SMSNotification(), PushNotification()]
)
service.notify_all("طلبك في الطريق!", "ahmad@example.com")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ===== قواعد تسمية PEP 8 =====
# متغيرات/دوال   -> snake_case
# ثوابت          -> UPPER_SNAKE_CASE
# أصناف          -> PascalCase
# خاص            -> _single_underscore / __double_underscore

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-001-greet.py
"""

# ✅ صحيح: مسافة واحدة حول عوامل التشغيل
total = price + tax

# ❌ خطأ: بدون مسافات، أو مسافات زائدة
total=price+tax
total  =  price  +  tax

# ✅ صحيح: بدون مسافة قبل الفاصلة، مسافة بعدها
def greet(name, age):
    pass

# ✅ صحيح: بدون مسافات حول = في المعاملات الافتراضية (دون type hint)
def greet(name="أحمد"):
    pass

# ✅ لكن مع type hint، نضيف مسافات حول =
def greet(name: str = "أحمد"):
    pass

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-002-result.py
"""

# ❌ سطر طويل جدًا، يصعب قراءته دفعة واحدة
result = calculate_total_price(unit_price, quantity, discount_percentage, tax_rate, shipping_cost)

# ✅ مقسّم بشكل مقروء
result = calculate_total_price(
    unit_price,
    quantity,
    discount_percentage,
    tax_rate,
    shipping_cost,
)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-003-ex24_2.py
"""

# ✅ الترتيب الموصى به: مكتبات قياسية، ثم مكتبات خارجية، ثم وحدات المشروع
import json
import os
from pathlib import Path

import pandas as pd
import requests

from my_project.models import User
from my_project.utils import format_date

# ❌ تجنّب الاستيراد الشامل -- يُخفي مصدر الأسماء ويُسبب تعارضات
from os import *

# ✅ استورد ما تحتاجه فقط بوضوح
from os import getcwd, path

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.3: التنسيق الآلي: `black`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-004-calculate.py
"""

# قبل black (ملف messy.py)
def   calculate(x,y) :
    result=x+y
    if result>10:
        return   'كبير'
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.3: التنسيق الآلي: `black`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-005-calculate.py
"""

# بعد black -- تلقائيًا!
def calculate(x, y):
    result = x + y
    if result > 10:
        return "كبير"
    else:
        return "صغير"

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.4: الفحص الآلي للأخطاء والجودة: `ruff`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-006-process_data.py
"""

# problematic.py
import os  # مستورد لكن غير مستخدم
import sys

def process_data(data):
    unused_variable = 42  # معرّف لكن غير مستخدم
    result = []
    for item in data:
        if item == None:  # يجب استخدام "is None" وليس "== None"
            continue
        result.append(item)
    return result

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-007-process_order.py
"""

# ❌ سيء: دالة واحدة تفعل أشياء كثيرة جدًا
def process_order(order_data):
    # التحقق من الصحة
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر")

    # حساب الإجمالي
    total = sum(item["price"] * item["qty"] for item in order_data["items"])

    # تطبيق الخصم
    if order_data.get("coupon"):
        total *= 0.9

    # حفظ في قاعدة البيانات
    print(f"حفظ الطلب بإجمالي {total}")

    # إرسال بريد إلكتروني
    print("إرسال تأكيد عبر البريد...")

    return total

# ✅ جيد: كل دالة مسؤولة عن شيء واحد فقط
def validate_order(order_data: dict) -> None:
    """التحقق من صحة بيانات الطلب."""
    if not order_data.get("items"):
        raise ValueError("لا توجد عناصر في الطلب")

def calculate_total(items: list[dict]) -> float:
    """حساب إجمالي سعر العناصر."""
    return sum(item["price"] * item["qty"] for item in items)

def apply_discount(total: float, has_coupon: bool) -> float:
    """تطبيق خصم 10% إذا وُجد كوبون."""
    return total * 0.9 if has_coupon else total

def save_order(total: float) -> None:
    """حفظ الطلب في قاعدة البيانات."""
    print(f"حفظ الطلب بإجمالي {total}")

def send_confirmation_email() -> None:
    """إرسال بريد تأكيد الطلب."""
    print("إرسال تأكيد عبر البريد...")

def process_order(order_data: dict) -> float:
    """تنسيق عملية معالجة الطلب الكاملة."""
    validate_order(order_data)
    total = calculate_total(order_data["items"])
    total = apply_discount(total, bool(order_data.get("coupon")))
    save_order(total)
    send_confirmation_email()
    return total

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-008-calculate_price.py
"""

# ❌ سيء: ما معنى 0.15؟ ولماذا 3؟
def calculate_price(amount):
    if amount > 1000:
        return amount * 0.85
    return amount * 0.95

for attempt in range(3):
    try_connection()

# ✅ جيد: الثوابت المُسمّاة تشرح نفسها
BULK_DISCOUNT_THRESHOLD = 1000
BULK_DISCOUNT_RATE = 0.85
STANDARD_DISCOUNT_RATE = 0.95
MAX_CONNECTION_ATTEMPTS = 3

def calculate_price(amount: float) -> float:
    if amount > BULK_DISCOUNT_THRESHOLD:
        return amount * BULK_DISCOUNT_RATE
    return amount * STANDARD_DISCOUNT_RATE

for attempt in range(MAX_CONNECTION_ATTEMPTS):
    try_connection()

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-009-get_discount.py
"""

# ❌ سيء: تداخل عميق (Nested Ifs) صعب التتبع
def get_discount(user):
    if user is not None:
        if user.is_active:
            if user.membership == "premium":
                return 0.2
            else:
                return 0.1
        else:
            return 0
    else:
        return 0

# ✅ جيد: Guard Clauses -- خروج مبكر، أقل تداخلًا
def get_discount(user) -> float:
    if user is None:
        return 0
    if not user.is_active:
        return 0
    if user.membership == "premium":
        return 0.2
    return 0.1

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.5: مبادئ أساسية لكتابة دوال وأصناف نظيفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-010-ex24_5.py
"""

# ❌ تعليق زائد لا يُضيف أي معلومة
x = x + 1  # إضافة 1 إلى x

# ❌ تعليق يشرح "ماذا" بدلاً من "لماذا" (الكود نفسه يوضّح "ماذا")
# نحسب السعر الإجمالي بضرب السعر في الكمية
total = price * quantity

# ✅ تعليق مفيد يشرح "لماذا" -- معلومة لا يمكن استنتاجها من الكود نفسه
# نستخدم هامش أمان 5% لتعويض تقلبات أسعار الصرف اليومية
total = price * quantity * 1.05

# ✅ تعليق يشرح قرارًا غير بديهي
# ملاحظة: نستخدم format التاريخ القديم هنا للتوافق مع النظام القديم (legacy system)
date_string = date.strftime("%d/%m/%Y")

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-011-__init__.py
"""

# ❌ ينتهك SRP: صنف واحد يفعل أشياء كثيرة غير مرتبطة
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

    def save_to_file(self, path: str) -> None:  # مسؤولية مختلفة: التخزين!
        with open(path, "w") as f:
            f.write(str(self.data))

    def send_email(self, address: str) -> None:  # مسؤولية مختلفة: الاتصال!
        print(f"إرسال التقرير إلى {address}")

# ✅ يتبع SRP: كل صنف مسؤول عن شيء واحد
class Report:
    def __init__(self, data: list[dict]):
        self.data = data

    def calculate_summary(self) -> dict:
        return {"total": sum(d["value"] for d in self.data)}

class ReportStorage:
    @staticmethod
    def save_to_file(report: Report, path: str) -> None:
        with open(path, "w") as f:
            f.write(str(report.data))

class ReportMailer:
    @staticmethod
    def send_email(report: Report, address: str) -> None:
        print(f"إرسال التقرير إلى {address}")

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-012-pay.py
"""

from abc import ABC, abstractmethod

# ✅ يتبع OCP: إضافة طريقة دفع جديدة لا تتطلب تعديل أي كود قديم
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...

class CreditCardPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر بطاقة الائتمان"

class PayPalPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر PayPal"

# لإضافة طريقة دفع جديدة، أضف صنفًا جديدًا فقط -- بدون لمس الكود القديم!
class CryptoPayment(PaymentMethod):
    def pay(self, amount: float) -> str:
        return f"دفع {amount} عبر العملات الرقمية"

def checkout(method: PaymentMethod, amount: float) -> None:
    print(method.pay(amount))

checkout(CreditCardPayment(), 100)
checkout(CryptoPayment(), 50)

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-013-fly.py
"""

# ❌ ينتهك LSP: الصنف الفرعي يُغيّر السلوك المتوقع بشكل مفاجئ
class Bird:
    def fly(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("البطريق لا يطير!")  # مفاجأة سيئة لأي كود يتوقع fly()

# ✅ يتبع LSP: إعادة تصميم الهرمية لتعكس الواقع بدقة
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def move(self) -> str:
        return "أسبح في الماء"  # سلوك صحيح ومتوقع، بدون كسر توقعات المستخدم

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-014-work.py
"""

# ❌ ينتهك ISP: واجهة ضخمة تُجبر كل شيء على تطبيق كل شيء
class Worker(ABC):
    @abstractmethod
    def work(self) -> str: ...
    @abstractmethod
    def eat(self) -> str: ...

class RobotWorker(Worker):
    def work(self) -> str:
        return "أعمل 24 ساعة"

    def eat(self) -> str:
        raise NotImplementedError("الروبوت لا يأكل!")  # إجبار غير منطقي

# ✅ يتبع ISP: واجهات صغيرة ومتخصصة، يطبّق كل صنف ما يحتاجه فقط
class Workable(ABC):
    @abstractmethod
    def work(self) -> str: ...

class Eatable(ABC):
    @abstractmethod
    def eat(self) -> str: ...

class HumanWorker(Workable, Eatable):
    def work(self) -> str:
        return "أعمل 8 ساعات"

    def eat(self) -> str:
        return "أتناول الغداء"

class RobotWorker(Workable):  # لا حاجة لتطبيق eat() أبدًا
    def work(self) -> str:
        return "أعمل 24 ساعة"

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-015-__init__.py
"""

# ❌ ينتهك DIP: الصنف مرتبط مباشرة بتفصيل محدد (MySQL)
class UserRepository:
    def __init__(self):
        self.db = MySQLDatabase()  # اعتماد مباشر وصلب على تفصيل محدد

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# ✅ يتبع DIP: الاعتماد على تجريد (Protocol كما تعلمنا في الفصل 15)
from typing import Protocol

class Database(Protocol):
    def query(self, sql: str) -> list: ...

class UserRepository:
    def __init__(self, db: Database):  # نعتمد على أي شيء يطابق الواجهة
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# الآن يمكن تمرير MySQL أو PostgreSQL أو حتى قاعدة بيانات وهمية للاختبار
repo = UserRepository(db=MySQLDatabase())
repo_test = UserRepository(db=FakeDatabaseForTesting())

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-016-calculate_circle_area.py
"""

# ❌ تكرار -- نفس المنطق مكرر 3 مرات
def calculate_circle_area(radius):
    return 3.14159 * radius ** 2

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

def calculate_sphere_volume(radius):
    return (4/3) * 3.14159 * radius ** 3

# ✅ استخراج الثابت المشترك
import math  # استخدام math.pi بدلاً من كتابة الرقم يدويًا في كل مكان

def calculate_circle_area(radius: float) -> float:
    return math.pi * radius ** 2

def calculate_circle_circumference(radius: float) -> float:
    return 2 * math.pi * radius

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-017-is_even.py
"""

# ❌ معقّد دون داعٍ
def is_even(n: int) -> bool:
    return True if n % 2 == 0 else False

# ✅ بسيط ومباشر
def is_even(n: int) -> bool:
    return n % 2 == 0

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.7: ممارسات إضافية للكود النظيف

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-018-__init__.py
"""

# ❌ مبالغة: نظام إعدادات معقّد لميزة بسيطة قد لا تُستخدم أبدًا
class ConfigurableGreeter:
    def __init__(self, language="ar", formal=False, emoji_support=True,
                 max_length=100, custom_templates=None, plugins=None):
        # ... 20 سطرًا من المرونة غير المستخدمة فعليًا

# ✅ ابدأ بسيطًا، وأضف المرونة فقط عند الحاجة الفعلية
def greet(name: str) -> str:
    return f"مرحبًا، {name}!"

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.8: نموذج عملي: قبل وبعد (Refactoring كامل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-019-proc.py
"""

# ===== قبل: كود يعمل لكنه غير نظيف =====
def proc(l):
    r = []
    for i in l:
        if i['type'] == 'A':
            if i['amt'] > 100:
                r.append({'id': i['id'], 'final': i['amt'] * 0.9})
            else:
                r.append({'id': i['id'], 'final': i['amt']})
        elif i['type'] == 'B':
            r.append({'id': i['id'], 'final': i['amt'] * 0.95})
    return r

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.8: نموذج عملي: قبل وبعد (Refactoring كامل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-020-calculate_type_a_discount.py
"""

# ===== بعد: نظيف، واضح، وقابل للصيانة =====

TYPE_A = "A"
TYPE_B = "B"

TYPE_A_BULK_THRESHOLD = 100
TYPE_A_BULK_DISCOUNT = 0.9
TYPE_B_STANDARD_DISCOUNT = 0.95

def calculate_type_a_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع A (خصم إضافي للكميات الكبيرة)."""
    if amount > TYPE_A_BULK_THRESHOLD:
        return amount * TYPE_A_BULK_DISCOUNT
    return amount

def calculate_type_b_discount(amount: float) -> float:
    """حساب السعر النهائي لعناصر النوع B (خصم ثابت)."""
    return amount * TYPE_B_STANDARD_DISCOUNT

def process_items(items: list[dict]) -> list[dict]:
    """معالجة قائمة عناصر وحساب السعر النهائي لكل منها حسب نوعه.

    Args:
        items: قائمة قواميس، كل منها يحتوي 'id', 'type', 'amt'.

    Returns:
        قائمة قواميس تحتوي 'id' و'final' (السعر النهائي بعد الخصم).
    """
    processed_items = []

    for item in items:
        if item["type"] == TYPE_A:
            final_price = calculate_type_a_discount(item["amt"])
        elif item["type"] == TYPE_B:
            final_price = calculate_type_b_discount(item["amt"])
        else:
            continue  # تخطَّ الأنواع غير المعروفة بدلاً من كسر البرنامج

        processed_items.append({"id": item["id"], "final": final_price})

    return processed_items

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-021-calculate_shipping.py
"""

def calculate_shipping(weight):
    if weight > 20:
        return weight * 5.5
    return weight * 3.2

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-022-calculate_shipping.py
"""

# 2
# d -> user_data
# tmp -> temporary_file_path
# lst2 -> filtered_products
# flag1 -> is_user_verified

# 3
HEAVY_PACKAGE_THRESHOLD_KG = 20
HEAVY_PACKAGE_RATE = 5.5
STANDARD_RATE = 3.2

def calculate_shipping(weight: float) -> float:
    if weight > HEAVY_PACKAGE_THRESHOLD_KG:
        return weight * HEAVY_PACKAGE_RATE
    return weight * STANDARD_RATE

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-023-handle_user_signup.py
"""

def handle_user_signup(email, password):
    if "@" not in email:
        raise ValueError("بريد غير صالح")
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")
    hashed = password[::-1]  # تشفير وهمي للتبسيط
    print(f"حفظ المستخدم: {email}")
    print(f"إرسال بريد ترحيبي إلى: {email}")
    return {"email": email, "password": hashed}

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-024-get_shipping_cost.py
"""

def get_shipping_cost(order):
    if order is not None:
        if order.weight > 0:
            if order.destination == "local":
                return order.weight * 2
            else:
                return order.weight * 5
    return 0

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-025-validate_email.py
"""

# 4
def validate_email(email: str) -> None:
    if "@" not in email:
        raise ValueError("بريد غير صالح")

def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")

def hash_password(password: str) -> str:
    return password[::-1]  # تشفير وهمي للتبسيط

def save_user(email: str, hashed_password: str) -> None:
    print(f"حفظ المستخدم: {email}")

def send_welcome_email(email: str) -> None:
    print(f"إرسال بريد ترحيبي إلى: {email}")

def handle_user_signup(email: str, password: str) -> dict:
    validate_email(email)
    validate_password(password)
    hashed = hash_password(password)
    save_user(email, hashed)
    send_welcome_email(email)
    return {"email": email, "password": hashed}

# 5
def get_shipping_cost(order) -> float:
    if order is None:
        return 0
    if order.weight <= 0:
        return 0
    if order.destination == "local":
        return order.weight * 2
    return order.weight * 5

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-026-send.py
"""

from abc import ABC, abstractmethod

class NotificationChannel(ABC):
    """تجريد (Abstraction) لأي قناة إشعار."""

    @abstractmethod
    def send(self, message: str, recipient: str) -> None:
        ...

class EmailNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📧 إرسال بريد إلى {recipient}: {message}")

class SMSNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"📱 إرسال رسالة نصية إلى {recipient}: {message}")

# إضافة قناة جديدة -- بدون أي تعديل على الكود القديم (مبدأ Open/Closed)
class PushNotification(NotificationChannel):
    def send(self, message: str, recipient: str) -> None:
        print(f"🔔 إرسال إشعار دفع إلى {recipient}: {message}")

class NotificationService:
    """تعتمد على التجريد فقط -- مبدأ Dependency Inversion."""

    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels

    def notify_all(self, message: str, recipient: str) -> None:
        for channel in self.channels:
            channel.send(message, recipient)

# الاستخدام
service = NotificationService(
    channels=[EmailNotification(), SMSNotification(), PushNotification()]
)
service.notify_all("طلبك في الطريق!", "ahmad@example.com")

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.11: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-027-ex24_11.py
"""

# ===== قواعد تسمية PEP 8 =====
# متغيرات/دوال   -> snake_case
# ثوابت          -> UPPER_SNAKE_CASE
# أصناف          -> PascalCase
# خاص            -> _single_underscore / __double_underscore

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
> # ❌ سيء جدًا -- لا يخبرنا بشيء
> def calc(x, y, z):
>     return x * y - z
>
> # ✅ ممتاز -- الاسم نفسه يشرح الغرض
> def calculate_net_price(unit_price: float, quantity: int, discount: float) -> float:
>     return unit_price * quantity - discount
> ```
> الكود الجيد **يُقرأ كالنثر**. إذا احتجت تعليقًا لشرح ماذا تفعل دالة أو متغيّر، فكّر أولًا: هل يمكن أن يشرح الاسم نفسه هذا بدلاً من التعليق؟

### المسافات والتباعد

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-24-chapter-24.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
> **⚠️ تحذير: الذكاء المفرط عدو الوضوح**
> ```python
> # ❌ «ذكي» جدًا لكن غير مقروء
> result = [x for x in range(100) if not any(x % i == 0 for i in range(2, int(x**0.5)+1)) and x > 1]
>
> # ✅ أوضح، حتى لو أطول قليلاً
> def is_prime(n: int) -> bool:
>     if n < 2:
>         return False
>     return all(n % i != 0 for i in range(2, int(n**0.5) + 1))
>
> result = [x for x in range(100) if is_prime(x)]
> ```
> الكود «الذكي جدًا» الذي يتباهى بحل كل شيء في سطر واحد غالبًا ما يكون عبئًا على كل من يقرأه لاحقًا — بما فيهم أنت. **الوضوح دائمًا أهم من الإيجاز.**

### YAGNI: لن تحتاجه (You Aren't Gonna Need It)

لا تبني مرونة أو ميزات «قد تحتاجها مستقبلاً» دون حاجة فعلية حالية. هذا يُعقّد الكود دون فائدة حقيقية.
