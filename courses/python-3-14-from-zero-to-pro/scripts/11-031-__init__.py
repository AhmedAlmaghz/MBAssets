#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-031-__init__.py
"""

# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"
