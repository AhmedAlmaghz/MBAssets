#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.8: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-032-ex9_8.py
"""

# ============ DICT ============
d = {"key": "value"}                 # إنشاء
d["key"]                             # وصول (يطلق KeyError إن لم يوجد)
d.get("key", "default")              # وصول آمن
d["new_key"] = "value"               # إضافة/تعديل
del d["key"]  /  d.pop("key")        # حذف
"key" in d                           # فحص وجود مفتاح
for k, v in d.items(): ...           # تكرار على المفاتيح والقيم
{k: v for k, v in ...}               # Dict Comprehension
d.update(other_dict)                 # دمج قاموسين
d.setdefault(k, default)             # إرجاع أو إنشاء افتراضي

from collections import defaultdict
dd = defaultdict(list)               # قاموس بقيمة افتراضية تلقائية

# ============ SET ============
s = {1, 2, 3}                        # إنشاء (لاحظ: {} وحدها = dict فارغ!)
empty_set = set()                    # مجموعة فارغة - الطريقة الوحيدة
s.add(4)                             # إضافة عنصر
s.remove(1)  /  s.discard(1)         # حذف (remove يطلق خطأ، discard آمن)
a | b   # union() - اتحاد
a & b   # intersection() - تقاطع
a - b   # difference() - فرق
a ^ b   # symmetric_difference() - فرق تماثلي
{x for x in ...}                     # Set Comprehension
