#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.4: Instance Attributes مقابل Class Attributes

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-013-ex13_4.py
"""

dog1.species = "Wolf"  # ⚠️ هذا لا يُعدّل Class Attribute!
                        # بل يُنشئ Instance Attribute جديد خاص بـ dog1 فقط!

print(dog1.species)  # Wolf (الآن Instance Attribute خاص به)
print(dog2.species)  # Canis familiaris (لم يتأثر إطلاقاً!)
print(Dog.species)   # Canis familiaris (الفئة نفسها لم تتغير)
