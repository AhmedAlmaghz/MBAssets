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
