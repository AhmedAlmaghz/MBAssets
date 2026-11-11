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
