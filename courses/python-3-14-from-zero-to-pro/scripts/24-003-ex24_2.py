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
