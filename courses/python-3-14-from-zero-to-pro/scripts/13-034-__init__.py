#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.11: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-034-__init__.py
"""

# تعريف فئة أساسية
class ClassName:
    class_attribute = "مشترك بين الجميع"  # Class Attribute

    def __init__(self, param1, param2):
        self.attr1 = param1    # Instance Attribute
        self.attr2 = param2

    def instance_method(self):          # self = الكائن نفسه
        return self.attr1

    @classmethod
    def class_method(cls):              # cls = الفئة نفسها
        return cls.class_attribute

    @staticmethod
    def static_method():                # لا self ولا cls
        return "دالة مستقلة منطقياً"

# إنشاء كائن
obj = ClassName("value1", "value2")

# Magic Methods الأساسية
class MyClass:
    def __init__(self, x): self.x = x
    def __str__(self): return f"قابل للقراءة: {self.x}"
    def __repr__(self): return f"MyClass({self.x})"
    def __eq__(self, other): return self.x == other.x
    def __len__(self): return self.x
    def __add__(self, other): return MyClass(self.x + other.x)

# خصوصية البيانات (اصطلاحية)
self.public_attr      # وصول حر
self._protected_attr  # "تحذير" - داخلي، لا تعتمد عليه من الخارج
self.__private_attr   # Name Mangling → _ClassName__private_attr

# Property - getter/setter أنيق
class Account:
    def __init__(self, balance):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("لا يمكن أن يكون سالباً")
        self._balance = value
