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
