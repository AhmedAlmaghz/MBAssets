#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-020-__init__.py
"""

class Playlist:
    def __init__(self, songs: list[str]):
        self.songs = songs

    def __getitem__(self, index: int) -> str:
        return self.songs[index]

    def __len__(self) -> int:
        return len(self.songs)

playlist = Playlist(["أغنية 1", "أغنية 2", "أغنية 3"])
print(playlist[0])  # أغنية 1  -- الفهرسة تعمل الآن!

for song in playlist:  # __getitem__ تجعل الكائن قابلاً للتكرار تلقائياً أيضاً!
    print(song)
