#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-003-test_simple_sentence.py
"""

import pytest

from textkit import is_palindrome, slugify, truncate, word_count

class TestSlugify:
    def test_simple_sentence(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self) -> None:
        assert slugify("  Python   Guide  ") == "python-guide"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

class TestTruncate:
    @pytest.mark.parametrize(
        "text, max_length, expected",
        [
            ("Python", 10, "Python"),
            ("Python is amazing", 10, "Python is..."),
            ("", 5, ""),
        ],
    )
    def test_truncate_cases(
        self, text: str, max_length: int, expected: str
    ) -> None:
        assert truncate(text, max_length) == expected

def test_word_count() -> None:
    assert word_count("Python is fun") == 3

def test_is_palindrome() -> None:
    assert is_palindrome("A man a plan a canal Panama") is True
    assert is_palindrome("Hello") is False
