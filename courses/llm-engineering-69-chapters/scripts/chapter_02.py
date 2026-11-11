"""Standalone mini-lab for chapter 02: الفصل الثاني: الشبكة العصبية الأولى من الصفر — من نقطة واحدة إلى شبكة تتعلّم."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_perceptron() -> None:
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    weights, bias = ([0.0, 0.0], 0.0)
    for epoch in range(12):
        errors = 0
        for inputs, target in data:
            prediction = int(sum((w * x for w, x in zip(weights, inputs))) + bias >= 0.5)
            error = target - prediction
            errors += error != 0
            weights = [w + 0.2 * error * x for w, x in zip(weights, inputs)]
            bias += 0.2 * error
        if errors == 0:
            break
    print('AND predictions:', [int(sum((w * x for w, x in zip(weights, x))) + bias >= 0.5) for x, _ in data])
    print('weights / bias:', [round(w, 3) for w in weights], round(bias, 3))

if __name__ == "__main__":
    print("Chapter 02 lab: A trainable binary neuron")
    run_perceptron()
