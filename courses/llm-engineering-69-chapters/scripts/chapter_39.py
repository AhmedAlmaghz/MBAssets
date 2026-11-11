"""Standalone mini-lab for chapter 39: الفصل 39: Vision Transformer (ViT) — من البكسل إلى الرمز."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vision_patches() -> None:
    image = [[row * 4 + col for col in range(4)] for row in range(4)]
    patch_size = 2
    patches = [[image[r + dr][c + dc] for dr in range(patch_size) for dc in range(patch_size)] for r in range(0, 4, patch_size) for c in range(0, 4, patch_size)]
    print('image / patch:', (4, 4), '/', (patch_size, patch_size))
    print('patch token count:', len(patches))
    print('flattened patch vectors:', patches)

if __name__ == "__main__":
    print("Chapter 39 lab: Split an image into Vision Transformer patches")
    run_vision_patches()
