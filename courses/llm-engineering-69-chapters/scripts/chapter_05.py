"""Standalone mini-lab for chapter 05: الفصل الخامس: أساسيات PyTorch العملية والبنية التحتية الحسابية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_torch() -> None:
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.25], [0.1, 0.3]])
    loss = (x @ weights).square().mean()
    loss.backward()
    print('torch:', torch.__version__, '| device:', x.device)
    print('input/output shapes:', tuple(x.shape), tuple((x @ weights).shape))
    print('loss / gradient norm:', round(loss.item(), 4), round(x.grad.norm().item(), 4))

if __name__ == "__main__":
    print("Chapter 05 lab: Tensor shapes, autograd, and devices")
    run_torch()
