"""Standalone mini-lab for chapter 23: الفصل 23: التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_distributed() -> None:
    batch, data_parallel, tensor_parallel, pipeline_stages = (96, 4, 2, 3)
    per_replica_batch = batch // data_parallel
    print('global batch:', batch)
    print('batch per data-parallel replica:', per_replica_batch)
    print('model partitions (tensor × pipeline):', tensor_parallel * pipeline_stages)
    print('3-D ranks:', data_parallel * tensor_parallel * pipeline_stages)

if __name__ == "__main__":
    print("Chapter 23 lab: Shard a batch across a 3-D parallel layout")
    run_distributed()
