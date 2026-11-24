#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس والعشرون: Pallas: نوى مخصّصة
JAX: From Zero to Production — Chapter 26 — Performance and Scale

Book slug : jax-from-zero-to-production
Chapter   : ch-26-ch26 (lesson 26)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-26-ch26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس والعشرون: Pallas: نوى مخصّصة
JAX: From Zero to Production — Chapter 26 — Performance and Scale

Book slug : jax-from-zero-to-production
Chapter   : ch-26-ch26 (lesson 26)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-26-ch26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس والعشرون: Pallas: نوى مخصّصة
JAX: From Zero to Production — Chapter 26 — Performance and Scale

Book slug : jax-from-zero-to-production
Chapter   : ch-26-ch26 (lesson 26)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import jax.experimental.pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y, block=256):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block,),
        in_specs=[pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
                  pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
    )(x, y)

x = jnp.arange(2048.0).reshape(8, 256)
y = jnp.ones_like(x)
print(jnp.allclose(add(x, y), x + y))   # True

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
def matmul_kernel(a_ref, b_ref, o_ref):
    # كل برنامج يراكم مساهمته في بلاطة الخرج
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])

def matmul(a, b, bm=128, bk=64, bn=128):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
    )(a, b)

A = jnp.ones((512, 256), jnp.float32)
B = jnp.ones((256, 512), jnp.float32)
print(jnp.allclose(matmul(A, B), A @ B, atol=1e-4))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
import time

def bench(fn, *args, n=50):
    fn(*args).block_until_ready()          # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args).block_until_ready()      # بدون هذا تقيس الطابور لا الحساب
    return (time.perf_counter() - t0) / n * 1e3

print("pallas ms:", bench(matmul, A, B))
print("xla    ms:", bench(lambda a, b: a @ b, A, B))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import jax.experimental.pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y, block=256):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block,),
        in_specs=[pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
                  pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
    )(x, y)

x = jnp.arange(2048.0).reshape(8, 256)
y = jnp.ones_like(x)
print(jnp.allclose(add(x, y), x + y))   # True

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def matmul_kernel(a_ref, b_ref, o_ref):
    # كل برنامج يراكم مساهمته في بلاطة الخرج
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])

def matmul(a, b, bm=128, bk=64, bn=128):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
    )(a, b)

A = jnp.ones((512, 256), jnp.float32)
B = jnp.ones((256, 512), jnp.float32)
print(jnp.allclose(matmul(A, B), A @ B, atol=1e-4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time

def bench(fn, *args, n=50):
    fn(*args).block_until_ready()          # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args).block_until_ready()      # بدون هذا تقيس الطابور لا الحساب
    return (time.perf_counter() - t0) / n * 1e3

print("pallas ms:", bench(matmul, A, B))
print("xla    ms:", bench(lambda a, b: a @ b, A, B))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس والعشرون: Pallas: نوى مخصّصة
JAX: From Zero to Production — Chapter 26 — Performance and Scale

Book slug : jax-from-zero-to-production
Chapter   : ch-26-ch26 (lesson 26)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import jax.experimental.pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y, block=256):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block,),
        in_specs=[pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
                  pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
    )(x, y)

x = jnp.arange(2048.0).reshape(8, 256)
y = jnp.ones_like(x)
print(jnp.allclose(add(x, y), x + y))   # True

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
def matmul_kernel(a_ref, b_ref, o_ref):
    # كل برنامج يراكم مساهمته في بلاطة الخرج
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])

def matmul(a, b, bm=128, bk=64, bn=128):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
    )(a, b)

A = jnp.ones((512, 256), jnp.float32)
B = jnp.ones((256, 512), jnp.float32)
print(jnp.allclose(matmul(A, B), A @ B, atol=1e-4))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch26.md
# ────────────────────────────────────────────────────────────────────────
import time

def bench(fn, *args, n=50):
    fn(*args).block_until_ready()          # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args).block_until_ready()      # بدون هذا تقيس الطابور لا الحساب
    return (time.perf_counter() - t0) / n * 1e3

print("pallas ms:", bench(matmul, A, B))
print("xla    ms:", bench(lambda a, b: a @ b, A, B))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import jax.experimental.pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y, block=256):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block,),
        in_specs=[pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
                  pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
    )(x, y)

x = jnp.arange(2048.0).reshape(8, 256)
y = jnp.ones_like(x)
print(jnp.allclose(add(x, y), x + y))   # True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def matmul_kernel(a_ref, b_ref, o_ref):
    # كل برنامج يراكم مساهمته في بلاطة الخرج
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])

def matmul(a, b, bm=128, bk=64, bn=128):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
    )(a, b)

A = jnp.ones((512, 256), jnp.float32)
B = jnp.ones((256, 512), jnp.float32)
print(jnp.allclose(matmul(A, B), A @ B, atol=1e-4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-26-ch26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time

def bench(fn, *args, n=50):
    fn(*args).block_until_ready()          # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args).block_until_ready()      # بدون هذا تقيس الطابور لا الحساب
    return (time.perf_counter() - t0) / n * 1e3

print("pallas ms:", bench(matmul, A, B))
print("xla    ms:", bench(lambda a, b: a @ b, A, B))
