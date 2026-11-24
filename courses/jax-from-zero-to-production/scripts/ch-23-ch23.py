#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث والعشرون: النماذج التوليدية
JAX: From Zero to Production — VAE

Book slug : jax-from-zero-to-production
Chapter   : ch-23-ch23 (lesson 23)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-23-ch23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث والعشرون: النماذج التوليدية
JAX: From Zero to Production — VAE

Book slug : jax-from-zero-to-production
Chapter   : ch-23-ch23 (lesson 23)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-23-ch23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث والعشرون: النماذج التوليدية
JAX: From Zero to Production — VAE

Book slug : jax-from-zero-to-production
Chapter   : ch-23-ch23 (lesson 23)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    latent: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.latent)(h), nn.Dense(self.latent)(h)   # μ, log σ²

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        h = nn.relu(nn.Dense(256)(z))
        return nn.Dense(784)(nn.relu(nn.Dense(256)(h)))             # logits

def vae_loss(params, x, key):
    mu, logvar = Encoder().apply({"params": params["enc"]}, x)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    logits = Decoder().apply({"params": params["dec"]}, z)
    rec = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits, x), -1))
    kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), -1))
    return rec + kl

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
def gan_step(g_params, d_params, real, key):
    z = jax.random.normal(key, (real.shape[0], 32))
    fake = G().apply({"params": g_params}, z)

    def d_loss(dp):
        r = D().apply({"params": dp}, real)
        f = D().apply({"params": dp}, fake)
        return jnp.mean(jax.nn.softplus(-r)) + jnp.mean(jax.nn.softplus(f))

    def g_loss(gp):
        f = D().apply({"params": d_params}, G().apply({"params": gp}, z))
        return jnp.mean(jax.nn.softplus(-f))

    return jax.grad(d_loss)(d_params), jax.grad(g_loss)(g_params)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
T = 1000
betas = jnp.linspace(1e-4, 0.02, T)
abar = jnp.cumprod(1.0 - betas)

def train_step(params, x0, key):
    k1, k2 = jax.random.split(key)
    t = jax.random.randint(k1, (x0.shape[0],), 0, T)
    eps = jax.random.normal(k2, x0.shape)
    a = abar[t][:, None, None, None]
    xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps

    def loss(p):
        return jnp.mean((model.apply({"params": p}, xt, t) - eps) ** 2)

    return loss(params), jax.grad(loss)(params)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    latent: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.latent)(h), nn.Dense(self.latent)(h)   # μ, log σ²

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        h = nn.relu(nn.Dense(256)(z))
        return nn.Dense(784)(nn.relu(nn.Dense(256)(h)))             # logits

def vae_loss(params, x, key):
    mu, logvar = Encoder().apply({"params": params["enc"]}, x)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    logits = Decoder().apply({"params": params["dec"]}, z)
    rec = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits, x), -1))
    kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), -1))
    return rec + kl

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def gan_step(g_params, d_params, real, key):
    z = jax.random.normal(key, (real.shape[0], 32))
    fake = G().apply({"params": g_params}, z)

    def d_loss(dp):
        r = D().apply({"params": dp}, real)
        f = D().apply({"params": dp}, fake)
        return jnp.mean(jax.nn.softplus(-r)) + jnp.mean(jax.nn.softplus(f))

    def g_loss(gp):
        f = D().apply({"params": d_params}, G().apply({"params": gp}, z))
        return jnp.mean(jax.nn.softplus(-f))

    return jax.grad(d_loss)(d_params), jax.grad(g_loss)(g_params)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
T = 1000
betas = jnp.linspace(1e-4, 0.02, T)
abar = jnp.cumprod(1.0 - betas)

def train_step(params, x0, key):
    k1, k2 = jax.random.split(key)
    t = jax.random.randint(k1, (x0.shape[0],), 0, T)
    eps = jax.random.normal(k2, x0.shape)
    a = abar[t][:, None, None, None]
    xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps

    def loss(p):
        return jnp.mean((model.apply({"params": p}, xt, t) - eps) ** 2)

    return loss(params), jax.grad(loss)(params)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث والعشرون: النماذج التوليدية
JAX: From Zero to Production — VAE

Book slug : jax-from-zero-to-production
Chapter   : ch-23-ch23 (lesson 23)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    latent: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.latent)(h), nn.Dense(self.latent)(h)   # μ, log σ²

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        h = nn.relu(nn.Dense(256)(z))
        return nn.Dense(784)(nn.relu(nn.Dense(256)(h)))             # logits

def vae_loss(params, x, key):
    mu, logvar = Encoder().apply({"params": params["enc"]}, x)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    logits = Decoder().apply({"params": params["dec"]}, z)
    rec = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits, x), -1))
    kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), -1))
    return rec + kl

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
def gan_step(g_params, d_params, real, key):
    z = jax.random.normal(key, (real.shape[0], 32))
    fake = G().apply({"params": g_params}, z)

    def d_loss(dp):
        r = D().apply({"params": dp}, real)
        f = D().apply({"params": dp}, fake)
        return jnp.mean(jax.nn.softplus(-r)) + jnp.mean(jax.nn.softplus(f))

    def g_loss(gp):
        f = D().apply({"params": d_params}, G().apply({"params": gp}, z))
        return jnp.mean(jax.nn.softplus(-f))

    return jax.grad(d_loss)(d_params), jax.grad(g_loss)(g_params)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch23.md
# ────────────────────────────────────────────────────────────────────────
T = 1000
betas = jnp.linspace(1e-4, 0.02, T)
abar = jnp.cumprod(1.0 - betas)

def train_step(params, x0, key):
    k1, k2 = jax.random.split(key)
    t = jax.random.randint(k1, (x0.shape[0],), 0, T)
    eps = jax.random.normal(k2, x0.shape)
    a = abar[t][:, None, None, None]
    xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps

    def loss(p):
        return jnp.mean((model.apply({"params": p}, xt, t) - eps) ** 2)

    return loss(params), jax.grad(loss)(params)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    latent: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.latent)(h), nn.Dense(self.latent)(h)   # μ, log σ²

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        h = nn.relu(nn.Dense(256)(z))
        return nn.Dense(784)(nn.relu(nn.Dense(256)(h)))             # logits

def vae_loss(params, x, key):
    mu, logvar = Encoder().apply({"params": params["enc"]}, x)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    logits = Decoder().apply({"params": params["dec"]}, z)
    rec = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits, x), -1))
    kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), -1))
    return rec + kl

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def gan_step(g_params, d_params, real, key):
    z = jax.random.normal(key, (real.shape[0], 32))
    fake = G().apply({"params": g_params}, z)

    def d_loss(dp):
        r = D().apply({"params": dp}, real)
        f = D().apply({"params": dp}, fake)
        return jnp.mean(jax.nn.softplus(-r)) + jnp.mean(jax.nn.softplus(f))

    def g_loss(gp):
        f = D().apply({"params": d_params}, G().apply({"params": gp}, z))
        return jnp.mean(jax.nn.softplus(-f))

    return jax.grad(d_loss)(d_params), jax.grad(g_loss)(g_params)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-23-ch23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
T = 1000
betas = jnp.linspace(1e-4, 0.02, T)
abar = jnp.cumprod(1.0 - betas)

def train_step(params, x0, key):
    k1, k2 = jax.random.split(key)
    t = jax.random.randint(k1, (x0.shape[0],), 0, T)
    eps = jax.random.normal(k2, x0.shape)
    a = abar[t][:, None, None, None]
    xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps

    def loss(p):
        return jnp.mean((model.apply({"params": p}, xt, t) - eps) ** 2)

    return loss(params), jax.grad(loss)(params)
