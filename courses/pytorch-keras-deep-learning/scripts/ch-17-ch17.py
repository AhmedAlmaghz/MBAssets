#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النماذج التوليدية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 17 — GAN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-17-ch17 (lesson 17)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-17-ch17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النماذج التوليدية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 17 — GAN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-17-ch17 (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch17.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class Generator(nn.Module):
    """يحوّل ضجيجاً (B, 100) إلى عيّنة (B, 1, 28, 28)."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """يعيد احتمال أن تكون العيّنة حقيقية."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


G = Generator()
D = Discriminator()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
print(G(torch.randn(4, 100)).shape)   # torch.Size([4, 1, 28, 28])

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch17.html
# ────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F

def train_step(G, D, opt_g, opt_d, real, z_dim=100):
    """خطوة تدريب واحدة: المميِّز خطوة، والمولِّد خطوة."""
    batch = real.size(0)

    # 1) تدريب المميِّز: حقيقي = 1، مولَّد = 0
    fake = G(torch.randn(batch, z_dim, device=real.device)).detach()
    d_real = D(real)
    d_fake = D(fake)
    loss_d = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) \
           + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # 2) تدريب المولِّد: هدفه أن يخدع المميِّز
    fake = G(torch.randn(batch, z_dim, device=real.device))
    d_out = D(fake)
    loss_g = F.binary_cross_entropy(d_out, torch.ones_like(d_out))

    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()

# الناتج المتوقع: (0.69, 0.69) في أول خطوة  ≈ ln(2)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch17.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """مُرمِّز تلقائي: مُرمِّز + مُفكِّ ترميز + تشريح."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # إعادة المعاينة (reparameterisation): z = mu + sigma * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        recon = self.decoder(z).view(-1, 1, 28, 28)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()

# الناتج: (loss_total, recon_loss, kl)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch17.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """يتنبّأ بالضجيج المضاف في كل خطوة."""

    def __init__(self, steps=1000):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer(
            "alpha_bars", torch.cumprod(alphas, dim=0)
        )  # جدول الضجيج التراكمي لكل خطوة

    def forward(self, x, t, noise):
        """يعيد الضجيج المتوقع في الخطوة t."""
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x + (1 - a).sqrt() * noise
        return self.unet(x_t, t), noise

# مثال على الجدولة: نجمع بيتا من 1e-4 إلى 0.02 عبر 1000 خطوة
betas = torch.linspace(1e-4, 0.02, 1000)
print(betas[0].item(), betas[-1].item())
# الناتج: 0.0001 0.02

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-17-ch17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class Generator(nn.Module):
    """يحوّل ضجيجاً (B, 100) إلى عيّنة (B, 1, 28, 28)."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """يعيد احتمال أن تكون العيّنة حقيقية."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


G = Generator()
D = Discriminator()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
print(G(torch.randn(4, 100)).shape)   # torch.Size([4, 1, 28, 28])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-17-ch17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F

def train_step(G, D, opt_g, opt_d, real, z_dim=100):
    """خطوة تدريب واحدة: المميِّز خطوة، والمولِّد خطوة."""
    batch = real.size(0)

    # 1) تدريب المميِّز: حقيقي = 1، مولَّد = 0
    fake = G(torch.randn(batch, z_dim, device=real.device)).detach()
    d_real = D(real)
    d_fake = D(fake)
    loss_d = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) \
           + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))

    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # 2) تدريب المولِّد: هدفه أن يخدع المميِّز
    fake = G(torch.randn(batch, z_dim, device=real.device))
    d_out = D(fake)
    loss_g = F.binary_cross_entropy(d_out, torch.ones_like(d_out))

    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()

# الناتج المتوقع: (0.69, 0.69) في أول خطوة  ≈ ln(2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-17-ch17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """مُرمِّز تلقائي: مُرمِّز + مُفكِّ ترميز + تشريح."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # إعادة المعاينة (reparameterisation): z = mu + sigma * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        recon = self.decoder(z).view(-1, 1, 28, 28)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()

# الناتج: (loss_total, recon_loss, kl)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-17-ch17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """يتنبّأ بالضجيج المضاف في كل خطوة."""

    def __init__(self, steps=1000):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer(
            "alpha_bars", torch.cumprod(alphas, dim=0)
        )  # جدول الضجيج التراكمي لكل خطوة

    def forward(self, x, t, noise):
        """يعيد الضجيج المتوقع في الخطوة t."""
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x + (1 - a).sqrt() * noise
        return self.unet(x_t, t), noise

# مثال على الجدولة: نجمع بيتا من 1e-4 إلى 0.02 عبر 1000 خطوة
betas = torch.linspace(1e-4, 0.02, 1000)
print(betas[0].item(), betas[-1].item())
# الناتج: 0.0001 0.02
