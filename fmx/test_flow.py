import mlx.core as mx
import mlx.nn as nn

from fmx.analytic import gaussian_marginal_field
from fmx.checkpoint import latest_ckpt, load, save
from fmx.data import toy_sampler
from fmx.nets import MLPConfig, UNetConfig
from fmx.sample import sample
from fmx.sanity import behavior_report
from fmx.train import fm_loss, train


class ZeroField(nn.Module):
    def __call__(self, t, x):
        return mx.zeros_like(x)


def test_loss_is_zero_when_field_equals_target():
    # ZeroField predicts 0; with x0 == x1 the target x1 - x0 is 0, so loss is 0.
    x = mx.zeros((8, 2))
    t = mx.random.uniform(0, 1, (8,))
    assert fm_loss(ZeroField(), x, x, t).item() == 0.0


def test_loss_sums_every_non_batch_axis():
    # target = x1 - x0 = ones, prediction 0 -> squared error 1 per pixel; summed
    # over 28*28 pixels and meaned over the batch -> 784.
    x0 = mx.zeros((3, 28, 28, 1))
    x1 = mx.ones((3, 28, 28, 1))
    t = mx.zeros((3,))
    assert abs(fm_loss(ZeroField(), x0, x1, t).item() - 28 * 28) < 1e-3


def test_unet_velocity_shape_and_sampler_roundtrip():
    model = UNetConfig(base=16).build()
    x = mx.random.normal((5, 28, 28, 1))
    t = mx.random.uniform(0, 1, (5,))
    assert model(t, x).shape == x.shape
    assert sample(model, x, n_steps=4).shape == (5, 5, 28, 28, 1)


def test_learned_field_matches_gaussian_oracle():
    mu = mx.array([1.5, -0.5])
    a = mx.array([[0.8, 0.0], [0.4, 0.6]])  # sigma = a a^T, anisotropic
    sigma = a @ a.T

    def sampler(n):
        return mu[None, :] + mx.random.normal((n, 2)) @ a.T

    mx.random.seed(0)
    model = MLPConfig().build()
    train(model, sampler, steps=4000, batch=1024, log_every=4000)

    # Error weighted by the marginal density N(t*mu, S_t): the field is only
    # constrained where x_t has mass, so tail grid points must not dominate.
    g = mx.linspace(-2.0, 3.0, 21)
    gx, gy = mx.meshgrid(g, g)
    grid = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    eye = mx.eye(2)
    worst = 0.0
    for t in (0.25, 0.5, 0.75):
        s_t = (1 - t) ** 2 * eye + t**2 * sigma
        d = grid - t * mu[None, :]
        s_inv = mx.linalg.inv(s_t, stream=mx.cpu)  # pyright: ignore[reportArgumentType]
        w = mx.exp(-0.5 * (d @ s_inv * d).sum(-1))
        analytic = gaussian_marginal_field(t, grid, mu, sigma)
        pred = model(mx.full((grid.shape[0],), t), grid)
        err = mx.sqrt(((pred - analytic) ** 2).sum(-1))
        worst = max(worst, (w * err).sum().item() / w.sum().item())
    assert worst < 0.15, f"density-weighted field error {worst}"


def test_samples_match_target_moments():
    mx.random.seed(0)
    model = MLPConfig().build()
    train(model, toy_sampler("two_moons"), steps=3000, batch=512, log_every=3000)
    gen = sample(model, mx.random.normal((2000, 2)), n_steps=50)[-1]
    data = toy_sampler("two_moons")(5000)
    assert mx.abs(gen.mean(0) - data.mean(0)).max().item() < 0.15
    assert mx.abs(gen.std(0) - data.std(0)).max().item() < 0.15


def test_checkpoint_roundtrip(tmp_path):
    cfg = UNetConfig(base=16)
    model = cfg.build()
    save(tmp_path / "ck", model, cfg, meta={"steps": 1})
    reloaded, cfg2, meta = load(tmp_path / "ck")
    assert cfg2 == cfg and meta["steps"] == 1
    x = mx.random.normal((2, 28, 28, 1))
    t = mx.random.uniform(0, 1, (2,))
    assert mx.allclose(model(t, x), reloaded(t, x), atol=1e-5)


def test_latest_ckpt_filters_by_config_type(tmp_path):
    import os

    save(tmp_path / "mlp", MLPConfig().build(), MLPConfig(), meta={})
    save(tmp_path / "old_unet", UNetConfig(base=16).build(), UNetConfig(base=16), meta={})
    save(tmp_path / "new_unet", UNetConfig(base=16).build(), UNetConfig(base=16), meta={})
    os.utime(tmp_path / "old_unet" / "config.json", (1, 1))  # force old_unet oldest
    assert latest_ckpt(tmp_path, config_type="UNetConfig").name == "new_unet"


def test_sanity_confirms_t_and_x_dependence():
    mx.random.seed(0)
    model = UNetConfig(base=16).build()
    r = behavior_report(model, mx.random.normal((8, 28, 28, 1)))
    assert r["t_conditioning_ok"] and r["x_conditioning_ok"]


def test_sanity_flags_a_t_blind_net():
    # a net that ignores t is the #1 silent flow-matching bug; it must be caught.
    class TBlind(nn.Module):
        def __call__(self, t, x):
            return x * 2.0

    r = behavior_report(TBlind(), mx.random.normal((8, 2)))
    assert not r["t_conditioning_ok"]
    assert r["x_conditioning_ok"]


def test_train_reports_loss_and_grad_norm():
    logged = []
    train(MLPConfig().build(), toy_sampler("two_moons"), steps=20, batch=64,
          log_every=10, on_log=lambda i, m: logged.append(m))
    assert logged and all({"loss", "grad_norm"} <= m.keys() for m in logged)
    assert all(m["grad_norm"] >= 0 and m["loss"] >= 0 for m in logged)
