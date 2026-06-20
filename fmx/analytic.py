import mlx.core as mx
from mlx.core import array


def gaussian_marginal_field(t: float, x: array, mu: array, sigma: array) -> array:
    # Closed-form marginal velocity u_t(x) = E[x1 - x0 | x_t = x] for x0 ~ N(0, I),
    # x1 ~ N(mu, sigma), independent straight-line interpolant. Affine in x, so it
    # is an exact oracle for the field the loss should recover. x: (N, d).
    d = mu.shape[0]
    eye = mx.eye(d)
    s_t = (1 - t) ** 2 * eye + t**2 * sigma
    cov_v_xt = t * sigma - (1 - t) * eye
    # inv has no GPU kernel; mlx's stub mistypes mx.cpu, hence the ignore.
    s_inv = mx.linalg.inv(s_t, stream=mx.cpu)  # pyright: ignore[reportArgumentType]
    return mu[None, :] + (x - t * mu[None, :]) @ (s_inv @ cov_v_xt.T)
