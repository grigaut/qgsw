"""

Comparison between QG and SW solutions in vortex shear instability.

"""

# ruff: noqa
import numpy as np
import sys
import torch
import torch.nn.functional as F

sys.path.append("../src")

from qgsw.models.core.helmholtz import compute_laplace_dstI, dstI2D
from qgsw.models import SW, QG, SWFilterBarotropic

torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64


def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


# grid
nx = 96
ny = 96
nl = 1
L = 100000  # 100km
Lx = L
Ly = L
dx = torch.tensor(Lx / nx, device=device, dtype=dtype)
dy = torch.tensor(Ly / ny, device=device, dtype=dtype)
x_, y_ = (
    torch.linspace(-Lx / 2, Lx / 2, nx + 1, dtype=dtype, device=device),
    torch.linspace(-Ly / 2, Ly / 2, ny + 1, dtype=dtype, device=device),
)
x, y = torch.meshgrid(x_, y_, indexing="ij")

xc_ = 0.5 * (x_[1:] + x_[:-1])
yc_ = 0.5 * (y_[1:] + y_[:-1])
xc, yc = torch.meshgrid(xc_, yc_, indexing="ij")
rc = torch.sqrt(xc**2 + yc**2)
# circular domain mask
apply_mask = False
mask = (rc < L / 2).type(torch.float64) if apply_mask else torch.ones_like(xc)

H = torch.zeros(nl, 1, 1, dtype=dtype, device=device)
if nl == 1:
    H[0, 0, 0] = 1000  # 1km

# density/gravity
rho = 1000.0
g_prime = torch.zeros(nl, 1, 1, dtype=dtype, device=device)
if nl == 1:
    g_prime[0, 0, 0] = 10.0

flip_sign = False
# Burger and Rossby
for Bu, Ro in [
    # (1., 0.5),
    (1.0, 0.1),
    # (1., 0.02),
]:
    print(f"Ro={Ro} Bu={Bu}")

    r0, r1, r2 = 0.1 * Lx, 0.1 * Lx, 0.14 * Lx

    # set coriolis with burger number
    f0 = torch.sqrt(g_prime[0, 0, 0] * H[0, 0, 0] / Bu / r0**2)
    if flip_sign:
        f0 *= -1
    beta = 0
    f = f0 + beta * (y - Ly / 2)

    z = x + 1j * y
    theta = torch.angle(z)
    r = torch.sqrt(x**2 + y**2)

    # create rankine vortex with tripolar perturbation
    epsilon = 1e-3
    r *= 1 + epsilon * torch.cos(theta * 3)
    soft_step = lambda x: torch.sigmoid(x / 100)
    mask_core = soft_step(r0 - r)
    mask_ring = soft_step(r - r1) * soft_step(r2 - r)
    vor = 1.0 * (-mask_core / mask_core.mean() + mask_ring / mask_ring.mean())
    if flip_sign:
        vor *= -1
    laplace_dstI = compute_laplace_dstI(
        nx, ny, dx, dy, {"device": device, "dtype": dtype}
    )
    psi_hat = dstI2D(vor[1:-1, 1:-1]) / laplace_dstI
    psi = F.pad(dstI2D(psi_hat), (1, 1, 1, 1)).unsqueeze(0).unsqueeze(0)

    # set psi amplitude to have correct Rossby number
    u, v = grad_perp(psi, dx, dy)
    u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
    psi *= Ro * f0 * r0 / u_norm_max
    p_init = psi * f0

    # wind forcing, bottom drag
    taux = 0.0
    tauy = 0.0
    bottom_drag_coef = 0.0

    param_qg = {
        "nx": nx,
        "ny": ny,
        "nl": nl,
        "dx": dx,
        "dy": dy,
        "H": H,
        "g_prime": g_prime,
        "f": f,
        "taux": taux,
        "tauy": tauy,
        "bottom_drag_coef": bottom_drag_coef,
        "device": device,
        "dtype": dtype,
        "mask": mask,
        "compile": True,
        "slip_coef": 1.0,
        "dt": 0.0,
    }

    param_sw = {
        "nx": nx,
        "ny": ny,
        "nl": nl,
        "dx": dx,
        "dy": dy,
        "H": H,
        "rho": rho,
        "g_prime": g_prime,
        "f": f,
        "taux": taux,
        "tauy": tauy,
        "mask": mask,
        "bottom_drag_coef": bottom_drag_coef,
        "device": device,
        "dtype": dtype,
        "slip_coef": 1,
        "compile": True,
        "barotropic_filter": False,
        "barotropic_filter_spectral": False,
        "dt": 0.0,  # time-step (s)
    }

    qg_multilayer = QG(param_qg)

    u_init, v_init, h_init = qg_multilayer.G(p_init)

    u_max, v_max, c = (
        torch.abs(u_init).max().item() / dx,
        torch.abs(v_init).max().item() / dy,
        torch.sqrt(g_prime[0, 0, 0] * H.sum()),
    )
    print(f"u_max {u_max:.2e}, v_max {v_max:.2e}, c {c:.2e}")
    cfl_adv = 0.5
    cfl_gravity = 5 if param_sw["barotropic_filter"] else 0.5
    dt = min(cfl_adv * dx / u_max, cfl_adv * dy / v_max, cfl_gravity * dx / c)

    qg_multilayer.dt = dt
    qg_multilayer.u = torch.clone(u_init)
    qg_multilayer.v = torch.clone(v_init)
    qg_multilayer.h = torch.clone(h_init)
    qg_multilayer.compute_diagnostic_variables()

    w_qg = (qg_multilayer.omega.squeeze() / qg_multilayer.area).cpu().numpy()

    # update time step
    param_sw["dt"] = dt
    if param_sw["barotropic_filter"]:
        sw_multilayer = SWFilterBarotropic(param_sw)
    else:
        sw_multilayer = SW(param_sw)
    sw_multilayer.u = torch.clone(u_init)
    sw_multilayer.v = torch.clone(v_init)
    sw_multilayer.h = torch.clone(h_init)
    sw_multilayer.compute_diagnostic_variables()

    # time params
    t = 0

    qg_multilayer.compute_time_derivatives()
    wa_0 = qg_multilayer.omega_a.squeeze().cpu().numpy()

    w_0 = qg_multilayer.omega.squeeze() / qg_multilayer.dx / qg_multilayer.dy
    tau = 1.0 / torch.sqrt(w_0.pow(2).mean()).cpu().item()
    print(f"tau = {tau *f0:.2f} f0-1")

    t_end = 8 * tau
    freq_plot = int(t_end / 100 / dt) + 1
    freq_checknan = 100
    freq_log = int(t_end / 10 / dt) + 1
    n_steps = int(t_end / dt) + 1

    if freq_plot > 0:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update({"font.size": 18})
        palette = plt.cm.bwr  # .with_extremes(bad='grey')

        plt.ion()

        f, a = plt.subplots(1, 3, figsize=(18, 8))
        a[0].set_title("$\omega_{qg}$")
        a[1].set_title("$\omega_{sw}$")
        a[2].set_title("$\omega_{qg} - \omega_{sw}$")
        [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(3)]
        f.tight_layout()
        plt.pause(0.1)

    for n in range(0, n_steps + 1):
        if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            mask_w = sw_multilayer.masks.not_w[0, 0].cpu().numpy()
            w_qg = (
                (qg_multilayer.omega / qg_multilayer.area / qg_multilayer.f0)
                .cpu()
                .numpy()
            )
            w_sw = (
                (sw_multilayer.omega / sw_multilayer.area / sw_multilayer.f0)
                .cpu()
                .numpy()
            )
            wM = max(np.abs(w_qg).max(), np.abs(w_sw).max())

            kwargs = dict(
                cmap=palette, origin="lower", vmin=-wM, vmax=wM, animated=True
            )
            a[0].imshow(np.ma.masked_where(mask_w, w_qg[0, 0]).T, **kwargs)
            a[1].imshow(np.ma.masked_where(mask_w, w_sw[0, 0]).T, **kwargs)
            a[2].imshow(
                np.ma.masked_where(mask_w, (w_qg - w_sw)[0, 0]).T, **kwargs
            )
            f.suptitle(
                f'Ro={Ro:.2f}, Bu={Bu:.2f}, t={t/tau:.2f}$\\tau$, '
                f'{"neg." if flip_sign else "pos"} $f_0$'
            )
            plt.pause(0.05)

        qg_multilayer.step()
        sw_multilayer.step()
        t += dt
        if n % freq_checknan == 0:
            if torch.isnan(qg_multilayer.h).any():
                raise ValueError(
                    f"Stopping, NAN number in QG h at iteration {n}."
                )
            if torch.isnan(sw_multilayer.h).any():
                raise ValueError(
                    f"Stopping, NAN number in SW h at iteration {n}."
                )

        if freq_log > 0 and n % freq_log == 0:
            print(f"n={n:05d}, {qg_multilayer.get_print_info()}")
