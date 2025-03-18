# ruff: noqa: PGH004
# ruff: noqa
import os
import numpy as np
import sys
import torch

sys.path.append("../src")


from qgsw.models import SW, QG, SWFilterBarotropic

torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

### Double-gyre on beta plane with idealized wind forcing

# grid
# Lx = 5120.0e3
# nx = 256
Lx = 2560.0e3
nx = 128
Ly = 5120.0e3
ny = 256
mask = torch.ones(nx, ny, dtype=dtype, device=device)
# mask[0,0] = 0
# mask[0,-1] = 0

H = torch.zeros(3, 1, 1, dtype=dtype, device=device)
H[0, 0, 0] = 400.0
H[1, 0, 0] = 1100.0
H[2, 0, 0] = 2600.0

# density/gravity
rho = 1000
g_prime = torch.zeros(3, 1, 1, dtype=dtype, device=device)
g_prime[0, 0, 0] = 9.81
g_prime[1, 0, 0] = 0.025
g_prime[2, 0, 0] = 0.0125


# coriolis beta plane
f0 = 9.375e-5  # mean coriolis (s^-1)
beta = 1.754e-11  # coriolis gradient (m^-1 s^-1)

# bottom drag
bottom_drag_coef = 0.5 * f0 * 2.0 / 2600

param = {
    "nl": 3,
    "H": H,
    "rho": rho,
    "g_prime": g_prime,
    "bottom_drag_coef": bottom_drag_coef,
    "device": device,
    "dtype": dtype,
    "slip_coef": 1.0,
    "interp_fd": False,
    "dt": 4000,  # time-step (s)
    "compile": True,
    "barotropic_filter": True,
    "barotropic_filter_spectral": True,
    "mask": mask,
}


for model, name, dt, start_file in [
    # (QG, 'qg', 4000, ''),
    (SW, "sw", 4000, ""),
    # (SW, 'sw', 4000, 'run_outputs/qg_256x256_dt4000_slip1.0/uvh_100y_010d.npz'),
]:
    param["nx"] = nx
    param["ny"] = ny
    param["dt"] = dt

    dx = Lx / nx
    dy = Ly / ny
    param["dx"] = dx
    param["dy"] = dy
    x, y = torch.meshgrid(
        torch.linspace(0, Lx, nx + 1, dtype=torch.float64, device=device),
        torch.linspace(0, Ly, ny + 1, dtype=torch.float64, device=device),
        indexing="ij",
    )

    # set time step given barotropic mode for SW
    if model == SW:
        c = torch.sqrt(H.sum() * g_prime[0, 0, 0]).cpu().item()
        cfl = 20 if param["barotropic_filter"] else 0.5
        if param["barotropic_filter"]:
            model = SWFilterBarotropic
        dt = float(int(cfl * min(dx, dy) / c))
        print(f"dt = {dt:.1f} s.")
        param["dt"] = dt

    print(f"Double gyre config, {name} model, {nx}x{ny} grid, dt {dt:.1f}s.")

    # corolis param grid
    f = f0 + beta * (y - Ly / 2)
    param["f"] = f

    # wind forcing
    mag = 0.08  # Wind stress magnitude (Pa m-1 kg s-2)
    tau0 = mag / rho
    y_ugrid = 0.5 * (y[:, 1:] + y[:, :-1])
    taux = tau0 * torch.cos(2 * torch.pi * (y_ugrid - Ly / 2) / Ly)[1:-1, :]
    tauy = 0.0
    param["taux"] = taux
    param["tauy"] = tauy

    qgsw_multilayer = model(param)

    if start_file:
        print(f"Starting from file {start_file}...")
        zipf = np.load(start_file)
        qgsw_multilayer.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])

    t = 0

    freq_checknan = 10
    freq_log = 100

    n_steps = int(10 * 365 * 24 * 3600 / dt) + 1
    n_steps_save = int(2 * 365 * 24 * 3600 / dt)
    freq_save = int(15 * 24 * 3600 / dt)
    freq_plot = int(15 * 24 * 3600 / dt)

    uM, vM, hM = 0, 0, 0

    if freq_save > 0:
        output_dir = (
            f"run_outputs/{name}_{nx}x{ny}_dt{dt}_slip{param['slip_coef']}/"
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to {output_dir}")

    if freq_plot > 0:
        import matplotlib.pyplot as plt

        plt.ion()
        nl_plot = 0
        if model == QG:
            npx, npy = 2, 1
            f, a = plt.subplots(npy, npx, figsize=(12, 12))
            a[0].set_title("$\\omega_g$")
            a[1].set_title("$\\omega_a$")
            [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(npx)]
        else:
            npx, npy = 3, 1
            f, a = plt.subplots(npy, npx, figsize=(16, 6))
            a[0].set_title("$u$")
            a[1].set_title("$v$")
            a[2].set_title("$h$")
            [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(npx)]

        plt.tight_layout()
        plt.pause(0.1)
        plot_kwargs = {"cmap": "bwr", "origin": "lower", "animated": True}

    for n in range(1, n_steps + 1):
        qgsw_multilayer.step()
        t += dt

        n_years = int(t // (365 * 24 * 3600))
        n_days = int(t % (365 * 24 * 3600) // (24 * 3600))

        if n % freq_checknan == 0 and torch.isnan(qgsw_multilayer.p).any():
            raise ValueError(f"Stopping, NAN number in p at iteration {n}.")

        if freq_log > 0 and n % freq_log == 0:
            print(
                f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, {qgsw_multilayer.get_print_info()}"
            )

        if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            u, v, h = qgsw_multilayer.get_physical_uvh_as_ndarray()
            uM, vM = (
                max(uM, 0.8 * np.abs(u).max()),
                max(vM, 0.8 * np.abs(v).max()),
            )
            hM = max(hM, 0.8 * np.abs(h).max())
            if model == QG:
                wM = 0.2
                w = (
                    (
                        qgsw_multilayer.omega
                        / qgsw_multilayer.area
                        / qgsw_multilayer.f0
                    )
                    .cpu()
                    .numpy()
                )
                w_a = (
                    (qgsw_multilayer.omega_a / qgsw_multilayer.f0)
                    .cpu()
                    .numpy()
                )

                a[0].imshow(w[0, nl_plot].T, vmin=-wM, vmax=wM, **plot_kwargs)
                a[1].imshow(
                    w_a[0, nl_plot].T,
                    vmin=-0.2 * wM,
                    vmax=0.2 * wM,
                    **plot_kwargs,
                )
            else:
                a[0].imshow(u[0, nl_plot].T, vmin=-uM, vmax=uM, **plot_kwargs)
                a[1].imshow(v[0, nl_plot].T, vmin=-vM, vmax=vM, **plot_kwargs)
                a[2].imshow(h[0, nl_plot].T, vmin=-hM, vmax=hM, **plot_kwargs)

            f.suptitle(f"{n_years} yrs, {n_days:03d} days")
            plt.pause(0.05)

        if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
            filename = os.path.join(
                output_dir, f"uvh_{n_years:03d}y_{n_days:03d}d.npz"
            )
            u, v, h = qgsw_multilayer.get_physical_uvh_as_ndarray()
            if model == QG:
                u_a = qgsw_multilayer.u_a.cpu().numpy()
                v_a = qgsw_multilayer.v_a.cpu().numpy()
                np.savez(
                    filename,
                    u=u.astype("float32"),
                    v=v.astype("float32"),
                    u_a=u_a.astype("float32"),
                    v_a=v_a.astype("float32"),
                    h=h.astype("float32"),
                )
                print(f"saved u,v,h,u_a,v_a to {filename}")
            else:
                np.savez(
                    filename,
                    u=u.astype("float32"),
                    v=v.astype("float32"),
                    h=h.astype("float32"),
                )
                print(f"saved u,v,h to {filename}")
