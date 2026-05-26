#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# Problem parameters
T = 20
dt = 0.2
nx = 3
nu = 2
num_traj = 1000
num_plot = 12
save_file = "mpc_dataset.npz"

# actuator limits
v_min = 0.0
v_max = 2.0
omega_max = 1.5

# obstacle and obj
x_hat = np.array([[3.5], [0.0], [0.0]], dtype=float)
obstacle_center = np.array([[0.0], [0.0]], dtype=float)
Q = np.array([[0.51020408, 0.0], [0.0, 0.90702948]], dtype=float)
b_obs = 1.0

Q_stage = np.diag([1.0, 1.0, 1.0])
Q_terminal = np.diag([10.0, 10.0, 1.0])
R = np.diag([0.1, 0.1])

# Sampling and solver settings
seed = 7
maxiter = 180
ftol = 1e-5
sample_x_bounds = (-4.0, -2.2)
sample_y_bounds = (-2.4, 2.4)
sample_theta_bounds = (-0.45, 0.45)

out_dir = Path(__file__).resolve().parent
rng = np.random.default_rng(seed)

def unpack_z(z):
    X = z[: T * nx].reshape(T, nx)
    U = z[T * nx :].reshape(T, nu)
    return X, U

def pack_z(X, U):
    return np.concatenate((X.reshape(-1), U.reshape(-1)))

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def dynamics_step(x, u):
    return np.array(
        [
            x[0] + dt * u[0] * np.cos(x[2]),
            x[1] + dt * u[0] * np.sin(x[2]),
            x[2] + dt * u[1],
        ],
        dtype=float,
    )

def obstacle_margin_from_points(points):
    diff = points - obstacle_center.ravel()
    return np.einsum("ti,ij,tj->t", diff, Q, diff) - b_obs

def sample_initial_state():
    while True:
        x0 = np.array(
            [
                rng.uniform(*sample_x_bounds),
                rng.uniform(*sample_y_bounds),
                rng.uniform(*sample_theta_bounds),
            ],
            dtype=float,
        )
        if obstacle_margin_from_points(x0[:2].reshape(1, 2))[0] > 0.2:
            return x0

def objective(z):
    X, U = unpack_z(z)
    err = X - x_hat.ravel()
    err[:, 2] = wrap_to_pi(err[:, 2])
    state_cost = np.einsum("ti,ij,tj->", err, Q_stage, err)
    terminal_cost = float(err[-1] @ Q_terminal @ err[-1])
    control_cost = np.einsum("ti,ij,tj->", U, R, U)
    return float(state_cost + terminal_cost + control_cost)

def dynamics_constraint(z, x0):
    X, U = unpack_z(z)
    residual = np.zeros((T, nx), dtype=float)
    residual[0] = X[0] - dynamics_step(x0, U[0])
    for k in range(1, T):
        residual[k] = X[k] - dynamics_step(X[k - 1], U[k])
    return residual.reshape(-1)

def obstacle_constraint(z):
    X, _ = unpack_z(z)
    return obstacle_margin_from_points(X[:, :2])

def solve_mpc(x0):
    X0 = np.repeat(x0.reshape(1, nx), T, axis=0)
    U0 = np.zeros((T, nu), dtype=float)
    z0 = pack_z(X0, U0)
    bounds = [(None, None)] * (T * nx) + [(v_min, v_max), (-omega_max, omega_max)] * T
    return minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda z: dynamics_constraint(z, x0)},
            {"type": "ineq", "fun": obstacle_constraint},
        ],
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )


def obstacle_boundary(num_points=400):
    # x.T @ Q @ x only depends on the symmetric part of Q.
    Q_sym = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Q_sym)
    if np.any(eigvals <= 0.0):
        raise ValueError("Q must have a positive definite symmetric part to plot a bounded obstacle.")

    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    axes = np.sqrt(b_obs / eigvals)
    boundary = obstacle_center + eigvecs @ np.diag(axes) @ circle
    return boundary[0], boundary[1]


def save_plots(x0_samples, Z_all, cost_all):
    valid_idx = np.where(np.isfinite(cost_all))[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No solved trajectories to plot.")

    plot_idx = valid_idx[: min(num_plot, len(valid_idx))]
    n_cols = 4
    n_rows = int(np.ceil(len(plot_idx) / n_cols))
    obs_x, obs_y = obstacle_boundary()

    fig_x, axs_x = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    axs_x = np.array(axs_x).reshape(-1)
    for k, idx in enumerate(plot_idx):
        ax = axs_x[k]
        X, _ = unpack_z(Z_all[idx, :, 0])
        X_full = np.vstack((x0_samples[idx, :, 0], X))
        ax.plot(X_full[:, 0], X_full[:, 1], "-o", linewidth=1.5, markersize=4, label="state path")
        ax.scatter(X_full[0, 0], X_full[0, 1], c="tab:green", s=40, label="start")
        ax.scatter(X_full[-1, 0], X_full[-1, 1], c="tab:red", s=40, label="end")
        ax.scatter(x_hat[0, 0], x_hat[1, 0], c="tab:orange", marker="x", s=55, label="target")
        ax.fill(obs_x, obs_y, color="tab:purple", alpha=0.12, label="obstacle")
        ax.plot(obs_x, obs_y, color="tab:purple", linewidth=1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Trajectory {k + 1}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend()
    for k in range(len(plot_idx), len(axs_x)):
        axs_x[k].axis("off")
    fig_x.tight_layout()
    fig_x.savefig(out_dir / "x_grid_plot.png", dpi=200)
    plt.close(fig_x)

    fig_u, axs_u = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows))
    axs_u = np.array(axs_u).reshape(-1)
    for k, idx in enumerate(plot_idx):
        ax = axs_u[k]
        _, U = unpack_z(Z_all[idx, :, 0])
        ax.plot(U[:, 0], U[:, 1], "-o", linewidth=1.5, markersize=4, label="control path")
        ax.scatter(U[0, 0], U[0, 1], c="tab:green", s=40, label="start")
        ax.scatter(U[-1, 0], U[-1, 1], c="tab:red", s=40, label="end")
        ax.axvline(v_min, color="r", linestyle="--", linewidth=1.2, label="bounds")
        ax.axvline(v_max, color="r", linestyle="--", linewidth=1.2)
        ax.axhline(omega_max, color="r", linestyle="--", linewidth=1.2)
        ax.axhline(-omega_max, color="r", linestyle="--", linewidth=1.2)
        ax.set_xlabel("v")
        ax.set_ylabel("omega")
        ax.set_title(f"Controls {k + 1}")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend()
    for k in range(len(plot_idx), len(axs_u)):
        axs_u[k].axis("off")
    fig_u.tight_layout()
    fig_u.savefig(out_dir / "u_grid_plot.png", dpi=200)
    plt.close(fig_u)


def main():
    z_dim = T * nx + T * nu
    x0_samples = np.zeros((num_traj, nx, 1), dtype=float)
    Z_all = np.full((num_traj, z_dim, 1), np.nan, dtype=float)
    cost_all = np.full(num_traj, np.nan, dtype=float)
    status_all = np.array([""] * num_traj, dtype=object)

    for i in tqdm(range(num_traj)):
        x0 = sample_initial_state()
        x0_samples[i, :, 0] = x0
        result = solve_mpc(x0)

        if result.success:
            status_all[i] = "optimal"
            Z_all[i, :, 0] = result.x
            cost_all[i] = result.fun
        else:
            status_all[i] = f"solve_failed: {result.message}"
            print(f"Warning: trajectory {i} failed with status '{status_all[i]}'")

    # save trajectories
    np.savez(
        out_dir / save_file,
        x0_samples=x0_samples,
        x_hat=x_hat,
        Z_all=Z_all,
        cost_all=cost_all,
        status_all=status_all,
        nx=nx,
        nu=nu,
        T=T,
        dt=dt,
        v_min=v_min,
        v_max=v_max,
        omega_max=omega_max,
        obstacle_center=obstacle_center,
        Q=Q,
        b_obs=b_obs,
        Q_stage=Q_stage,
        Q_terminal=Q_terminal,
        R=R,
        seed=seed,
    )
    print(f"Saved trajectories to {out_dir / save_file}")

    valid_count = int(np.sum(np.isfinite(cost_all)))
    if valid_count != num_traj:
        print(f"Warning: only {valid_count} of {num_traj} trajectories were solved.")

    save_plots(x0_samples, Z_all, cost_all)

if __name__ == "__main__":
    main()
