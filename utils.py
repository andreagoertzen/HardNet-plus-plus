import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MPCFullTrajectoryDataset(Dataset):
    def __init__(
        self,
        data,
        accepted_statuses=("optimal",),
        include_terminal=True,
        dtype=torch.float32,
    ):
        x0_all = data["x0_samples"].reshape(data["x0_samples"].shape[0], -1)
        z_all = data["Z_all"].reshape(data["Z_all"].shape[0], -1)
        status = data["status_all"]
        T = int(np.asarray(data["T"]).reshape(-1)[0])
        nx = int(np.asarray(data["nx"]).reshape(-1)[0])
        nu = int(np.asarray(data["nu"]).reshape(-1)[0])

        valid = np.isfinite(z_all).all(axis=1)
        if accepted_statuses is not None:
            if isinstance(accepted_statuses, str):
                accepted_statuses = (accepted_statuses,)
            valid = valid & np.isin(status, accepted_statuses)

        x_samples = []
        y_samples = []
        n_states_per_traj = T + 1 if include_terminal else T
        for x0, z in zip(x0_all[valid], z_all[valid]):
            X = z[: T * nx].reshape(T, nx)
            U = z[T * nx :].reshape(T, nu)
            X_full = np.vstack((x0.reshape(1, nx), X))

            for k in range(n_states_per_traj):
                future_X = X_full[k + 1 :]
                if future_X.shape[0] < T:
                    pad_X = np.repeat(X_full[-1].reshape(1, nx), T - future_X.shape[0], axis=0)
                    future_X = np.vstack((future_X, pad_X))
                else:
                    future_X = future_X[:T]

                future_U = U[k:] if k < T else np.zeros((0, nu), dtype=float)
                if future_U.shape[0] < T:
                    pad_U = np.zeros((T - future_U.shape[0], nu), dtype=float)
                    future_U = np.vstack((future_U, pad_U))
                else:
                    future_U = future_U[:T]

                x_samples.append(X_full[k])
                y_samples.append(np.concatenate((future_X.reshape(-1), future_U.reshape(-1))))

        if len(x_samples) == 0:
            raise ValueError("No valid trajectory states found in dataset.")

        self.x = torch.tensor(np.asarray(x_samples), dtype=dtype, device=device)
        self.y = torch.tensor(np.asarray(y_samples), dtype=dtype, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_full_trajectory_dataloaders(
    data,
    batch_size: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 13,
    include_terminal: bool = True,
):
    dataset = MPCFullTrajectoryDataset(data, include_terminal=include_terminal)
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def sample_fresh_initial_states(
    data,
    num_samples,
    seed=101,
    sample_x_bounds=(-4.0, -2.2),
    sample_y_bounds=(-2.4, 2.4),
    sample_theta_bounds=(-0.45, 0.45),
    obstacle_margin=0.2,
):
    rng = np.random.default_rng(seed)
    nx = int(np.asarray(data["nx"]).reshape(-1)[0])
    Q = np.asarray(data["Q"], dtype=float)
    b_obs = float(np.asarray(data["b_obs"]).reshape(-1)[0])
    obstacle_center = np.asarray(data["obstacle_center"], dtype=float).reshape(-1)

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
            if obstacle_margin_from_points(x0[:2].reshape(1, 2))[0] > obstacle_margin:
                return x0

    return np.asarray([sample_initial_state() for _ in range(num_samples)])


def make_mpc_cfg(
    T,
    nx,
    nu,
    dt,
    x_hat,
    Q_stage,
    Q_terminal,
    R,
    obstacle_center,
    Q,
    b_obs,
    v_min,
    v_max,
    omega_max,
):
    return {
        "T": int(T),
        "nx": int(nx),
        "nu": int(nu),
        "dt": float(dt),
        "x_hat": np.asarray(x_hat, dtype=float).reshape(int(nx)),
        "Q_stage": np.asarray(Q_stage, dtype=float),
        "Q_terminal": np.asarray(Q_terminal, dtype=float),
        "R": np.asarray(R, dtype=float),
        "obstacle_center": np.asarray(obstacle_center, dtype=float).reshape(-1),
        "Q": np.asarray(Q, dtype=float),
        "b_obs": float(b_obs),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "omega_max": float(omega_max),
    }


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dynamics_step(state, control, dt):
    px, py, theta = state
    v_k, omega_k = control
    return np.array(
        [
            px + dt * v_k * np.cos(theta),
            py + dt * v_k * np.sin(theta),
            theta + dt * omega_k,
        ],
        dtype=float,
    )


def unpack_z(z, T, nx, nu):
    X = z[: T * nx].reshape(T, nx)
    U = z[T * nx :].reshape(T, nu)
    return X, U


def pack_z(X, U):
    return np.concatenate((X.reshape(-1), U.reshape(-1)))


def mpc_objective(z, cfg):
    X, U = unpack_z(z, cfg["T"], cfg["nx"], cfg["nu"])
    err = X - cfg["x_hat"].reshape(1, cfg["nx"])
    err[:, 2] = wrap_to_pi(err[:, 2])
    state_cost = np.einsum("ti,ij,tj->", err, cfg["Q_stage"], err)
    terminal_cost = float(err[-1] @ cfg["Q_terminal"] @ err[-1])
    control_cost = np.einsum("ti,ij,tj->", U, cfg["R"], U)
    return float(state_cost + terminal_cost + control_cost)


def dynamics_residual(z, x0, cfg):
    X, U = unpack_z(z, cfg["T"], cfg["nx"], cfg["nu"])
    residual = np.zeros((cfg["T"], cfg["nx"]), dtype=float)
    residual[0] = X[0] - dynamics_step(x0, U[0], cfg["dt"])
    for k in range(1, cfg["T"]):
        residual[k] = X[k] - dynamics_step(X[k - 1], U[k], cfg["dt"])
    return residual.reshape(-1)


def obstacle_margin(z, cfg):
    X, _ = unpack_z(z, cfg["T"], cfg["nx"], cfg["nu"])
    diff = X[:, :2] - cfg["obstacle_center"]
    return np.einsum("ti,ij,tj->t", diff, cfg["Q"], diff) - cfg["b_obs"]


def rollout_controls(x0, U, cfg):
    X = np.zeros((U.shape[0] + 1, cfg["nx"]), dtype=float)
    X[0] = x0
    for k, u in enumerate(U):
        X[k + 1] = dynamics_step(X[k], u, cfg["dt"])
    return X


def controls_to_full_z(u_flat, x0, cfg):
    U = u_flat.reshape(cfg["T"], cfg["nu"])
    X = rollout_controls(x0, U, cfg)
    return pack_z(X[1:], U)


def proportional_control_guess(x0, cfg, turn_bias=0.0):
    U = np.zeros((cfg["T"], cfg["nu"]), dtype=float)
    x = x0.copy()
    v_nom = 0.65 * cfg["v_max"] + 0.35 * cfg["v_min"]
    for k in range(cfg["T"]):
        to_goal = cfg["x_hat"][:2] - x[:2]
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        heading_error = wrap_to_pi(desired_heading - x[2])
        omega = np.clip(
            2.0 * heading_error + turn_bias,
            -cfg["omega_max"],
            cfg["omega_max"],
        )
        remaining_time = max((cfg["T"] - k) * cfg["dt"], cfg["dt"])
        v = np.clip(
            np.linalg.norm(to_goal) / remaining_time,
            cfg["v_min"],
            cfg["v_max"],
        )
        v = 0.5 * v + 0.5 * v_nom
        U[k] = [v, omega]
        x = dynamics_step(x, U[k], cfg["dt"])
    return U.reshape(-1)


def initial_control_guesses(x0, cfg):
    guesses = []
    v_mid = 0.5 * (cfg["v_min"] + cfg["v_max"])
    base = np.zeros((cfg["T"], cfg["nu"]), dtype=float)
    base[:, 0] = v_mid
    guesses.append(base.reshape(-1))

    for omega in [
        0.0,
        -0.35 * cfg["omega_max"],
        0.35 * cfg["omega_max"],
        -0.75 * cfg["omega_max"],
        0.75 * cfg["omega_max"],
    ]:
        arc = base.copy()
        arc[:, 1] = omega
        guesses.append(arc.reshape(-1))

    for turn_bias in [
        0.0,
        -0.5 * cfg["omega_max"],
        0.5 * cfg["omega_max"],
        -cfg["omega_max"],
        cfg["omega_max"],
    ]:
        guesses.append(proportional_control_guess(x0, cfg, turn_bias=turn_bias))

    unique = []
    for guess in guesses:
        if not any(np.allclose(guess, existing) for existing in unique):
            unique.append(guess)
    return unique


def build_ipopt_solver(cfg):
    try:
        import casadi as ca
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CasADi is required for the IPOPT ground-truth solver. "
            "Run eval.py in an environment with casadi installed."
        ) from exc

    T = cfg["T"]
    nx = cfg["nx"]
    nu = cfg["nu"]
    dt = cfg["dt"]

    X = ca.MX.sym("X", T * nx)
    U = ca.MX.sym("U", T * nu)
    x0_param = ca.MX.sym("x0", nx)
    z = ca.vertcat(X, U)

    Q_stage = ca.DM(cfg["Q_stage"])
    Q_terminal = ca.DM(cfg["Q_terminal"])
    R = ca.DM(cfg["R"])
    Q_obs = ca.DM(cfg["Q"])
    x_hat = ca.DM(cfg["x_hat"])
    obstacle_center = ca.DM(cfg["obstacle_center"])

    objective_expr = ca.MX(0)
    constraints = []
    for k in range(T):
        xk = X[k * nx : (k + 1) * nx]
        uk = U[k * nu : (k + 1) * nu]
        prev = x0_param if k == 0 else X[(k - 1) * nx : k * nx]

        next_dyn = ca.vertcat(
            prev[0] + dt * uk[0] * ca.cos(prev[2]),
            prev[1] + dt * uk[0] * ca.sin(prev[2]),
            prev[2] + dt * uk[1],
        )
        constraints.extend([xk[i] - next_dyn[i] for i in range(nx)])

        err = xk - x_hat
        err = ca.vertcat(err[0], err[1], ca.atan2(ca.sin(err[2]), ca.cos(err[2])))
        objective_expr += ca.mtimes([err.T, Q_stage, err]) + ca.mtimes([uk.T, R, uk])

        obs_diff = xk[0:2] - obstacle_center
        constraints.append(ca.mtimes([obs_diff.T, Q_obs, obs_diff]) - cfg["b_obs"])

    err_terminal = X[(T - 1) * nx : T * nx] - x_hat
    err_terminal = ca.vertcat(
        err_terminal[0],
        err_terminal[1],
        ca.atan2(ca.sin(err_terminal[2]), ca.cos(err_terminal[2])),
    )
    objective_expr += ca.mtimes([err_terminal.T, Q_terminal, err_terminal])

    nlp = {"x": z, "p": x0_param, "f": objective_expr, "g": ca.vertcat(*constraints)}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
    }
    solver = ca.nlpsol(f"ground_truth_ipopt_{abs(id(cfg))}", "ipopt", nlp, opts)

    lbx = np.full(T * nx + T * nu, -np.inf, dtype=float)
    ubx = np.full(T * nx + T * nu, np.inf, dtype=float)
    for k in range(T):
        u_offset = T * nx + k * nu
        lbx[u_offset] = cfg["v_min"]
        ubx[u_offset] = cfg["v_max"]
        lbx[u_offset + 1] = -cfg["omega_max"]
        ubx[u_offset + 1] = cfg["omega_max"]

    lbg = []
    ubg = []
    for _ in range(T):
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)
        lbg.append(0.0)
        ubg.append(np.inf)

    return {
        "solver": solver,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": np.asarray(lbg),
        "ubg": np.asarray(ubg),
    }


def solve_mpc_ipopt(x0, cfg):
    from types import SimpleNamespace

    if "_ipopt_solver" not in cfg:
        cfg["_ipopt_solver"] = build_ipopt_solver(cfg)
    bundle = cfg["_ipopt_solver"]
    z0_list = [
        controls_to_full_z(u_guess, x0, cfg)
        for u_guess in initial_control_guesses(x0, cfg)
    ]

    best = None
    best_score = np.inf
    for z0 in z0_list:
        try:
            sol = bundle["solver"](
                x0=z0,
                p=x0,
                lbx=bundle["lbx"],
                ubx=bundle["ubx"],
                lbg=bundle["lbg"],
                ubg=bundle["ubg"],
            )
            z_sol = np.asarray(sol["x"]).reshape(-1)
            stats = bundle["solver"].stats()
            success = bool(stats.get("success", False))
            fun = mpc_objective(z_sol, cfg)
            margin = float(np.min(obstacle_margin(z_sol, cfg)))
            dyn_resid = float(np.max(np.abs(dynamics_residual(z_sol, x0, cfg))))
            score = fun + 1e8 * max(-margin, 0.0) ** 2 + 1e8 * dyn_resid**2
            score += 0.0 if success else 1e3
            if score < best_score:
                best_score = score
                best = SimpleNamespace(
                    x=z_sol,
                    fun=fun,
                    success=success,
                    message=stats.get("return_status", "unknown"),
                    nit=int(stats.get("iter_count", -1)),
                    min_margin=margin,
                    max_dyn_resid=dyn_resid,
                    score=score,
                )
        except RuntimeError as exc:
            if best is None:
                best = SimpleNamespace(
                    x=z0,
                    fun=mpc_objective(z0, cfg),
                    success=False,
                    message=str(exc),
                    nit=-1,
                    min_margin=float(np.min(obstacle_margin(z0, cfg))),
                    max_dyn_resid=float(np.max(np.abs(dynamics_residual(z0, x0, cfg)))),
                    score=np.inf,
                )

    if best is None:
        raise RuntimeError("IPOPT did not return any candidate solution.")
    return best


def solve_ground_truth_mpc_ipopt(
    x0,
    T,
    nx,
    nu,
    dt,
    x_hat,
    Q_stage,
    Q_terminal,
    R,
    obstacle_center,
    Q,
    b_obs,
    v_min,
    v_max,
    omega_max,
):
    cache_key = (
        int(T),
        int(nx),
        int(nu),
        float(dt),
        float(b_obs),
        float(v_min),
        float(v_max),
        float(omega_max),
        tuple(np.asarray(x_hat, dtype=float).reshape(-1)),
        tuple(np.asarray(obstacle_center, dtype=float).reshape(-1)),
        tuple(np.asarray(Q, dtype=float).reshape(-1)),
        tuple(np.asarray(Q_stage, dtype=float).reshape(-1)),
        tuple(np.asarray(Q_terminal, dtype=float).reshape(-1)),
        tuple(np.asarray(R, dtype=float).reshape(-1)),
    )
    cache = getattr(solve_ground_truth_mpc_ipopt, "_cache", {})
    if cache_key not in cache:
        cache[cache_key] = make_mpc_cfg(
            T,
            nx,
            nu,
            dt,
            x_hat,
            Q_stage,
            Q_terminal,
            R,
            obstacle_center,
            Q,
            b_obs,
            v_min,
            v_max,
            omega_max,
        )
        solve_ground_truth_mpc_ipopt._cache = cache

    cfg = cache[cache_key]
    return solve_mpc_ipopt(np.asarray(x0, dtype=float).reshape(cfg["nx"]), cfg)

