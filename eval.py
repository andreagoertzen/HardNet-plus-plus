import torch
import numpy as np
from utils import sample_fresh_initial_states, unpack_z
from torch.utils.data import DataLoader, TensorDataset
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_traj_rows(
    X_full_batch,
    U_full_batch,
    x_hat,
    Q,
    quad_ub,
    u_b,
    X_full_opt=None,
    U_full_opt=None,
    save_folder=None,
    tag=None,
):
    'open loop plotting'
    X_full_batch = np.asarray(X_full_batch)
    U_full_batch = np.asarray(U_full_batch)
    x_hat = np.asarray(x_hat).reshape(-1)
    Q = np.asarray(Q)
    n_batch, _, nx = X_full_batch.shape
    nu = U_full_batch.shape[2]

    if X_full_opt is not None:
        X_full_opt = np.asarray(X_full_opt)
    if U_full_opt is not None:
        U_full_opt = np.asarray(U_full_opt)

    point_alpha = 0.7
    n_rows = n_batch
    n_x_cols = 1
    n_u_cols = 1

    fig_x, axs_x = plt.subplots(n_rows, n_x_cols, figsize=(5.5 * n_x_cols, 2.8 * n_rows))
    fig_u, axs_u = plt.subplots(n_rows, n_u_cols, figsize=(5.2 * n_u_cols, 2.8 * n_rows))
    axs_x = np.array(axs_x).reshape(n_rows, n_x_cols)
    axs_u = np.array(axs_u).reshape(n_rows, n_u_cols)

    theta = np.linspace(0, 2 * np.pi, 300)
    Q_sym = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Q_sym)
    if np.any(eigvals <= 0.0):
        raise ValueError("Q must have a positive definite symmetric part to plot a bounded obstacle.")
    ellipse = (np.sqrt(quad_ub) * (eigvecs @ (np.stack([np.cos(theta), np.sin(theta)], axis=0) / np.sqrt(eigvals)[:, None]))).T

    for i in range(n_rows):
        X_full = X_full_batch[i]
        U_full = U_full_batch[i]
        X_opt_i = X_full_opt[i] if X_full_opt is not None else None
        U_opt_i = U_full_opt[i] if U_full_opt is not None else None

        ax = axs_x[i, 0]
        ax.plot(X_full[:, 0], X_full[:, 1], "-o", label="state path", alpha=point_alpha)
        if X_opt_i is not None:
            ax.plot(X_opt_i[:, 0], X_opt_i[:, 1], "--o", label="optimal path", alpha=point_alpha)
        ax.scatter(X_full[0, 0], X_full[0, 1], c="tab:green", s=50, label="start", alpha=point_alpha)
        ax.scatter(X_full[-1, 0], X_full[-1, 1], c="tab:red", s=50, label="end", alpha=point_alpha)
        ax.scatter(x_hat[0], x_hat[1], c="tab:orange", marker="x", s=70, label="target", alpha=point_alpha)
        ax.plot(ellipse[:, 0], ellipse[:, 1], color="blue", linewidth=1.2, label="keep-out ellipsoid")
        ax.fill(ellipse[:, 0], ellipse[:, 1], color="blue", alpha=0.08)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Case {i+1}: (x, y)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        if i == 0:
            ax.legend()

        ax = axs_u[i, 0]
        ax.plot(U_full[:, 0], U_full[:, 1], "-o", label="control path", alpha=point_alpha)
        if U_opt_i is not None:
            ax.plot(U_opt_i[:, 0], U_opt_i[:, 1], "--o", label="optimal path", alpha=point_alpha)
        ax.scatter(U_full[0, 0], U_full[0, 1], c="tab:green", s=50, label="start", alpha=point_alpha)
        ax.scatter(U_full[-1, 0], U_full[-1, 1], c="tab:red", s=50, label="end", alpha=point_alpha)
        ax.axvline(u_b[0], color="r", linestyle="--", linewidth=1.2, label="bounds")
        ax.axvline(u_b[1], color="r", linestyle="--", linewidth=1.2)
        ax.axhline(u_b[2], color="r", linestyle="--", linewidth=1.2)
        ax.axhline(u_b[3], color="r", linestyle="--", linewidth=1.2)
        ax.set_xlabel("v")
        ax.set_ylabel("omega")
        ax.set_title(f"Case {i+1}: (v, omega)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    fig_x.tight_layout()
    fig_u.tight_layout()

    if save_folder is not None:
        x_fig_name = f"{save_folder}/x_rows_plot{tag}.png"
        u_fig_name = f"{save_folder}/u_rows_plot{tag}.png"
        fig_x.savefig(x_fig_name, dpi=200)
        fig_u.savefig(u_fig_name, dpi=200)

    plt.close(fig_x)
    plt.close(fig_u)

def torch_objective(z, T, nx, nu, x_hat, Q_stage, Q_terminal, R):
    x_seq = z[: T * nx].reshape(T, nx)
    u_seq = z[T * nx :].reshape(T, nu)
    x_target = torch.tensor(x_hat.reshape(1, nx), dtype=z.dtype, device=z.device)
    Q_stage_t = torch.tensor(Q_stage, dtype=z.dtype, device=z.device)
    Q_terminal_t = torch.tensor(Q_terminal, dtype=z.dtype, device=z.device)
    R_t = torch.tensor(R, dtype=z.dtype, device=z.device)

    err = x_seq - x_target
    err[:, 2] = torch.atan2(torch.sin(err[:, 2]), torch.cos(err[:, 2]))
    state_cost = torch.einsum("ti,ij,tj->", err, Q_stage_t, err)
    terminal_cost = err[-1] @ Q_terminal_t @ err[-1]
    control_cost = torch.einsum("ti,ij,tj->", u_seq, R_t, u_seq)
    return state_cost + terminal_cost + control_cost

def obstacle_margin(z, T, nx, nu, obstacle_center, Q, b_obs):
    X, _ = unpack_z(z, T, nx, nu)
    diff = X[:, :2] - obstacle_center
    return np.einsum("bi,ij,bj->b", diff, Q, diff) - b_obs

def solve_ground_truth_mpc(x0, T, nx, nu, dt, x_hat, Q_stage, Q_terminal, R, obstacle_center, Q, b_obs, v_min, v_max, omega_max):
    from utils import solve_ground_truth_mpc_ipopt

    return solve_ground_truth_mpc_ipopt(
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
    )

def eval(parent_dir):
    cpu_d = torch.device("cpu")
    script_dir = Path(__file__).resolve().parent
    ### LOAD DATA
    npz_path = script_dir / "mpc_dataset.npz"
    data = np.load(npz_path, allow_pickle=True)
    n_samples=100
    x0_samples = sample_fresh_initial_states(data, n_samples, seed=101)
    x0_tensor = torch.tensor(x0_samples, dtype=torch.float32, device=device)
    dummy_y = torch.zeros((x0_tensor.shape[0], 1), dtype=torch.float32, device=device)
    test_loader = DataLoader(TensorDataset(x0_tensor, dummy_y), batch_size=n_samples, shuffle=False)

    n_in = data["x0_samples"].shape[1]
    nx = int(np.asarray(data["nx"]).reshape(-1)[0])
    nu = int(np.asarray(data["nu"]).reshape(-1)[0])
    params = np.load(f'{parent_dir}/model_params.npz', allow_pickle=True)
    model_params = {
            'n_in': n_in,
            'nx': int(np.asarray(params["nx"]).reshape(-1)[0]),
            'nu': int(np.asarray(params["nu"]).reshape(-1)[0]),
            'Q': params["Q"],
            'b_obs': float(np.asarray(params["b_obs"]).reshape(-1)[0]),
            'v_min': float(np.asarray(params["v_min"]).reshape(-1)[0]),
            'v_max': float(np.asarray(params["v_max"]).reshape(-1)[0]),
            'omega_max': float(np.asarray(params["omega_max"]).reshape(-1)[0]),
            'T': int(np.asarray(params["T"]).reshape(-1)[0]),
            'dt': float(np.asarray(params["dt"]).reshape(-1)[0]),
            'obstacle_center': np.asarray(params["obstacle_center"]).reshape(-1),
            'constrained': params["constrained"].item() if np.asarray(params["constrained"]).shape == () else params["constrained"],
            'n_proj': int(np.asarray(params["n_proj"]).reshape(-1)[0]),
            'epsilon': float(np.asarray(params["epsilon"]).reshape(-1)[0]),
            'dc3':bool(np.asarray(params["dc3"]).reshape(-1)[0])
        }
    if model_params["dc3"]:
        model_params.update({
        "dc3_steps": int(np.asarray(params["dc3_steps"]).reshape(-1)[0]),
        "dc3_stepsize": float(np.asarray(params["dc3_stepsize"]).reshape(-1)[0]),
        "dc3_momentum": float(np.asarray(params.get("dc3_momentum",0.0)).reshape(-1)[0])
        })

    T_out = model_params["T"]
    folder = parent_dir
    if model_params["dc3"]:
        print('using dc3 model')
        from model_dc3 import Model
    else:
        from model import Model
    
    model = Model(model_params).to(device)

    best_model = torch.load(f'{folder}/best_model.pt',map_location=device)
    model.load_state_dict(best_model["model_state_dict"])
    model.eval()

    T = model_params["T"]
    nx = model_params["nx"]
    nu = model_params["nu"]
    dt = model_params["dt"]
    b_obs = model_params["b_obs"]
    v_min = model_params["v_min"]
    v_max = model_params["v_max"]
    omega_max = model_params["omega_max"]
    Q = np.asarray(model_params["Q"], dtype=float)
    obstacle_center = np.asarray(model_params["obstacle_center"], dtype=float).reshape(-1)
    x_hat = data["x_hat"]
    Q_stage = np.asarray(data["Q_stage"], dtype=float)
    Q_terminal = np.asarray(data["Q_terminal"], dtype=float)
    R = np.asarray(data["R"], dtype=float)
    max_sub = 0.0

    with torch.no_grad():
        control_violation_sum = 0.0
        dyn_eq_violation_sum = 0.0
        obstacle_violation_sum = 0.0
        control_violation_max_all = 0.0
        dyn_violation_max_all = 0.0
        obstacle_violation_max_all = 0.0
        timing_msgs = []
        for idx_batch, (x_test,_) in enumerate(test_loader):
            x_test = x_test.to(device)
            batch_size = x_test.shape[0]
            y_pred = model(x_test)
                
            model_suboptimality = 0
            n_optimal = 0

            suboptimality_plotting = []
            plot_indices = []
            U_solvers = []
            X_solvers = []
            for ind in range(batch_size):
                model_obj = torch_objective(y_pred[ind,:], T, nx, nu, x_hat, Q_stage, Q_terminal, R).to(cpu_d)

                x0_np = x_test[ind,:].detach().cpu().numpy().reshape(nx)

                result = solve_ground_truth_mpc(
                    x0_np, T, nx, nu, dt, x_hat, Q_stage, Q_terminal, R,
                    obstacle_center, Q, b_obs, v_min, v_max, omega_max,
                )
                margin = float(np.min(obstacle_margin(result.x, T, nx, nu, obstacle_center, Q, b_obs)))
                if result.success and margin >= -5e-4 and np.isfinite(result.fun):
                    true_obj = result.fun
                    sub = torch.relu(model_obj - true_obj) / true_obj
                    sub_value = float(sub.item())
                    model_suboptimality += sub_value
                    n_optimal += 1
                    max_sub = max(max_sub, sub_value)
                    if len(plot_indices) < 4:
                        plot_indices.append(ind)
                        suboptimality_plotting.append(sub_value)
                        X_solver, U_solver = unpack_z(result.x, T, nx, nu)
                        X_solvers.append(torch.tensor(X_solver, dtype=torch.float32))
                        U_solvers.append(torch.tensor(U_solver, dtype=torch.float32))
                else:
                    print('solver solution not optimal')
            if n_optimal > 0:
                optimality_msg = f"suboptimality (average across {n_optimal}: {model_suboptimality/n_optimal})"
            else:
                optimality_msg = "suboptimality: no optimal solver solutions"
            optimality_max_msg = f'maximum suboptimality: {max_sub}'
            print(optimality_msg)
            print(optimality_max_msg)

            ## CONSTRAINT VIOLATIONS
            x_seq = y_pred[:, :T_out * nx].reshape(batch_size, T_out, nx)
            u_seq = y_pred[:, T_out * nx:].reshape(batch_size, T_out, nu)

            prev_x = torch.cat((x_test[:, 0:1], x_seq[:, :-1, 0]), dim=1)
            prev_y = torch.cat((x_test[:, 1:2], x_seq[:, :-1, 1]), dim=1)
            prev_theta = torch.cat((x_test[:, 2:3], x_seq[:, :-1, 2]), dim=1)

            dyn_x = x_seq[:, :, 0] - (prev_x + dt * u_seq[:, :, 0] * torch.cos(prev_theta))
            dyn_y = x_seq[:, :, 1] - (prev_y + dt * u_seq[:, :, 0] * torch.sin(prev_theta))
            dyn_theta = x_seq[:, :, 2] - (prev_theta + dt * u_seq[:, :, 1])
            dyn_residual = torch.stack((dyn_x, dyn_y, dyn_theta), dim=2)
            dyn_violation = dyn_residual.abs().max()
            dyn_eq_violation_sum += dyn_residual.abs().mean() * batch_size
            dyn_violation_max_all = max(dyn_violation_max_all, dyn_violation.item())

            obs_diff = x_seq[:, :, :2] - torch.tensor(obstacle_center, dtype=torch.float32, device=device).view(1, 1, -1)
            obs_val = torch.einsum("btn,nm,btm->bt", obs_diff, torch.tensor(Q, dtype=torch.float32, device=device), obs_diff) - b_obs
            obstacle_violation = torch.relu(-obs_val)
            obstacle_violation_max = obstacle_violation.max()
            obstacle_violation_sum += obstacle_violation.mean() * batch_size
            obstacle_violation_max_all = max(obstacle_violation_max_all, obstacle_violation_max.item())

            v_lower = torch.relu(torch.tensor(v_min, dtype=torch.float32, device=device) - u_seq[:, :, 0])
            v_upper = torch.relu(u_seq[:, :, 0] - torch.tensor(v_max, dtype=torch.float32, device=device))
            omega_lower = torch.relu(torch.tensor(-omega_max, dtype=torch.float32, device=device) - u_seq[:, :, 1])
            omega_upper = torch.relu(u_seq[:, :, 1] - torch.tensor(omega_max, dtype=torch.float32, device=device))
            control_violation = torch.stack((v_lower, v_upper, omega_lower, omega_upper), dim=2)
            control_violation_max = control_violation.max()
            control_violation_sum += control_violation.mean() * batch_size
            control_violation_max_all = max(control_violation_max_all, control_violation_max.item())

            ## plotting
            if len(X_solvers) < 4:
                print(f"Skipping plot_traj_rows: only {len(X_solvers)} optimal solver solutions.")
            else:
                x0_plot_batch = x_test[plot_indices[:4]].reshape(4, 1, nx).to(cpu_d)
                X_pred_batch = y_pred[plot_indices[:4], :T*nx].reshape(4, T, nx).to(cpu_d)
                X_pred_batch = torch.cat((x0_plot_batch, X_pred_batch), dim=1)
                U_pred_batch = y_pred[plot_indices[:4], T*nx:].reshape(4, T, nu).to(cpu_d)
                X_opt_batch = torch.stack(X_solvers[:4], dim=0).to(cpu_d)
                X_opt_batch = torch.cat((x0_plot_batch, X_opt_batch), dim=1)
                U_opt_batch = torch.stack(U_solvers[:4], dim=0).to(cpu_d)
                plot_traj_rows(
                    X_full_batch=X_pred_batch,
                    U_full_batch=U_pred_batch,
                    x_hat=x_hat,
                    Q=Q,
                    quad_ub=b_obs,
                    u_b=(v_min, v_max, omega_max, -omega_max),
                    X_full_opt=X_opt_batch,
                    U_full_opt=U_opt_batch,
                    save_folder=folder,
                    tag=f"_b{idx_batch}",
                )
    

    eq_max_msg = f"equality constraint max abs error: {dyn_violation_max_all}"
    eq_msg = f"equality constraint avg abs error: {dyn_eq_violation_sum/len(test_loader.dataset)}"
    x_ineq_max_msg = f"inequality constraint for x max abs error: {obstacle_violation_max_all}"
    x_ineq_msg = f"inequality constraint for x avg abs error: {obstacle_violation_sum/len(test_loader.dataset)}"
    u_ineq_max_msg = f"inequality constraint for u max abs error: {control_violation_max_all}"
    u_ineq_msg = f"inequality constraint for u avg abs error: {control_violation_sum/len(test_loader.dataset)}"

    print(eq_msg)
    print(eq_max_msg)
    print(x_ineq_msg)
    print(x_ineq_max_msg)
    print(u_ineq_msg)
    print(u_ineq_max_msg)

    with open(f"{folder}/details.txt", "a") as f:
        f.write(eq_msg + "\n")
        f.write(eq_max_msg + "\n")
        f.write(x_ineq_msg + "\n")
        f.write(x_ineq_max_msg + "\n")
        f.write(u_ineq_msg + "\n")
        f.write(u_ineq_max_msg + "\n")
        f.write(optimality_msg + "\n")
        f.write(optimality_max_msg + "\n")
        for msg in timing_msgs:
            f.write(msg + "\n")


if __name__ == "__main__":
    eval(sys.argv[1])
