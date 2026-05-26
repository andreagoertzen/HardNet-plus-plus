import torch
from utils import create_full_trajectory_dataloaders
import numpy as np 
import os
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime
from pathlib import Path

def build_model_params_from_dataset(data, constrained, train_params):
    q_obs = data["Q"]
    b_obs = float(np.asarray(data["b_obs"]).reshape(-1)[0])

    obstacle_center = np.asarray(data["obstacle_center"]).reshape(-1)
    T_out = train_params["T_out"]

    model_params = {
        "n_in": int(data["x0_samples"].shape[1]),
        "nx": int(np.asarray(data["nx"]).reshape(-1)[0]),
        "nu": int(np.asarray(data["nu"]).reshape(-1)[0]),
        "T": T_out,
        "dt": float(np.asarray(data["dt"]).reshape(-1)[0]),
        "Q": q_obs,
        "b_obs": b_obs,
        "v_min": float(np.asarray(data["v_min"]).reshape(-1)[0]),
        "v_max": float(np.asarray(data["v_max"]).reshape(-1)[0]),
        "omega_max": float(np.asarray(data["omega_max"]).reshape(-1)[0]),
        "obstacle_center": obstacle_center,
        "constrained": constrained,
        "n_proj": train_params["n_proj"],
        "epsilon": train_params["epsilon"],
        "dc3": train_params["dc3"]
    }
    if train_params["dc3"]:
        model_params.update({
        "dc3_steps": train_params["dc3_steps"],
        "dc3_stepsize": train_params["dc3_stepsize"],
        "dc3_momentum": train_params["dc3_momentum"]
        })
    return model_params

def compute_constraint_violations(z, x0, model_params):
    batch_size = z.shape[0]
    T = model_params["T"]
    nx = model_params["nx"]
    nu = model_params["nu"]
    dt = model_params["dt"]

    x_seq = z[:, : T * nx].reshape(batch_size, T, nx)
    u_seq = z[:, T * nx :].reshape(batch_size, T, nu)

    prev_x = torch.cat((x0[:, 0:1], x_seq[:, :-1, 0]), dim=1)
    prev_y = torch.cat((x0[:, 1:2], x_seq[:, :-1, 1]), dim=1)
    prev_theta = torch.cat((x0[:, 2:3], x_seq[:, :-1, 2]), dim=1)

    dyn_x = x_seq[:, :, 0] - (prev_x + dt * u_seq[:, :, 0] * torch.cos(prev_theta))
    dyn_y = x_seq[:, :, 1] - (prev_y + dt * u_seq[:, :, 0] * torch.sin(prev_theta))
    dyn_theta = x_seq[:, :, 2] - (prev_theta + dt * u_seq[:, :, 1])
    dyn_residual = torch.stack((dyn_x, dyn_y, dyn_theta), dim=2)

    q_obs = torch.tensor(model_params["Q"], dtype=z.dtype, device=z.device)
    obstacle_center = torch.tensor(model_params["obstacle_center"], dtype=z.dtype, device=z.device).view(1, 1, -1)
    b_obs = torch.tensor(model_params["b_obs"], dtype=z.dtype, device=z.device)
    obs_diff = x_seq[:, :, : q_obs.shape[0]] - obstacle_center
    obs_val = torch.einsum("bti,ij,btj->bt", obs_diff, q_obs, obs_diff) - b_obs

    v_min = torch.tensor(model_params["v_min"], dtype=z.dtype, device=z.device)
    v_max = torch.tensor(model_params["v_max"], dtype=z.dtype, device=z.device)
    omega_max = torch.tensor(model_params["omega_max"], dtype=z.dtype, device=z.device)

    v_lower = torch.relu(v_min - u_seq[:, :, 0])
    v_upper = torch.relu(u_seq[:, :, 0] - v_max)
    omega_lower = torch.relu(-omega_max - u_seq[:, :, 1])
    omega_upper = torch.relu(u_seq[:, :, 1] - omega_max)
    control_violation = v_lower + v_upper + omega_lower + omega_upper
    obstacle_violation = torch.relu(-obs_val)

    return dyn_residual, obstacle_violation, control_violation


def train(params,save_dir):
    lamb = params["lambda"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### LOAD DATA
    script_dir = Path(__file__).resolve().parent
    npz_path = script_dir / params["npz_path"]
    data = np.load(npz_path, allow_pickle=True)
    train_loader, val_loader, test_loader = create_full_trajectory_dataloaders(
        data,
        batch_size=1000,
    )

    model_params = build_model_params_from_dataset(data, params["constrained"], params)

    T_out = model_params["T"]
    nx = model_params["nx"]
    nu = model_params["nu"]
    print(model_params["Q"])
    if params["dc3"]:
        folder = f'{save_dir}/dc3_{model_params["dc3_steps"]}iterates_Tout{T_out}_lr{params["lr"]}_lambda{lamb}_dc3stepsize{model_params["dc3_stepsize"]}_momentum{model_params["dc3_momentum"]}'
    elif model_params["constrained"]:
        folder = f'{save_dir}/{model_params["n_proj"]}iterates_Tout{T_out}_lr{params["lr"]}_lambda{lamb}_eps{model_params["epsilon"]}'
    else:
        folder = f'{save_dir}/unconstrained_model_lr{params["lr"]}_lambda{lamb}'
    print(folder)

    if not os.path.exists(folder):
            os.makedirs(folder)
    np.savez(f"{folder}/model_params.npz", **model_params)
    n_epochs = params["epochs"]

    # initialize model
    if params["dc3"]:
        print('Training with DC3 Model')
        from model_dc3 import Model
    else:
        from model import Model
    model = Model(model_params).to(device)
    lr = params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    control_violations = []
    dyn_eq_violations = []
    obstacle_violations = []
    best_val_loss = float("inf")
    best_ckpt_path = f"{folder}/best_model.pt"

    fig, ax = plt.subplots(figsize=(7, 4))
    fig_cons, ax_cons = plt.subplots(figsize=(7, 4))
    train_start = time.perf_counter()
    x_hat = torch.tensor(data["x_hat"], dtype=torch.float32, device=device).reshape(nx)
    Q_stage = torch.tensor(data["Q_stage"], dtype=torch.float32, device=device)
    Q_terminal = torch.tensor(data["Q_terminal"], dtype=torch.float32, device=device)
    R = torch.tensor(data["R"], dtype=torch.float32, device=device)

    x_batch = None
    y_batch = None
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            # if x_batch is None:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            batch_size = x_batch.shape[0]
            train_count += batch_size

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x_batch)

            x_seq = y_pred[:, :T_out * nx].reshape(batch_size, T_out, nx)
            u_seq = y_pred[:, T_out * nx:].reshape(batch_size, T_out, nu)
            err = x_seq - x_hat.view(1, 1, nx)
            theta_err = torch.atan2(torch.sin(err[:, :, 2]), torch.cos(err[:, :, 2]))
            err = torch.cat((err[:, :, :2], theta_err.unsqueeze(2)), dim=2)
            state_cost = torch.einsum("bti,ij,btj->b", err, Q_stage, err)
            terminal_cost = torch.einsum("bi,ij,bj->b", err[:, -1], Q_terminal, err[:, -1])
            control_cost = torch.einsum("bti,ij,btj->b", u_seq, R, u_seq)
            dyn_residual, obstacle_violation, control_violation = compute_constraint_violations(y_pred, x_batch, model_params)
            if lamb>0:
                reg_term = lamb * (torch.linalg.norm(dyn_residual.reshape(batch_size, -1), dim=1) + torch.linalg.norm(obstacle_violation.reshape(batch_size, -1), dim=1) + torch.linalg.norm(control_violation.reshape(batch_size, -1), dim=1))
            else:
                reg_term = 0
            loss_per_sample = state_cost + terminal_cost + control_cost + reg_term
            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_size

        train_losses.append(train_loss_sum / train_count)

        # evaluate model and constraint violations between batches
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            control_violation_sum = 0.0
            dyn_eq_violation_sum = 0.0
            obstacle_violation_sum = 0.0

            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                y_pred = model(x_batch)

                batch_size = x_batch.shape[0]
                val_count += batch_size

                x_seq = y_pred[:, :T_out * nx].reshape(batch_size, T_out, nx)
                u_seq = y_pred[:, T_out * nx:].reshape(batch_size, T_out, nu)
                err = x_seq - x_hat.view(1, 1, nx)
                theta_err = torch.atan2(torch.sin(err[:, :, 2]), torch.cos(err[:, :, 2]))
                err = torch.cat((err[:, :, :2], theta_err.unsqueeze(2)), dim=2)
                state_cost = torch.einsum("bti,ij,btj->b", err, Q_stage, err)
                terminal_cost = torch.einsum("bi,ij,bj->b", err[:, -1], Q_terminal, err[:, -1])
                control_cost = torch.einsum("bti,ij,btj->b", u_seq, R, u_seq)
                val_loss_per_sample = state_cost + terminal_cost + control_cost
                val_loss = val_loss_per_sample.mean()

                val_loss_sum += val_loss.item() * batch_size

                dyn_residual, obstacle_violation, control_violation = compute_constraint_violations(
                    y_pred, x_batch, model_params
                )

                dyn_eq_violation_sum += dyn_residual.abs().mean().item() * batch_size
                obstacle_violation_sum += obstacle_violation.mean().item() * batch_size
                control_violation_sum += control_violation.mean().item() * batch_size

        val_losses.append(val_loss_sum / val_count)
        control_violations.append(control_violation_sum / val_count)
        dyn_eq_violations.append(dyn_eq_violation_sum / val_count)
        obstacle_violations.append(obstacle_violation_sum / val_count)

        current_val = val_losses[-1]#.item()
        if current_val < best_val_loss:
            best_val_loss = current_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_ckpt_path,
            )
            print(f"Saved new best model at epoch {epoch} with val_loss={best_val_loss:.6e}")

        ax.clear()
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_yscale("log")
        ax.set_title("Train and Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        fig.savefig(f"{folder}/train_val_loss.png", dpi=200)

        ax_cons.clear()
        ax_cons.plot(control_violations, label="control bounds violation")
        ax_cons.plot(dyn_eq_violations, label="dynamics violation")
        ax_cons.plot(obstacle_violations, label="obstacle violation")
        ax_cons.set_xlabel("Epoch")
        ax_cons.set_ylabel("Mean violation")
        ax_cons.set_yscale("log")
        ax_cons.set_title("Constraint Violations (Validation)")
        ax_cons.grid(True, alpha=0.3)
        ax_cons.legend(loc='upper right')
        fig_cons.savefig(f"{folder}/constraint_violations.png", dpi=200)

    train_time_sec = time.perf_counter() - train_start
    with open(f"{folder}/details.txt", "w") as f:
        f.write(f"Total training time: {train_time_sec:.2f} s ({train_time_sec/60:.2f} min)\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='specify number of epochs', default=500)
    parser.add_argument('--T_out', type=int, help='specify trajectory length of MPC problem', default=10)
    parser.add_argument('--constrained', action='store_true', help='specify whether the HardNet++ projection layer is active')
    parser.add_argument('--n_proj', type=int, help='specify number projection iterations', default=500)
    parser.add_argument('--epsilon', type=float, help='specify regularization factor', default=0.3)
    parser.add_argument('--lr',type=float, help='specify learning rate', default=1e-4)
    parser.add_argument('--lambda',type=float,help='regularization parameter',default=0.0)
    parser.add_argument('--npz_path', type=str, help='dataset path relative to obst_avoid/', default='mpc_dataset.npz')
    parser.add_argument('--dc3', action='store_true', help='use DC3 equality completion and inequality correction')
    parser.add_argument('--dc3_steps', type=int, help='number of DC3 correction steps', default=100)
    parser.add_argument('--dc3_stepsize', type=float, help='DC3 inequality correction step size', default=1e-1)
    parser.add_argument('--dc3_momentum', type=float, help='DC3 inequality correction momentum', default=0)

    args = parser.parse_args()
    params = vars(args)
    now = datetime.now()
    save_time_str = now.strftime("%m%d_%H")
    save_dir = 'Trained_Models/' + save_time_str
    model = train(params,save_dir)
