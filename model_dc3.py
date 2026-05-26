import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self, model_params):
        super(Model, self).__init__()

        self.n_in = model_params["n_in"]
        self.nx = model_params["nx"]
        self.nu = model_params["nu"]
        self.T = model_params["T"]
        self.dt = model_params["dt"]
        self.n_out = self.T * (self.nx + self.nu)
        self.partial_out = self.T * self.nu

        self.vmin = float(model_params["v_min"])
        self.vmax = float(model_params["v_max"])
        self.omega_max = float(model_params["omega_max"])
        self.register_buffer("quad_lb", torch.tensor(model_params["b_obs"], dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(model_params["Q"], dtype=torch.float32))
        self.register_buffer("Qsym", self.Q + self.Q.T)
        obstacle_center = model_params.get("obstacle_center", np.zeros(self.Q.shape[0], dtype=np.float32))
        obstacle_center = np.asarray(obstacle_center, dtype=np.float32).reshape(-1)
        self.register_buffer("obstacle_center", torch.tensor(obstacle_center, dtype=torch.float32))
        self.obs_dim = int(self.Q.shape[0])

        self.use_dc3 = bool(model_params["dc3"])
        self.dc3_steps = int(model_params["dc3_steps"])
        self.dc3_stepsize = float(model_params["dc3_stepsize"])
        self.dc3_momentum = float(model_params["dc3_momentum"])

        self.l1 = nn.Linear(self.n_in, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, self.partial_out)
        self.activation = nn.ReLU()

    def rollout(self, U, x0):
        states = []
        prev = x0
        for k in range(self.T):
            next_state = torch.stack(
                (
                    prev[:, 0] + self.dt * U[:, k, 0] * torch.cos(prev[:, 2]),
                    prev[:, 1] + self.dt * U[:, k, 0] * torch.sin(prev[:, 2]),
                    prev[:, 2] + self.dt * U[:, k, 1],
                ),
                dim=1,
            )
            states.append(next_state)
            prev = next_state
        return torch.stack(states, dim=1)

    def complete_partial(self, x0, U_flat):
        batch = U_flat.shape[0]
        U = U_flat.reshape(batch, self.T, self.nu)
        X = self.rollout(U, x0)
        return torch.cat((X.reshape(batch, self.T * self.nx), U_flat), dim=1)

    def split_full(self, Y):
        batch = Y.shape[0]
        X = Y[:, : self.T * self.nx].reshape(batch, self.T, self.nx)
        U = Y[:, self.T * self.nx :].reshape(batch, self.T, self.nu)
        return X, U

    def ineq_partial_grad(self, x0, Y):
        X, U = self.split_full(Y)
        batch = U.shape[0]

        obs_diff = X[:, :, : self.obs_dim] - self.obstacle_center.view(1, 1, -1)
        obs_val = torch.einsum("bti,ij,btj->bt", obs_diff, self.Q, obs_diff)
        obs_dist = torch.relu(self.quad_lb.to(dtype=Y.dtype, device=Y.device) - obs_val)
        obs_grad = -2.0 * obs_dist.unsqueeze(-1) * torch.matmul(obs_diff, self.Qsym.T)

        state_grad = torch.zeros_like(X)
        state_grad[:, :, : self.obs_dim] = obs_grad

        v_min = torch.tensor(self.vmin, dtype=Y.dtype, device=Y.device)
        v_max = torch.tensor(self.vmax, dtype=Y.dtype, device=Y.device)
        omega_max = torch.tensor(self.omega_max, dtype=Y.dtype, device=Y.device)

        v_lower = torch.relu(v_min - U[:, :, 0])
        v_upper = torch.relu(U[:, :, 0] - v_max)
        omega_lower = torch.relu(-omega_max - U[:, :, 1])
        omega_upper = torch.relu(U[:, :, 1] - omega_max)

        control_grad = torch.zeros_like(U)
        control_grad[:, :, 0] = -2.0 * v_lower + 2.0 * v_upper
        control_grad[:, :, 1] = -2.0 * omega_lower + 2.0 * omega_upper

        prev_theta = torch.cat((x0[:, 2:3], X[:, :-1, 2]), dim=1)
        cos_prev = torch.cos(prev_theta)
        sin_prev = torch.sin(prev_theta)

        grad_U = torch.zeros_like(U)
        adj_next = torch.zeros(batch, self.nx, dtype=Y.dtype, device=Y.device)
        for k in range(self.T - 1, -1, -1):
            adj = state_grad[:, k, :] + adj_next
            grad_U[:, k, 0] = control_grad[:, k, 0] + self.dt * (
                adj[:, 0] * cos_prev[:, k] + adj[:, 1] * sin_prev[:, k]
            )
            grad_U[:, k, 1] = control_grad[:, k, 1] + self.dt * adj[:, 2]

            if k > 0:
                v_k = U[:, k, 0]
                adj_prev = torch.zeros_like(adj)
                adj_prev[:, 0] = adj[:, 0]
                adj_prev[:, 1] = adj[:, 1]
                adj_prev[:, 2] = (
                    -self.dt * v_k * sin_prev[:, k] * adj[:, 0]
                    + self.dt * v_k * cos_prev[:, k] * adj[:, 1]
                    + adj[:, 2]
                )
                adj_next = adj_prev

        Y_grad = torch.zeros_like(Y)
        Y_grad[:, self.T * self.nx :] = grad_U.reshape(batch, self.T * self.nu)
        return Y_grad

    def grad_steps(self, x0, Y, num_steps):
        Y_new = Y
        old_Y_step = 0.0
        for _ in range(num_steps):
            Y_step = self.ineq_partial_grad(x0, Y_new)
            new_Y_step = self.dc3_stepsize * Y_step + self.dc3_momentum * old_Y_step
            Y_new = Y_new - new_Y_step
            U_flat = Y_new[:, self.T * self.nx :]
            Y_new = self.complete_partial(x0, U_flat)
            old_Y_step = new_Y_step
        return Y_new

    def forward(self, x):
        x0 = x
        U = self.l1(x)
        U = self.activation(U)
        U = self.l2(U)
        U = self.activation(U)
        U = self.l3(U)

        Y = self.complete_partial(x0, U)

        if self.use_dc3:
            Y = self.grad_steps(x0, Y, self.dc3_steps)

        return Y
