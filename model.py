import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self,model_params):
        super(Model,self).__init__()

        self.constrained = model_params["constrained"]
        self.n_in = model_params["n_in"]
        self.nx = model_params["nx"]
        self.nu = model_params["nu"]
        self.quad_lb = torch.tensor(model_params["b_obs"],dtype=torch.float32)
        self.vmin = model_params["v_min"]
        self.vmax = model_params["v_max"]
        self.omega_max = model_params["omega_max"]
        self.T = model_params["T"]
        self.n_out = self.T*(self.nx+self.nu)
        self.dt = model_params["dt"]

        self.l1 = nn.Linear(self.n_in,200)
        self.l2 = nn.Linear(200,200)
        self.l3 = nn.Linear(200,self.n_out)
        self.activation = nn.ReLU()

        if model_params["constrained"]:
            print('INITIALIZING CONSTRAINED MODEL')
            self.n_proj = model_params["n_proj"]
            self.register_buffer("Q", torch.tensor(model_params["Q"],dtype=torch.float32))
            self.register_buffer("Qsym", self.Q + self.Q.T)
            obstacle_center = model_params.get("obstacle_center", np.zeros(self.Q.shape[0], dtype=np.float32))
            obstacle_center = np.asarray(obstacle_center, dtype=np.float32).reshape(-1)
            self.register_buffer("obstacle_center", torch.tensor(obstacle_center, dtype=torch.float32))
            self.obs_dim = int(self.Q.shape[0])
            self.register_buffer("epsilon", torch.tensor(model_params["epsilon"],dtype=torch.float32))

            nvars = self.T * (self.nx + self.nu)
            self.n_eq = 3 
            self.n_ineq = 3
            lb = torch.zeros(self.n_ineq*self.T,1)
            lb[:self.T] = torch.ones(self.T,1)*self.quad_lb
            lb[self.T:self.T*2] = torch.ones(self.T,1)*self.vmin
            lb[self.T*2:self.T*3] = torch.ones(self.T,1)*(-self.omega_max)
            self.register_buffer("lb",torch.cat((torch.zeros(self.n_eq*self.T,1), lb), dim=0))
            ub = torch.zeros(self.n_ineq*self.T,1)
            ub[:self.T] = torch.ones(self.T,1) * float("inf")
            ub[self.T:self.T*2] = torch.ones(self.T,1)*self.vmax
            ub[self.T*2:self.T*3] = torch.ones(self.T,1)*(self.omega_max)
            self.register_buffer("ub",torch.cat((torch.zeros(self.n_eq*self.T,1), ub), dim=0))
            self.register_buffer("J_template",torch.zeros((self.n_eq+self.n_ineq)*self.T,nvars))
    
    def HN_pp(self,x,x0):
        epsilon = self.epsilon
        device = x.device
        batch = x.shape[0]
        T = self.T
        nx = self.nx
        nu = self.nu
        n_eq = self.n_eq * T
        n_ineq = self.n_ineq * T
        ineq_lb = self.lb[n_eq:].reshape(1, -1).expand(batch, -1)
        ineq_ub = self.ub[n_eq:].reshape(1, -1).expand(batch, -1)
        I = torch.eye(n_eq + n_ineq, device=device).unsqueeze(0)
        time_idx = torch.arange(T, device=device, dtype=torch.long)
        eq_row = self.n_eq * time_idx
        state_col = nx * time_idx
        u_col = nx * T + nu * time_idx
        obs_row = n_eq + time_idx
        v_row = n_eq + T + time_idx
        omega_row = n_eq + 2 * T + time_idx
        obs_col_offsets = torch.arange(self.obs_dim, device=device, dtype=torch.long)
        obs_rows = obs_row.unsqueeze(1).expand(-1, self.obs_dim)
        obs_cols = state_col.unsqueeze(1) + obs_col_offsets.unsqueeze(0)

        for i in range(self.n_proj):
            # vars go x1, y1, theta1, ..., xT, yT, thetaT, v0, omega0, ..., v_{T-1}, omega_{T-1}
            # equality constraints go c1_t1, c2_t1, c3_t1, ..., c1_tT, c2_tT, c3_tT
            # inequality constraints go c_obs_t1, ..., c_obs_tT, c_v_t1, ..., c_v_tT, c_w_t1, ..., c_w_tT
            # x0 is fixed input data, not part of the decision vector.
            J = self.J_template.unsqueeze(0).expand(batch, -1, -1).clone()

            X = x[:, : nx * T].reshape(batch, T, nx)
            U = x[:, nx * T :].reshape(batch, T, nu)

            prev_x = torch.cat((x0[:, 0:1], X[:, :-1, 0]), dim=1)
            prev_y = torch.cat((x0[:, 1:2], X[:, :-1, 1]), dim=1)
            prev_theta = torch.cat((x0[:, 2:3], X[:, :-1, 2]), dim=1)
            cos_prev = torch.cos(prev_theta)
            sin_prev = torch.sin(prev_theta)

            # equality constraints 
            eq_x = -X[:, :, 0] + prev_x + self.dt * U[:, :, 0] * cos_prev
            eq_y = -X[:, :, 1] + prev_y + self.dt * U[:, :, 0] * sin_prev
            eq_theta = -X[:, :, 2] + prev_theta + self.dt * U[:, :, 1]
            c_eq = torch.stack((eq_x, eq_y, eq_theta), dim=2).reshape(batch, n_eq)

            # inequality constraints
            obs_states = X[:, :, : self.obs_dim]
            obs_diff = obs_states - self.obstacle_center.view(1, 1, -1)
            obs_grad = torch.matmul(obs_diff, self.Qsym.T)
            c_obs = torch.einsum("bti,ij,btj->bt", obs_diff, self.Q, obs_diff)
            c_v = U[:, :, 0]
            c_omega = U[:, :, 1]
            c_ineq = torch.cat((c_obs, c_v, c_omega), dim=1)

            # first-step dynamics rows: -x_1 + f(x0, u0) = 0 (x0 is fixed)
            J[:, eq_row, state_col] = -1.0
            J[:, eq_row + 1, state_col + 1] = -1.0
            J[:, eq_row + 2, state_col + 2] = -1.0
            J[:, eq_row, u_col] = self.dt * cos_prev
            J[:, eq_row + 1, u_col] = self.dt * sin_prev
            J[:, eq_row + 2, u_col + 1] = self.dt

            # remaining dynamics rows: -x_{k+1} + f(x_k, u_k) = 0 for k = 1, ..., T-1
            if T > 1:
                dyn_row = eq_row[1:]
                prev_state_col = state_col[:-1]
                prev_v = U[:, 1:, 0]
                prev_cos = cos_prev[:, 1:]
                prev_sin = sin_prev[:, 1:]

                J[:, dyn_row, prev_state_col] = 1.0
                J[:, dyn_row + 1, prev_state_col + 1] = 1.0
                J[:, dyn_row + 2, prev_state_col + 2] = 1.0
                J[:, dyn_row, prev_state_col + 2] = -self.dt * prev_v * prev_sin
                J[:, dyn_row + 1, prev_state_col + 2] = self.dt * prev_v * prev_cos

            # inequality rows
            J[:, obs_rows, obs_cols] = obs_grad
            J[:, v_row, u_col] = 1.0
            J[:, omega_row, u_col + 1] = 1.0

            v_eq = c_eq
            v_ineq = torch.relu(c_ineq - ineq_ub) - torch.relu(ineq_lb - c_ineq)
            v = torch.cat((v_eq, v_ineq), dim=1)

            correction = epsilon * I

            JJt = J @ J.transpose(1, 2) + correction
            # numerical cleanup
            JJt = 0.5 * (JJt + JJt.transpose(1, 2))

            L = torch.linalg.cholesky(JJt)
            y = torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1)
            dx = (J.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)
            x = x - dx

        return x 

    def forward(self,x):
        x0 = x
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        if self.constrained:
            x = self.HN_pp(x,x0)

        return x
