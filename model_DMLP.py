# Incrementally dissipative MLP model
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, eps=1e-3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.eps = eps

        # Dimensions
        self.n = hidden_size
        self.m = input_size
        self.p = output_size
        self.q = hidden_size
        self.L = num_layers

        # Parameters
        self.C = nn.Parameter(torch.randn(self.p, self.n) * 0.01)
        self.Y = nn.Parameter(torch.randn(self.L * self.q + self.n, self.L * self.q + self.n) * 0.01)

        self.Wu = nn.Parameter(torch.randn(self.q, self.m) * 0.01)
        self.W1 = nn.Parameter(torch.randn(self.q, self.n) * 0.01)

        self.M = nn.Parameter(torch.randn(self.n, self.n) * 0.01)
        self.X3 = nn.Parameter(torch.randn(self.m, self.m) * 0.01)
        self.Y3 = nn.Parameter(torch.randn(self.m, self.m) * 0.01)
        self.Z3 = nn.Parameter(torch.randn(max(self.p - self.m, 0), self.m) * 0.01)

        self.bx = nn.Parameter(torch.randn(self.m) * 0.01)
        self.by = nn.Parameter(torch.randn(self.p) * 0.01)
        self.bz = nn.ParameterList([nn.Parameter(torch.randn(self.q) * 0.01) for _ in range(self.L)])

        self.gamma = 0.5

        self.W1_sub = nn.Parameter(torch.randn(max(self.L - 1, 1), self.q, self.q) * 0.01)
        self.B = nn.Parameter(torch.randn(self.n, self.q) * 0.01)

        self.alpha_diag_param = nn.Parameter(torch.zeros(self.L * self.n))

        # Optional stabilizer for deeper stacks 
        self.ln = nn.ModuleList([nn.LayerNorm(self.q) for _ in range(self.L)])

    def _make_QSR(self, device, dtype, ep1=1.0):
        """
        Generalize Q,S,R to avoid numpy. Keeps your structure but works for m,p.
        Q: (p,p)=0, R: (m,m)=0
        S: (m,p) with last m columns = 0.5*I if p>=m
        """
        p, m = self.p, self.m
        Q = torch.zeros(p, p, device=device, dtype=dtype)
        R = torch.zeros(m, m, device=device, dtype=dtype)

        S = torch.zeros(m, p, device=device, dtype=dtype)
        if p >= m:
            S[:, (p - m):p] = 0.5 * torch.eye(m, device=device, dtype=dtype)
        else:
            S[:p, :p] = 0.5 * torch.eye(p, device=device, dtype=dtype)

        Lq = ep1 * torch.eye(p, device=device, dtype=dtype)
        Q1 = -(Lq.T @ Lq)       
        Lr = 0.5 / ep1
        return Q, S, R, Lq, Q1, Lr

    def proj_W1form(self, H21, L, q):
        device, dtype = H21.device, H21.dtype
        n = H21.shape[1]
        W1_like = torch.zeros(L * q, n, device=device, dtype=dtype)
        W1_like[0:q, :] = H21[0:q, :]
        return W1_like

    def build_W_for_Lambda_minus_Wsym(self, H, L, n):
        """
        Build strictly block-lower W (only first subdiagonal blocks) so -W - W^T matches H off-diagonal.
        """
        W = torch.zeros_like(H)
        for i in range(1, L):
            r = slice(i * n, (i + 1) * n)
            c = slice((i - 1) * n, i * n)
            W[r, c] = -H[r, c]
        return W
    def _build_I2(self, device, dtype):
        """
        Build (L*q, L*q) matrix with only first subdiagonal blocks nonzero.
        For each subdiagonal block (i, i-1), set:
            mW2[i,i-1] = W1_sub[i-1] + gamma * I_q
        All other blocks are zero.

        Shape: (L*q, L*q)
        """
        L, q = self.L, self.q
        gamma = float(self.gamma)
        Iq = torch.eye(q, device=device, dtype=dtype)

        rows = []
        for i in range(L):
            row_blocks = []
            for j in range(L):
                if (i == j + 1) and (L > 1):
                    blk = Iq
                    row_blocks.append(blk)
                else:
                    row_blocks.append(torch.zeros(q, q, device=device, dtype=dtype))
            rows.append(torch.cat(row_blocks, dim=1))

        return torch.cat(rows, dim=0)
    def _build_mW2(self, device, dtype):
        """
        Build (L*q, L*q) matrix with only first subdiagonal blocks nonzero.
        For each subdiagonal block (i, i-1), set:
            mW2[i,i-1] = W1_sub[i-1] + gamma * I_q
        All other blocks are zero.

        Shape: (L*q, L*q)
        """
        L, q = self.L, self.q
        gamma = float(self.gamma)
        Iq = torch.eye(q, device=device, dtype=dtype)

        rows = []
        for i in range(L):
            row_blocks = []
            for j in range(L):
                if (i == j + 1) and (L > 1):
                    blk = self.W1_sub[j].to(device=device, dtype=dtype) 
                    row_blocks.append(blk)
                else:
                    row_blocks.append(torch.zeros(q, q, device=device, dtype=dtype))
            rows.append(torch.cat(row_blocks, dim=1))

        return torch.cat(rows, dim=0)
    def forward(self, u, x, batch_first=False):
        """
        u: (seq_len, m)
        x: (n,) or (1,n)
        returns: Y_out (seq_len,p), P (n,n), h (seq_len,n)
        """
        device, dtype = u.device, u.dtype
        n, m, q, p, L = self.n, self.m, self.q, self.p, self.L
        gamma, eps = self.gamma, self.eps

        if x.dim() == 1:
            x_cur = x.view(1, -1)
        else:
            x_cur = x

        # ---- Q,S,R, D22 (torch only) ----
        Q, S, R, Lq, Q1, Lr = self._make_QSR(device, dtype, ep1=1.0)

        Iq = torch.eye(q, device=device, dtype=dtype)
        Im = torch.eye(m, device=device, dtype=dtype)

        Mmat = self.X3.T @ self.X3 + (self.Y3 - self.Y3.T)
        if self.Z3.numel() > 0:
            Mmat = Mmat + self.Z3.T @ self.Z3
        Mmat = Mmat + eps * Im

        IplusM = Im + Mmat
        IminusM = Im - Mmat

        N_top = torch.linalg.solve(IplusM.T, IminusM.T).T

        if self.Z3.numel() > 0:
            Z = self.Z3.to(device=device, dtype=dtype)
            N_bot = -2.0 * (torch.linalg.solve(IplusM.T, Z.T).T)
            N = torch.cat([N_top, N_bot], dim=0)  # (p,m)
        else:
            N = N_top  # (m,m) when p==m

        D22 = S.T + N * Lr  # (p,m)

        # ---- Build WuNew and W1New with ONLY FIRST block nonzero  ----
        zeros_u = [torch.zeros_like(self.Wu) for _ in range(L - 1)]
        WuNew = torch.cat([self.Wu] + zeros_u, dim=0)  # (L*q, m)

        zeros_1 = [torch.zeros_like(self.W1) for _ in range(L - 1)]
        W1New = torch.cat([self.W1] + zeros_1, dim=0)  # (L*q, n)

        LambdaWu = gamma * WuNew

        # ---- H construction ----
        C = self.C.to(device=device, dtype=dtype)
        topW = C.T @ S.T                       # (n,m)
        W = torch.cat([topW, -LambdaWu], dim=0)  # (n+Lq, m)

        R1 = R + (S @ D22) + (D22.T @ S.T) + (D22.T @ Q @ D22)
        R1 = R1 + eps * torch.eye(m, device=device, dtype=dtype)

        I2n = torch.eye(L * q + n, device=device, dtype=dtype)
        WRinv = torch.linalg.solve(R1, W.T)  # (m, n+Lq)
        Ymat = self.Y.to(device=device, dtype=dtype)
        H = (Ymat @ Ymat.T) + (W @ WRinv) + eps * I2n

        # blocks
        H11 = H[:n, :n]
        H12 = H[:n, n:(L * q + n)]
        H21 = H12.T
        H22 = H[n:(L * q + n), n:(L * q + n)]

        # your projection + add W1New
        H21_star = self.proj_W1form(H21, L, q) + W1New
        delta12 = H21_star - H21
        delta = torch.linalg.norm(delta12, ord=2) + eps

        P = H11 - (C.T @ Q @ C) + delta * torch.eye(n, device=device, dtype=dtype)

        # P1
        B0 = torch.zeros(q, L * q, device=device, dtype=dtype)
        B0[:, (L - 1) * q:L * q] = torch.eye(q, device=device, dtype=dtype)

        P1 = (B0.T @ self.B.to(device=device, dtype=dtype).T @ P @ self.B.to(device=device, dtype=dtype) @ B0) \
             + H22 + delta * torch.eye(L * q, device=device, dtype=dtype)

        # LambdaW2
        mW2 = self._build_mW2(device, dtype)
        LambdaW2 = 0.5 * (self.build_W_for_Lambda_minus_Wsym(-P1, L, q) + mW2)

        residual = P1 + LambdaW2 + LambdaW2.T
        ILq = torch.eye(L * q, device=device, dtype=dtype)
        I2 = self._build_I2(device, dtype)
        lambda_max_bound = torch.linalg.norm(torch.linalg.inv(ILq-gamma*I2-gamma*I2.T)@residual, ord=2) + eps

        # ---- NEW alpha: diagonal learnable but ALWAYS >= lambda_max_bound ----
        sp = F.softplus(self.alpha_diag_param.to(device=device, dtype=dtype))
        alpha_diag = lambda_max_bound + sp + eps                 # (L*n,)
        alpha = torch.diag(alpha_diag)                           # (L*n,L*n)

        # Solve instead of inv: alpha X = (...)
        Wu = (1.0 / gamma) * torch.linalg.solve(alpha, LambdaWu)        # (Lq,m)
        W1 = (-1.0 / gamma) * torch.linalg.solve(alpha, H21_star)       # (Lq,n)
        W2 = (1.0 / gamma) * torch.linalg.solve(alpha, LambdaW2)        # (Lq,Lq)

        # Effective per-layer blocks
        W2_eff = [torch.zeros(q, q, device=device, dtype=dtype)]
        if L > 1:
            for l in range(1, L):
                W2_eff.append(W2[l * q:(l + 1) * q, (l - 1) * q:l * q])

        W1_eff = [W1[l * q:(l + 1) * q, :] for l in range(L)]
        Wu_eff = [Wu[l * q:(l + 1) * q, :] for l in range(L)]

        # ---- rollout ----
        seq_len = u.shape[0]
        x_list, h_list = [], []

        for t in range(seq_len):
            z_layers = []
            for l in range(L):
                prev = torch.zeros_like(x_cur) if l == 0 else z_layers[l - 1]

                pre = (u[t:t+1, :] @ Wu_eff[l].T) \
                      + (x_cur @ W1_eff[l].T) \
                      + (prev @ W2_eff[l].T)  + prev + self.bz[l].view(1, -1)
                # pre = self.ln[l](pre) # for layer norm
                z_new = torch.tanh(pre)
                # z_new = torch.relu(pre) 

                z_layers.append(z_new)

            x_next = z_layers[-1] @ self.B.to(device=device, dtype=dtype).T + self.bx.view(1, -1)
            
            x_list.append(x_cur)
            h_list.append(x_next)
            x_cur = x_next

        X_tm1 = torch.cat(x_list, dim=0)  # (seq_len,n)
        h = torch.cat(h_list, dim=0)      # (seq_len,n)

        Y_out = (X_tm1 @ C.T) + (u @ D22.T) + self.by.view(1, -1).expand(seq_len, p)
        return Y_out, P, h
