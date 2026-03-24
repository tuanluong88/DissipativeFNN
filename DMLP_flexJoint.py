#############
# ============================================================
# Train FNN on 1-DOF flexible-joint (hybrid) manipulator dataset
# with incremental dissipativity model

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import model_DMLP as FNN


# =====================================
# 0. Seeds for repeatability
# =====================================
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------
# Plot helpers
# -------------------------------------
def plot_incremental_dissipativity(t_axis, w_inc, deltaV, w_minus_deltaV, title_suffix=""):
    t_axis = np.asarray(t_axis).reshape(-1)
    w_inc = np.asarray(w_inc).reshape(-1)
    deltaV = np.asarray(deltaV).reshape(-1)
    w_minus_deltaV = np.asarray(w_minus_deltaV).reshape(-1)

    # 1) main condition: w_inc - deltaV >= 0
    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, w_minus_deltaV, label=r"$w_{\mathrm{inc}}(k) - \Delta V(k)$")
    plt.axhline(0.0, linewidth=1.5, linestyle="--", label="0 boundary")
    plt.xlabel("Time [s]")
    plt.ylabel("Margin")
    plt.grid(True)
    plt.legend()
    plt.title("Incremental dissipativity check " + title_suffix)
    plt.tight_layout()
    plt.show()

def plot_results_flex(t_axis, q_true, th_true, thd_true, q_pred, th_pred, thd_pred):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, q_true, label="True q(t)")
    plt.plot(t_axis, q_pred, "--", label="Pred q(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("q [rad]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, th_true, label="True θ(t)")
    plt.plot(t_axis, th_pred, "--", label="Pred θ(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("θ [rad]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, thd_true, label="True θ̇(t)")
    plt.plot(t_axis, thd_pred, "--", label="Pred θ̇(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("θ̇ [rad/s]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def append_stats_txt(filename, mean_error, std_error, mean_time, std_time):
    with open(filename, "a") as f:
        f.write("\n")
        f.write("# =====================================\n")
        f.write("# Evaluation statistics\n")
        f.write(f"# mean_error = {mean_error:.8e}\n")
        f.write(f"# std_error  = {std_error:.8e}\n")
        f.write(f"# mean_time  = {mean_time:.8e}\n")
        f.write(f"# std_time   = {std_time:.8e}\n")
        f.write("# =====================================\n")
    print(f"Appended statistics to {filename}")

# -------------------------------------
# ADDED: append training loss history
# -------------------------------------
def append_training_loss_txt(filename, loss_hist, best_loss=None, every=1):
    """
    Append training loss history at the end of the txt file.
    loss_hist: list/array length num_epochs (float)
    best_loss: optional float
    every: write every k-th epoch (use 10/50 to reduce file size)
    """
    loss_hist = np.asarray(loss_hist, dtype=float).reshape(-1)

    with open(filename, "a") as f:
        f.write("\n")
        f.write("# =====================================\n")
        f.write("# Training loss history\n")
        if best_loss is not None:
            f.write(f"# best_train_loss = {float(best_loss):.8e}\n")
        f.write("# columns: epoch  train_loss\n")
        for ep, L in enumerate(loss_hist, start=1):
            if every > 1 and (ep % every) != 0 and ep != 1 and ep != len(loss_hist):
                continue
            f.write(f"{ep:d}  {float(L):.8e}\n")
        f.write("# =====================================\n")

    print(f"Appended training loss history to {filename}")

# -------------------------------------
# Load dataset
# -------------------------------------
data_path = f"flexjoint_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Cannot find {data_path}. Run the flex1dof generation code first to create it."
    )

data = np.loadtxt(data_path, delimiter=",", skiprows=1)
t   = data[:, 0]
q   = data[:, 1]
qd  = data[:, 2]
th  = data[:, 3]
thd = data[:, 4]
tau = data[:, 5]

# Input u = tau, output y = [q, th, thd]
u_all = tau.reshape(-1, 1)                      # (T,1)
y_all = np.stack([q, th, thd], axis=1)          # (T,3)

N = t.shape[0]
N_train = int(0.5 * N)

u_train = u_all[:N_train, :]
y_train = y_all[:N_train, :]

u_test = u_all
y_test = y_all
t_test = t

# -------------------------------------
# Settings
# -------------------------------------
load_model = False
input_size, hidden_size, output_size, num_layers = 1, 32, 3, 2
learning_rate = 1e-3
model_name = "DMLP_1dof.pt"
num_epochs = 20000
NUM_ITER = 1

out_txt = "DMLP_1dof.txt"

# -------------------------------------
# Convert to tensors
# -------------------------------------
u_seq  = torch.tensor(u_train, dtype=torch.float32, device=device)  # (Ttrain,1)
y_seq  = torch.tensor(y_train, dtype=torch.float32, device=device)  # (Ttrain,3)
x_Test = torch.tensor(u_test,  dtype=torch.float32, device=device)  # (T,1)

noise_std = 0.0
y_seq = y_seq + torch.randn_like(y_seq) * noise_std

# -------------------------------------
# Model / optimizer
# -------------------------------------
model = FNN.FNNModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x0 = torch.zeros(1, hidden_size, device=device)

# -------------------------------------
# Optional load
# -------------------------------------
if load_model and os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_train, _, _ = model(u_seq, x0, batch_first=True)
        y_pred_train_2d = y_pred_train.squeeze(0) if y_pred_train.ndim == 3 else y_pred_train
        loss0 = criterion(y_pred_train_2d, y_seq)
        print("Initial loaded model train loss:", float(loss0.item()))

# -------------------------------------
# Training
# -------------------------------------
error = []
runtime = []
loss_hist = []   # <-- ADDED

for _ in range(NUM_ITER):
    best_loss = np.inf
    start_time = time.time()
    loss_hist = []  # reset per run

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred, _, _ = model(u_seq, x0, batch_first=True)
        y_pred_2d = y_pred.squeeze(0) if y_pred.ndim == 3 else y_pred

        loss = criterion(y_pred_2d, y_seq)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_hist.append(loss_val)   # <-- ADDED

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), model_name)
            print(f"Epoch {epoch:5d} | Saving best model | Loss: {loss_val:.6e}")
        elif epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | Loss: {loss_val:.6e}")

    total_time = time.time() - start_time
    runtime.append(total_time)
    print(f"Best loss: {best_loss:.6e}")
    print(f"Training time (s): {total_time:.2f}")

    # -------------------------------------
    # Evaluation on full horizon (traj1)
    # -------------------------------------
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred_t, P_last, _ = model(x_Test, x0, batch_first=True)
        y_pred_np = (y_pred_t.squeeze(0) if y_pred_t.ndim == 3 else y_pred_t).cpu().numpy()

    plot_results_flex(
        t_test,
        y_test[:, 0], y_test[:, 1], y_test[:, 2],
        y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2]
    )

    err = np.mean(np.abs(y_pred_np[N_train:, :] - y_test[N_train:, :]))
    error.append(err)

mean_error = float(np.mean(error))
std_error  = float(np.std(error))
mean_time  = float(np.mean(runtime))
std_time   = float(np.std(runtime))
print(f"mean abs error = {mean_error:.6e}, std = {std_error:.6e}")
print(f"mean runtime   = {mean_time:.6f}s, std = {std_time:.6f}s")

# ============================================================
# Incremental dissipativity evaluation (2 predicted trajectories)
# ============================================================

Q = np.array([[0.0]], dtype=np.float32)
S = np.array([[0.5]], dtype=np.float32)
R = np.array([[0.0]], dtype=np.float32)

Q_t = torch.tensor(Q, device=device)
S_t = torch.tensor(S, device=device)
R_t = torch.tensor(R, device=device)

M_top = torch.cat([Q_t, S_t.T], dim=1)
M_bot = torch.cat([S_t, R_t], dim=1)
Mblk  = torch.cat([M_top, M_bot], dim=0)

uk_tensor = torch.tensor(u_test, dtype=torch.float32, device=device)  # (T,1)
Tlen = uk_tensor.shape[0]

eps_pert = 10.0
uk2_tensor = uk_tensor + eps_pert * torch.randn_like(uk_tensor)

yPred1 = torch.zeros((Tlen, output_size), device=device)
yPred2 = torch.zeros((Tlen, output_size), device=device)
h_list, h2_list, P_list = [], [], []

with torch.no_grad():
    h_t = x0.clone()
    for k in range(Tlen):
        inp_k = uk_tensor[k:k+1, :]
        y_k, P_k, h_t = model(inp_k, h_t)
        yPred1[k, :] = y_k.view(-1)
        h_list.append(h_t.detach().view(-1).cpu().numpy())
        P_list.append(P_k.detach().cpu())

with torch.no_grad():
    h2_t = x0.clone()
    for k in range(Tlen):
        inp2_k = uk2_tensor[k:k+1, :]
        y2_k, _, h2_t = model(inp2_k, h2_t)
        yPred2[k, :] = y2_k.view(-1)
        h2_list.append(h2_t.detach().view(-1).cpu().numpy())

P_t = torch.as_tensor(P_list[-1], dtype=torch.float32, device=device)
h_torch  = torch.tensor(np.stack(h_list),  dtype=torch.float32, device=device)
h2_torch = torch.tensor(np.stack(h2_list), dtype=torch.float32, device=device)

w_inc  = np.zeros(Tlen, dtype=np.float32)
deltaV = np.zeros(Tlen, dtype=np.float32)
dh_norm = np.zeros(Tlen, dtype=np.float32)

for k in range(1, Tlen):
    dy = (yPred1[k, 2:3] - yPred2[k, 2:3])          # Δthd
    du = (uk_tensor[k, :] - uk2_tensor[k, :])       # Δtau

    z = torch.cat([dy, du], dim=0).view(-1, 1)
    w_inc[k] = float((z.T @ Mblk.to(z.dtype) @ z).item())

    dh_prev = (h_torch[k-1, :] - h2_torch[k-1, :]).view(-1, 1)
    dh_next = (h_torch[k,   :] - h2_torch[k,   :]).view(-1, 1)

    V_prev = float((dh_prev.T @ P_t @ dh_prev).item())
    V_next = float((dh_next.T @ P_t @ dh_next).item())
    deltaV[k] = V_next - V_prev

    dh_norm[k] = float(torch.linalg.norm(dh_next).item())

w_minus_deltaV = w_inc - deltaV

plot_incremental_dissipativity(
    t_axis=t_test,
    w_inc=w_inc,
    deltaV=deltaV,
    w_minus_deltaV=w_minus_deltaV,
    title_suffix=f"(eps_pert={eps_pert})"
)