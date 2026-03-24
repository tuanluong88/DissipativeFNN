# Modeling MSD system using incrementally dissipative model
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import model_DMLP as FNN

# ============================================================
# MSD + Dissipative model (model_DMLP.py unchanged)
#
# Dataset: data/data_train_MSD.csv
# Expected columns:
#   time, x1, x2, u, y1, y2, y1_prev, y2_prev, x1_prev, x2_prev, u_prev
#
# We train: u -> y = [x, v] where we take:
#   x := y1_prev (col 6), v := y2_prev (col 7)
#
# Incremental dissipativity check uses: u = force, y = v (velocity)
# Supply: w = Δu * Δy  
#
# SAVE TXT includes:
#   time
#   traj1 true: x v u
#   pred1:      x v (rollout on u1)
#   traj2 true: NaN (not available) + u2
#   pred2:      x v (rollout on u2)
#   deltas: dx dv du
#   w_inc, deltaV, w_minus_deltaV
#   dh_norm
#   + appended stats
# ============================================================

# -------------------------------
# Seeds / device
# -------------------------------
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Plot helpers
# -------------------------------
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

def plot_results_msd(t_axis, x_true, v_true, x_pred, v_pred):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, x_true, label="True x(t)")
    plt.plot(t_axis, x_pred, "--", label="Pred x(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, v_true, label="True v(t)")
    plt.plot(t_axis, v_pred, "--", label="Pred v(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("v")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# -------------------------------
# Load dataset (MSD)
# -------------------------------
data_path = "msd_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find {data_path}")

data = np.loadtxt(data_path, delimiter=",", skiprows=1)
t   = data[:, 0]
x   = data[:, 1]
v   = data[:, 2]
u   = data[:, 3]

u_all = u.reshape(-1, 1)                      # (T,1)
y_all = np.stack([x, v], axis=1)              # (T,2)

N = t.shape[0]
N_train = int(0.5 * N)

u_train = u_all[:N_train, :]
y_train = y_all[:N_train, :]

u_test = u_all
y_test = y_all
t_test = t

# -------------------------------
# Settings 
# -------------------------------
load_model = False
input_size, hidden_size, output_size, num_layers = 1, 32, 2, 4
learning_rate = 1e-3
model_name = "DMLP_MSD.pt" #
num_epochs = 10000
NUM_ITER = 1
out_txt = "DMLP_MSD.txt"

# -------------------------------
# Tensors
# -------------------------------
u_seq  = torch.tensor(u_train, dtype=torch.float32, device=device)  # (Ttrain,1)
y_seq  = torch.tensor(y_train, dtype=torch.float32, device=device)  # (Ttrain,2)
x_Test = torch.tensor(u_test,  dtype=torch.float32, device=device)  # (T,1)

noise_std = 0.0 #0.0 #
y_seq = y_seq + torch.randn_like(y_seq) * noise_std

# -------------------------------
# Model / optimizer
# -------------------------------
model = FNN.FNNModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
x0 = torch.zeros(1, hidden_size, device=device)

# optional load
if load_model and os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_train, _, _ = model(u_seq, x0, batch_first=True)
        y_pred_train_2d = y_pred_train.squeeze(0) if y_pred_train.ndim == 3 else y_pred_train
        loss0 = criterion(y_pred_train_2d, y_seq)
        print("Initial loaded model train loss:", float(loss0.item()))

# -------------------------------
# Training
# -------------------------------
error = []
runtime = []

for _ in range(NUM_ITER):
    best_loss = np.inf
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred, _, _ = model(u_seq, x0, batch_first=True)
        y_pred_2d = y_pred.squeeze(0) if y_pred.ndim == 3 else y_pred

        loss = criterion(y_pred_2d, y_seq)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
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

    # evaluation on full horizon (traj1)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_t, _, _ = model(x_Test, x0, batch_first=True)
        y_pred_np = (y_pred_t.squeeze(0) if y_pred_t.ndim == 3 else y_pred_t).cpu().numpy()

    x_pred = y_pred_np[:, 0]
    v_pred = y_pred_np[:, 1]
    x_true = y_test[:, 0]
    v_true = y_test[:, 1]

    plot_results_msd(t_test, x_true, v_true, x_pred, v_pred)

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
# u = force, y = v (output index 1)
# w = Δu * Δy (same style as your 1DOF script)
# deltaV uses hidden state diff with P from the model
# ============================================================

# quadratic form for dy*du
Q = np.array([[0.0]], dtype=np.float32)
S = np.array([[0.5]], dtype=np.float32)
R = np.array([[0.0]], dtype=np.float32)

Q_t = torch.tensor(Q, device=device)
S_t = torch.tensor(S, device=device)
R_t = torch.tensor(R, device=device)

M_top = torch.cat([Q_t, S_t.T], dim=1)      # (1,2)
M_bot = torch.cat([S_t, R_t], dim=1)        # (1,2)
Mblk  = torch.cat([M_top, M_bot], dim=0)    # (2,2)

u1_tensor = torch.tensor(u_test, dtype=torch.float32, device=device)  # (T,1)
Tlen = u1_tensor.shape[0]

eps_pert = 10  # << tune (MSD force scale usually smaller than 10)
u2_tensor = u1_tensor + eps_pert * torch.randn_like(u1_tensor)

# rollout sequentially to collect y + hidden + P
yPred1 = torch.zeros((Tlen, output_size), device=device)
yPred2 = torch.zeros((Tlen, output_size), device=device)
h_list, h2_list, P_list = [], [], []

model.eval()
with torch.no_grad():
    h_t = x0.clone()
    for k in range(Tlen):
        inp_k = u1_tensor[k:k+1, :]
        y_k, P_k, h_t = model(inp_k, h_t)
        yPred1[k, :] = y_k.view(-1)
        h_list.append(h_t.detach().view(-1).cpu().numpy())
        P_list.append(P_k.detach().cpu())

with torch.no_grad():
    h2_t = x0.clone()
    for k in range(Tlen):
        inp2_k = u2_tensor[k:k+1, :]
        y2_k, _, h2_t = model(inp2_k, h2_t)
        yPred2[k, :] = y2_k.view(-1)
        h2_list.append(h2_t.detach().view(-1).cpu().numpy())

P_t = torch.as_tensor(P_list[-1], dtype=torch.float32, device=device)
h1 = torch.tensor(np.stack(h_list),  dtype=torch.float32, device=device)  # (T,H)
h2 = torch.tensor(np.stack(h2_list), dtype=torch.float32, device=device)  # (T,H)

w_inc   = np.zeros(Tlen, dtype=np.float32)
deltaV  = np.zeros(Tlen, dtype=np.float32)
dh_norm = np.zeros(Tlen, dtype=np.float32)

for k in range(1, Tlen):
    dy = (yPred1[k, 1:2] - yPred2[k, 1:2])      # Δv
    du = (u1_tensor[k, :] - u2_tensor[k, :])    # Δu

    z = torch.cat([dy, du], dim=0).view(-1, 1)
    w_inc[k] = float((z.T @ Mblk.to(z.dtype) @ z).item())

    dh_prev = (h1[k-1, :] - h2[k-1, :]).view(-1, 1)
    dh_next = (h1[k,   :] - h2[k,   :]).view(-1, 1)

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