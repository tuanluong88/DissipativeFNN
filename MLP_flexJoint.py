# ============================================================
# Baseline MLP for flexible joint model
# ============================================================
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# Plot helpers
# =====================================
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

# ============================================================
# MLP baseline
# ============================================================
class MLPBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        layers = []
        in_dim = hidden_size + input_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.Tanh())
        self.h_mlp = nn.Sequential(*layers)
        self.y_head = nn.Linear(hidden_size, output_size)

    def forward(self, u, x, batch_first=False):
        if x.dim() == 1:
            h = x.view(1, -1)
        else:
            h = x

        if u.dim() == 1:
            u = u.view(1, -1)

        T = u.shape[0]
        y_list, h_list = [], []

        for k in range(T):
            uk = u[k:k+1, :]
            yk = self.y_head(h)
            y_list.append(yk)
            h_list.append(h)

            hu = torch.cat([h, uk], dim=1)
            h = self.h_mlp(hu)

        Y_out = torch.cat(y_list, dim=0)
        H_out = torch.cat(h_list, dim=0)
        return Y_out, H_out

# ============================================================
# Load dataset
# ============================================================

data_path = f"flexjoint_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find {data_path}.")

data = np.loadtxt(data_path, delimiter=",", skiprows=1)
t   = data[:, 0]
q   = data[:, 1]
qd  = data[:, 2]
th  = data[:, 3]
thd = data[:, 4]
tau = data[:, 5]

u_all = tau.reshape(-1, 1)
y_all = np.stack([q, th, thd], axis=1)

N = t.shape[0]
N_train = int(0.5 * N)

u_train = u_all[:N_train, :]
y_train = y_all[:N_train, :]

u_test = u_all
y_test = y_all
t_test = t

# ============================================================
# Settings
# ============================================================
load_model = False
input_size, hidden_size, output_size, num_layers = 1, 32, 3, 2
learning_rate = 1e-3
model_name = "MLPbaseline_1dof.pt"
num_epochs = 20000
NUM_ITER = 1

out_txt = "mlpbaseline_1dof.txt"

# ============================================================
# Tensors
# ============================================================
u_seq  = torch.tensor(u_train, dtype=torch.float32, device=device)
y_seq  = torch.tensor(y_train, dtype=torch.float32, device=device)
x_Test = torch.tensor(u_test,  dtype=torch.float32, device=device)

noise_std = 0.0 # 0.3
y_seq = y_seq + torch.randn_like(y_seq) * noise_std

# ============================================================
# Model / optimizer
# ============================================================
model = MLPBaseline(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x0 = torch.zeros(1, hidden_size, device=device)

if load_model and os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_train, _ = model(u_seq, x0, batch_first=True)
        loss0 = criterion(y_pred_train, y_seq)
        print("Initial loaded model train loss:", float(loss0.item()))

# ============================================================
# Training
# ============================================================
error = []
runtime = []
loss_hist = []  # <-- ADDED

for _ in range(NUM_ITER):
    best_loss = np.inf
    start_time = time.time()
    loss_hist = []  # reset per run

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred, _ = model(u_seq, x0, batch_first=True)
        loss = criterion(y_pred, y_seq)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_hist.append(loss_val)  # <-- ADDED

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

    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred_t, _ = model(x_Test, x0, batch_first=True)
        y_pred_np = y_pred_t.cpu().numpy()

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
