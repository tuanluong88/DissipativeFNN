import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import model_DRNN as RNN
import csv
# =====================================
# 0. Seeds for repeatability
# =====================================
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def save_csv(filename, header, data_rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
# =====================================
# 1. Generate training data (forced mass-spring-damper)
# =====================================
def forcing_function(t, A=1.0, omega=1.5):
    return A * np.sin(omega * t)

def rk4_step(f, t, y, h, *args):
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t,         y,             *args), dtype=float)
    k2 = np.asarray(f(t + h/2.,  y + h*k1/2.,   *args), dtype=float)
    k3 = np.asarray(f(t + h/2.,  y + h*k2/2.,   *args), dtype=float)
    k4 = np.asarray(f(t + h,     y + h*k3,      *args), dtype=float)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6.

def rk4_solve(f, t0, tf, y0, h, *args):
    t_values = [t0]
    y_values = [np.asarray(y0, dtype=float)]
    t = t0
    y = np.asarray(y0, dtype=float)
    while t < tf:
        if t + h > tf:
            h = tf - t
        y = rk4_step(f, t, y, h, *args)
        t += h
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.vstack(y_values)

def nonhomogeneous_dho(t, y, m, mu, k, A, omega):
    x, v = y
    dxdt = v
    dvdt = (forcing_function(t, A, omega) - mu * v - k * x) / m
    return [dxdt, dvdt]

def plot_results(t_axis, x_true, v_true, x_pred, v_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, x_true, label="True x(t)")
    plt.plot(t_axis, x_pred, "--", label="Pred x(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement")
    plt.legend()
    plt.title("Displacement Prediction")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, v_true, label="True v(t)")
    plt.plot(t_axis, v_pred, "--", label="Pred v(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity")
    plt.legend()
    plt.title("Velocity Prediction")
    plt.tight_layout()
    plt.show()
def save_txt_msd_drnn(
    filename,
    t,
    u,
    x_true, v_true,
    x_pred, v_pred,
    yu, deltaV
):
    """
    Save results to txt with columns:
    time, u, x_true, v_true, x_pred, v_pred, yu, deltaV, yu_minus_deltaV
    """
    t = np.asarray(t).reshape(-1)
    u = np.asarray(u).reshape(-1)

    x_true = np.asarray(x_true).reshape(-1)
    v_true = np.asarray(v_true).reshape(-1)
    x_pred = np.asarray(x_pred).reshape(-1)
    v_pred = np.asarray(v_pred).reshape(-1)

    yu = np.asarray(yu).reshape(-1)
    deltaV = np.asarray(deltaV).reshape(-1)

    # Make all same length (safe)
    N = min(len(t), len(u), len(x_true), len(v_true), len(x_pred), len(v_pred), len(yu), len(deltaV))
    t = t[:N]; u = u[:N]
    x_true = x_true[:N]; v_true = v_true[:N]
    x_pred = x_pred[:N]; v_pred = v_pred[:N]
    yu = yu[:N]; deltaV = deltaV[:N]

    yu_minus_deltaV = yu - deltaV

    header = (
        "time\tu\t"
        "x_true\tv_true\t"
        "x_pred\tv_pred\t"
        "yu\tdeltaV\tyu_minus_deltaV"
    )

    out = np.column_stack([
        t, u,
        x_true, v_true,
        x_pred, v_pred,
        yu, deltaV, yu_minus_deltaV
    ])

    np.savetxt(filename, out, header=header, delimiter="\t", comments="", fmt="%.8e")
    print(f"[Saved] {filename}")
# =====================================
# 2. Settings
# =====================================
m, mu, k = 1.0, 2.5, 4.2
A, omega = 5.5, 1.0
y0 = [0.0, 0.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model =  False #True #
input_size, hidden_size, output_size, num_layers = 1, 64, 2, 2
learning_rate = 1e-3
model_name = "DRNN_MSD.pt"
num_epochs = 5000
dt = 0.05  # integration step
NUM_ITER = 5 # For averate error evaluation
filename="drnn_msd_results_with_dissipativity.txt"

# =====================================
# 3. Generate data
# =====================================
t, sol = rk4_solve(nonhomogeneous_dho, 0, 10, y0, dt, m, mu, k, A, omega)
tTest, solTest = rk4_solve(nonhomogeneous_dho, 0, 20, y0, dt, m, mu, k, A, omega)

u_test = forcing_function(tTest, A, omega)       # (T_test,)
y_test = solTest                                 # (T_test,2)

csv_filename = "msd_data.csv"
header = ["time", "x", "v","u"]

data_log = np.column_stack([tTest, y_test[:, 0], y_test[:, 1],u_test])
save_csv(csv_filename, header, data_log)

print(f"Saved: {csv_filename}")
