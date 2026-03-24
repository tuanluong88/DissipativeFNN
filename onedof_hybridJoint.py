import numpy as np
import csv
import matplotlib.pyplot as plt

# ============================================================
# 1-DOF Flexible-joint (Hybrid) Manipulator
# Link:
#   M qdd + Dq qd + Kg q + K(q - th) = 0
# Motor:
#   J thdd + Dth thd - K(q - th) = tau
#
# Incremental dissipativity with u=tau, y=thd:
# Storage:
#   V = 0.5*M*(Δqd)^2 + 0.5*J*(Δthd)^2 + 0.5*K*(Δ(q-th))^2 + 0.5*Kg*(Δq)^2
# gives:
#   Vdot = Δtau*Δthd - Dq*(Δqd)^2 - Dth*(Δthd)^2  <=  Δtau*Δthd
# ============================================================

np.random.seed(0)

def save_csv(filename, header, data_rows):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data_rows)
    print(f"Saved: {filename}")

# ----------------------------
# RK4
# ----------------------------
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
    while t < tf - 1e-12:
        hh = h if (t + h <= tf) else (tf - t)
        y = rk4_step(f, t, y, hh, *args)
        t += hh
        t_values.append(t)
        y_values.append(y.copy())
    return np.array(t_values), np.vstack(y_values)

# ----------------------------
# Input torques (two different)  (KEEP SAME SHAPE AS YOUR FILE)
# ----------------------------
def tau_fun(t, amp=2.0, w=1.0, amp2=0.7, w2=2.3, phase=0.0):
    return amp*np.sin(w*t + phase) + 0.25*amp2*np.sin(w2*t + 0.3*phase)

# ----------------------------
# 1-DOF flexible-joint dynamics
# state y = [q, qd, th, thd]
# ----------------------------
def flex1dof_dynamics(t, y, M, J, Dq, Dth, K, Kg, tau_params):
    q, qd, th, thd = y
    tau = tau_fun(t, **tau_params)
    qdd = (-Dq*qd - Kg*q - K*(q - th)) / M
    thdd = (tau - Dth*thd + K*(q - th)) / J
    return np.array([qd, qdd, thd, thdd], dtype=float)

# ----------------------------
# Incremental storage and exact Vdot
# ----------------------------
def incremental_storage(q1, qd1, th1, thd1, q2, qd2, th2, thd2, M, J, K, Kg):
    dq   = q1 - q2
    dqd  = qd1 - qd2
    dth  = th1 - th2
    dthd = thd1 - thd2
    deta = (q1 - th1) - (q2 - th2)   # Δ(q - th)
    V = 0.5*M*(dqd**2) + 0.5*J*(dthd**2) + 0.5*K*(deta**2) + 0.5*Kg*(dq**2)
    return float(V)

def incremental_Vdot_exact(dqd, dthd, dtau, Dq, Dth):
    supply = float(dtau * dthd)                  # Δu^T Δy (scalar)
    diss   = float(Dq*(dqd**2) + Dth*(dthd**2))  # >= 0
    Vdot_exact = supply - diss
    return Vdot_exact, supply, diss

# ----------------------------
# Downsample helper (SAVE with dt_save)
# ----------------------------
def downsample_by_time(t, *arrays, dt_save=0.5):
    """
    Pick samples closest to times: t0, t0+dt_save, ...
    Returns: t_ds, arrays_ds...
    """
    t = np.asarray(t).reshape(-1)
    t_out = np.arange(t[0], t[-1] + 1e-12, dt_save)
    idx = np.searchsorted(t, t_out)
    idx[idx >= len(t)] = len(t) - 1

    outs = [t[idx]]
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 1:
            outs.append(a[idx])
        else:
            outs.append(a[idx, :])
    return outs

# ----------------------------
# Plot helpers
# ----------------------------
def plot_time(t, a1, a2, title, ylabel, labels=("traj1","traj2")):
    plt.figure(figsize=(12,4))
    plt.plot(t, a1, label=labels[0])
    plt.plot(t, a2, "--", label=labels[1])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_overlay_dense_sparse(t_dense, a1_dense, a2_dense, t_sparse, a1_sparse, a2_sparse,
                                  title, ylabel, labels=("traj1","traj2")):
    """
    Dense dt_sim lines + sparse dt_save markers (same signals).
    """
    plt.figure(figsize=(12,4))
    plt.plot(t_dense, a1_dense, label=f"{labels[0]} (dt_sim)")
    plt.plot(t_dense, a2_dense, "--", label=f"{labels[1]} (dt_sim)")
    plt.plot(t_sparse, a1_sparse, "o", markersize=3, label=f"{labels[0]} (dt_save)")
    plt.plot(t_sparse, a2_sparse, "o", markersize=3, label=f"{labels[1]} (dt_save)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_check(t_mid, dVdt_fd, supply, rhs_exact):
    plt.figure(figsize=(12,4))
    plt.plot(t_mid, dVdt_fd, label="dV/dt (finite-diff)")
    plt.plot(t_mid, supply,  "--", label="supply = Δτ·Δθ̇")
    plt.title("Incremental dissipativity: dV/dt ≤ Δτ·Δθ̇")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    viol = dVdt_fd - supply
    plt.figure(figsize=(12,4))
    plt.plot(t_mid, viol, label="violation = dV/dt - supply")
    plt.axhline(0.0, linestyle="--")
    plt.title("Violation (should be ≤ 0, small + due to numerics)")
    plt.xlabel("Time")
    plt.ylabel("Power difference")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(t_mid, dVdt_fd,   label="dV/dt (finite-diff)")
    plt.plot(t_mid, rhs_exact, "--", label="supply - diss (theory)")
    plt.title("Sanity check: dV/dt vs (Δτ·Δθ̇ − diss)")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Settings
# ============================================================
dt_sim  = 0.001   # simulate with small step
dt_save = 0.2     # save sparse data
T       = 30.0

M  = 2.0
J  = 0.12
Dq = 1.2
Dth= 0.6
K  = 25.0
Kg = 4.0

tau_params_1 = dict(amp=2.0, w=1.0, amp2=0.7, w2=2.3, phase=0.0)
tau_params_2 = dict(amp=2.0, w=1.0, amp2=0.7, w2=2.3, phase=0.7)

# Initial conditions: [q, qd, th, thd]
y0_1 = np.array([0.0, 0.0,  0.1, 0.0], dtype=float)
y0_2 = np.array([0.2, 0.1, -0.1, 0.05], dtype=float)

# ============================================================
# Simulate two trajectories with dt_sim
# ============================================================
t, Y1 = rk4_solve(flex1dof_dynamics, 0.0, T, y0_1, dt_sim, M, J, Dq, Dth, K, Kg, tau_params_1)
_, Y2 = rk4_solve(flex1dof_dynamics, 0.0, T, y0_2, dt_sim, M, J, Dq, Dth, K, Kg, tau_params_2)

q1, qd1, th1, thd1 = Y1[:,0], Y1[:,1], Y1[:,2], Y1[:,3]
q2, qd2, th2, thd2 = Y2[:,0], Y2[:,1], Y2[:,2], Y2[:,3]

Tau1 = np.array([tau_fun(tt, **tau_params_1) for tt in t])
Tau2 = np.array([tau_fun(tt, **tau_params_2) for tt in t])

dq   = q1 - q2
dqd  = qd1 - qd2
dth  = th1 - th2
dthd = thd1 - thd2
dtau = Tau1 - Tau2

# ============================================================
# Downsample to dt_save for saving/training
# ============================================================
t_s, Y1_s, Y2_s, Tau1_s, Tau2_s, dq_s, dqd_s, dth_s, dthd_s, dtau_s = downsample_by_time(
    t, Y1, Y2, Tau1, Tau2, dq, dqd, dth, dthd, dtau, dt_save=dt_save
)
q1_s, qd1_s, th1_s, thd1_s = Y1_s[:,0], Y1_s[:,1], Y1_s[:,2], Y1_s[:,3]
q2_s, qd2_s, th2_s, thd2_s = Y2_s[:,0], Y2_s[:,1], Y2_s[:,2], Y2_s[:,3]

# ============================================================
# Save datasets (BOTH dt_sim and dt_save)
# ============================================================

# ----- Single trajectory (traj1) -----
single_sim = np.column_stack([t, q1, qd1, th1, thd1, Tau1])
save_csv(
    "flex1dof_single_dt_sim.csv",
    ["time","q","qd","th","thd","tau"],
    single_sim
)

single_save = np.column_stack([t_s, q1_s, qd1_s, th1_s, thd1_s, Tau1_s])
save_csv(
    f"flex1dof_single_dt_save_{dt_save}.csv",
    ["time","q","qd","th","thd","tau"],
    single_save
)

# ----- Incremental pair -----
pair_sim = np.column_stack([
    t,
    q1, qd1, th1, thd1, Tau1,
    q2, qd2, th2, thd2, Tau2,
    dq, dqd, dth, dthd, dtau
])
save_csv(
    "flex1dof_incremental_dt_sim.csv",
    ["time",
     "q1","qd1","th1","thd1","tau1",
     "q2","qd2","th2","thd2","tau2",
     "dq","dqd","dth","dthd","dtau"],
    pair_sim
)

pair_save = np.column_stack([
    t_s,
    q1_s, qd1_s, th1_s, thd1_s, Tau1_s,
    q2_s, qd2_s, th2_s, thd2_s, Tau2_s,
    dq_s, dqd_s, dth_s, dthd_s, dtau_s
])
save_csv(
    f"flex1dof_incremental_dt_save_{dt_save}.csv",
    ["time",
     "q1","qd1","th1","thd1","tau1",
     "q2","qd2","th2","thd2","tau2",
     "dq","dqd","dth","dthd","dtau"],
    pair_save
)

# ============================================================
# Incremental dissipativity check (on dt_sim for accuracy)
# ============================================================
V = np.array([incremental_storage(q1[i], qd1[i], th1[i], thd1[i],
                                  q2[i], qd2[i], th2[i], thd2[i],
                                  M, J, K, Kg)
              for i in range(len(t))], dtype=float)

# Finite-diff dV/dt (central) using dt_sim
dVdt_fd = (V[2:] - V[:-2]) / (2*dt_sim)
t_mid = t[1:-1]

rhs_exact = np.zeros_like(dVdt_fd)
supply    = np.zeros_like(dVdt_fd)
diss      = np.zeros_like(dVdt_fd)
for k, i in enumerate(range(1, len(t)-1)):
    rhs_exact[k], supply[k], diss[k] = incremental_Vdot_exact(dqd[i], dthd[i], dtau[i], Dq, Dth)

viol = dVdt_fd - supply
print("Incremental dissipativity check (dt_sim, u=tau, y=thd):")
print("  max(dVdt_fd - supply):", np.max(viol))
print("  mean(dVdt_fd - supply):", np.mean(viol))
print("  (should be near <= 0; small + due to numerics)")

# ============================================================
# Plots (dense dt_sim + sparse dt_save markers)
# ============================================================
plot_time_overlay_dense_sparse(t, q1, q2, t_s, q1_s, q2_s, "Link angle q(t)", "q [rad]")
plot_time_overlay_dense_sparse(t, th1, th2, t_s, th1_s, th2_s, "Motor angle θ(t)", "θ [rad]")
plot_time_overlay_dense_sparse(t, thd1, thd2, t_s, thd1_s, thd2_s, "Motor velocity θ̇(t) = output y", "θ̇ [rad/s]")
plot_time_overlay_dense_sparse(t, Tau1, Tau2, t_s, Tau1_s, Tau2_s, "Input torque τ(t)", "τ [Nm]")

plot_time_overlay_dense_sparse(t, dtau, 0*t, t_s, dtau_s, 0*t_s, "Incremental input Δτ(t)", "Δτ [Nm]", labels=("Δτ","0"))

plot_check(t_mid, dVdt_fd, supply, rhs_exact)

plt.figure(figsize=(12,4))
plt.plot(t_mid, supply, label="supply = Δτ·Δθ̇")
plt.plot(t_mid, diss,   label="diss = Dq(Δq̇)^2 + Dth(Δθ̇)^2")
plt.title("Supply vs Dissipation (incremental, dt_sim)")
plt.xlabel("Time")
plt.ylabel("Power")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
