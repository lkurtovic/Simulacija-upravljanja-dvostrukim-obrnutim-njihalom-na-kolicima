import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ---- fizikalni parametri ----
m, m1, m2 = 1.0, 0.3, 0.2
l1, l2 = 0.3, 0.25
d1, d2, d3 = 0.1, 0.05, 0.05
g = 9.81

# ---- linearizirani model ----
M0 = np.array([[m+m1+m2, l1*(m1+m2), m2*l2],
               [l1*(m1+m2), l1**2*(m1+m2), l1*l2*m2],
               [l2*m2, l1*l2*m2, l2**2*m2]])
S = np.array([[0, 0, 0],
              [0, g*(m1+m2)*l1, 0],
              [0, 0, g*l2*m2]])
D = np.diag([d1, d2, d3])
Bu = np.array([[1.], [0.], [0.]])
M0inv = np.linalg.inv(M0)

A = np.zeros((6, 6))
A[0:3, 3:6] = np.eye(3)
A[3:6, 0:3] = M0inv @ S
A[3:6, 3:6] = -M0inv @ D

B = np.zeros((6, 1))
B[3:6, :] = M0inv @ Bu


def nonlin(t, x, Kctrl):
    """Puni nelinearni model M(y) yddot = f(y,ydot,u,0), s regulatorom u=-K x."""
    q, th1, th2, qd, th1d, th2d = x
    u = float(-Kctrl @ x)
    M = np.array([
        [m+m1+m2, l1*(m1+m2)*np.cos(th1), m2*l2*np.cos(th2)],
        [l1*(m1+m2)*np.cos(th1), l1**2*(m1+m2), l1*l2*m2*np.cos(th1-th2)],
        [l2*m2*np.cos(th2), l1*l2*m2*np.cos(th1-th2), l2**2*m2]])
    f = np.array([
        l1*(m1+m2)*th1d**2*np.sin(th1) + m2*l2*th2d**2*np.sin(th2) + u,
        -l1*l2*m2*th2d**2*np.sin(th1-th2) + g*(m1+m2)*l1*np.sin(th1),
        l1*l2*m2*th1d**2*np.sin(th1-th2) + g*l2*m2*np.sin(th2)
    ]) - np.array([d1*qd, d2*th1d, d3*th2d])
    yddot = np.linalg.solve(M, f)
    return [qd, th1d, th2d, *yddot]


def evaluate(Kctrl, label, x0, tfinal=8):
    sol = solve_ivp(nonlin, [0, tfinal], x0, args=(Kctrl,), max_step=0.01)
    th1, th2 = sol.y[1, :], sol.y[2, :]
    u_hist = -(Kctrl @ sol.y)
    max_th1 = np.rad2deg(np.max(np.abs(th1)))
    max_th2 = np.rad2deg(np.max(np.abs(th2)))
    max_u = np.max(np.abs(u_hist))
    effort = trapz(u_hist**2, sol.t)
    tol = np.deg2rad(2)
    ok = (np.abs(th1) < tol) & (np.abs(th2) < tol)
    settle = next((sol.t[i] for i in range(len(sol.t)) if np.all(ok[i:])), None)
    settle_str = f"{settle:.2f}s" if settle is not None else "NE SMIRI SE"
    print(f"{label:32s} K={np.round(Kctrl, 4)}")
    print(f"{'':32s}  max|th1|={max_th1:7.1f}deg  max|th2|={max_th2:7.1f}deg  "
          f"max|u|={max_u:8.1f}N  trud={effort:10.1f}  t_smirivanja={settle_str}\n")


x0 = [0, np.deg2rad(15), np.deg2rad(-10), 0, 0, 0]

print("--- LQR (razlicite tezine Q, R) ---\n")
for Qdiag, R, name in [
    ([1, 100, 100, 1, 10, 10], 1, "LQR standard (koristen u radu)"),
    ([1, 500, 500, 1, 20, 20], 1, "LQR - jace penalizira kuteve"),
    ([1, 20, 20, 1, 5, 5], 1,     "LQR - stedi silu"),
]:
    Q = np.diag(Qdiag).astype(float)
    P = solve_continuous_are(A, B, Q, np.array([[R]]))
    Klqr = (np.linalg.inv([[R]]) @ B.T @ P).flatten()
    evaluate(Klqr, name, x0)