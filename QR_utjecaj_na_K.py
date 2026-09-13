import numpy as np
from scipy.linalg import solve_continuous_are

# =========================================================================
# FIZIKALNI PARAMETRI
# =========================================================================
m, m1, m2 = 1.0, 0.3, 0.2
l1, l2 = 0.3, 0.25
d1, d2, d3 = 0.1, 0.05, 0.05
g = 9.81

M0 = np.array([[m+m1+m2, l1*(m1+m2), m2*l2],
               [l1*(m1+m2), l1**2*(m1+m2), l1*l2*m2],
               [l2*m2, l1*l2*m2, l2**2*m2]])
S = np.array([[0, 0, 0],
              [0, g*(m1+m2)*l1, 0],
              [0, 0, g*l2*m2]])
D = np.diag([d1, d2, d3])
Bu = np.array([[1.], [0.], [0.]])

RHS = np.hstack([S, D, Bu])
SOL = np.linalg.solve(M0, RHS)
M0inv_S = SOL[:, 0:3]
M0inv_D = SOL[:, 3:6]
M0inv_Bu = SOL[:, 6:7]

A = np.zeros((6, 6))
A[0:3, 3:6] = np.eye(3)
A[3:6, 0:3] = M0inv_S
A[3:6, 3:6] = -M0inv_D

B = np.zeros((6, 1))
B[3:6, :] = M0inv_Bu


def getK(Qdiag, R):
    """Vraca LQR pojacanje K za zadanu dijagonalu Q i skalar R."""
    Q = np.diag(Qdiag).astype(float)
    Rmat = np.array([[R]])
    P = solve_continuous_are(A, B, Q, Rmat)
    K = np.linalg.solve(Rmat, B.T @ P).flatten()
    return K


base = [1, 100, 100, 1, 10, 10]

print("=== 1) Polazisna (bazna) tezina Q, R=1 ===")
K0 = getK(base, 1)
print("K =", np.round(K0, 3))

print("\n=== 2) Samo q2 (tezina na theta1) raste, ostalo fiksno na q1=1, q3=100, q4=1, q5=10, q6=10, R=1 ===")
for q2 in [50, 100, 200, 400, 800]:
    Qd = base.copy()
    Qd[1] = q2
    K = getK(Qd, 1)
    print(f"q2={q2:5d}  K = {np.round(K, 3)}")

print("\n=== 3) Samo R raste, Q fiksno na diag(1,100,100,1,10,10) ===")
for R in [0.5, 1, 2, 4, 8]:
    K = getK(base, R)
    print(f"R={R:4.1f}   K = {np.round(K, 3)}")

print("\n=== 4) Skaliranje CIJELOG Q istim faktorom, R fiksno na 1 ===")
for factor in [0.5, 1, 2, 4]:
    Qd = [q * factor for q in base]
    K = getK(Qd, 1)
    print(f"faktor={factor:4.1f}  K = {np.round(K, 3)}")

print("\n=== 5) Provjera: isto skaliranje Q I R zajedno -> K se NE mijenja ===")
for factor in [0.5, 1, 2, 4]:
    Qd = [q * factor for q in base]
    R = 1 * factor
    K = getK(Qd, R)
    print(f"faktor={factor:4.1f}  K = {np.round(K, 3)}")