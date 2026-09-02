import numpy as np
from scipy.linalg import solve_continuous_are

# ---- fizikalni parametri ----
m, m1, m2 = 1.0, 0.3, 0.2
l1, l2 = 0.3, 0.25
d1, d2, d3 = 0.1, 0.05, 0.05
g = 9.81

# ---- linearizirani model:  xdot = A x + B u ----
M0 = np.array([[m+m1+m2, l1*(m1+m2), m2*l2],
               [l1*(m1+m2), l1**2*(m1+m2), l1*l2*m2],
               [l2*m2, l1*l2*m2, l2**2*m2]])
K_ = np.array([[0, 0, 0],
               [0, g*(m1+m2)*l1, 0],
               [0, 0, g*l2*m2]])
D = np.diag([d1, d2, d3])
Bu = np.array([[1.], [0.], [0.]])
M0inv = np.linalg.inv(M0)

A = np.zeros((6, 6))
A[0:3, 3:6] = np.eye(3)
A[3:6, 0:3] = M0inv @ K_
A[3:6, 3:6] = -M0inv @ D

B = np.zeros((6, 1))
B[3:6, :] = M0inv @ Bu

# =========================================================================
# LQR DIZAJN
#   x = [q, theta1, theta2, qdot, theta1dot, theta2dot]
#   J = integral( x'Qx + u'Ru ) dt      (C = I, tj. y = x, vidi Hespanha 10.1)
# =========================================================================
Q = np.diag([1., 100., 100., 1., 10., 10.])   # tezine na stanje
R = np.array([[1.]])                           # tezina na upravljacku silu

# 1) rijesi Algebarsku Riccatijevu jednadzbu:  A'P + PA - PBR^-1B'P + Q = 0
P = solve_continuous_are(A, B, Q, R)

# 2) provjera - koliko je "ostatak" ARE jednadzbe blizu nule (mora biti ~0)
residual = A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
print("Norma ostatka ARE jednadzbe (treba biti ~0):", np.linalg.norm(residual))

# 3) pojacanje regulatora:  K = R^-1 B' P
Kctrl = (np.linalg.inv(R) @ B.T @ P)
print("\nLQR pojacanje K =")
print(np.round(Kctrl, 4))

# 4) zatvorena petlja:  Acl = A - B*K
Acl = A - B @ Kctrl
print("\nMatrica zatvorene petlje Acl = A - B*K:")
print(np.round(Acl, 4))

# 5) provjera stabilnosti - svi polovi moraju imati Re < 0
eig_open = np.linalg.eigvals(A)
eig_closed = np.linalg.eigvals(Acl)

print("\nPolovi OTVORENE petlje (A):")
for lam in eig_open:
    print(f"   {lam:.4f}")

print("\nPolovi ZATVORENE petlje (Acl = A-BK):")
for lam in eig_closed:
    print(f"   {lam:.4f}")

if np.all(eig_closed.real < 0):
    print("\n--> Zatvorena petlja je STABILNA (regulator uspjesno stabilizira sustav).")
else:
    print("\n--> POZOR: zatvorena petlja NIJE stabilna - promijeni Q ili R.")