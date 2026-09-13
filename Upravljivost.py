import numpy as np

m, m1, m2 = 1.0, 3000, 0.2
l1, l2 = 0.3, 4.5
d1, d2, d3 = 0.1, 0.05, 0.05
g = 9.81

# ---- linearizirani model oko theta1=theta2=0 ----

# ---- potrebno za konstruiranje A ----
M0 = np.array([
    [m+m1+m2,      l1*(m1+m2),      m2*l2],
    [l1*(m1+m2),   l1**2*(m1+m2),   l1*l2*m2],
    [l2*m2,        l1*l2*m2,        l2**2*m2]
])

S = np.array([
    [0, 0, 0],
    [0, g*(m1+m2)*l1, 0],
    [0, 0, g*l2*m2]
])

D  = np.diag([d1, d2, d3])
Bu = np.array([[1.], [0.], [0.]])

M0inv = np.linalg.inv(M0)

A = np.zeros((6, 6))
A[0:3, 3:6] = np.eye(3)   # retci od 0 do 3 i stupci od 3 do 6
A[3:6, 0:3] = M0inv @ S
A[3:6, 3:6] = -M0inv @ D

B = np.zeros((6, 1))
B[3:6, :] = M0inv @ Bu

# ---- upravljiva matrica i njen rang ----
# ---- konstrukcija [B, AB, A^2B,…,A^5B] ----
n = A.shape[0]
cols = [B]
Ai = np.eye(n)
for _ in range(1, n):
    Ai = Ai @ A
    cols.append(Ai @ B)
Co = np.hstack(cols)  # ovo je sada matrica 6x6 napravljena od matrica iz liste

# ---- singularne vrijednosti i rang preko njih ----
sigma = np.linalg.svd(Co, compute_uv=False)
r = np.linalg.matrix_rank(Co)

print("Singularne vrijednosti matrice C:")
for i, s in enumerate(sigma, start=1):
    print(f"   sigma_{i} = {s:.6g}")

print(f"\nrank(ctrb(A,B)) = {r} / {n}")
print("Sustav je", "POTPUNO UPRAVLJIV" if r == n else "NIJE potpuno upravljiv")