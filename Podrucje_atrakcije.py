import json
import os
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================================================================
# 1) FIZIKALNI PARAMETRI I LQR REGULATOR (standardni, kao u ostatku rada)
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
M0inv = np.linalg.inv(M0)

A = np.zeros((6, 6))
A[0:3, 3:6] = np.eye(3)
A[3:6, 0:3] = M0inv @ S
A[3:6, 3:6] = -M0inv @ D

B = np.zeros((6, 1))
B[3:6, :] = M0inv @ Bu

Q = np.diag([1., 100., 100., 1., 10., 10.])
R = np.array([[1.]])
P = solve_continuous_are(A, B, Q, R)
Kctrl = np.linalg.solve(R, B.T @ P).flatten()


def nonlinear_dynamics(t, x):
    q, th1, th2, qd, th1d, th2d = x
    u = -Kctrl @ x
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
    return [qd, th1d, th2d, yddot[0], yddot[1], yddot[2]]


def stabilizes(th1_deg, th2_deg, tf=6):
    """Vraca True ako se sustav stabilizira (ostaje unutar +-90 stupnjeva)."""
    x0 = [0, np.deg2rad(th1_deg), np.deg2rad(th2_deg), 0, 0, 0]
    sol = solve_ivp(nonlinear_dynamics, [0, tf], x0,
                     max_step=0.02, rtol=1e-7, atol=1e-9)
    return np.max(np.abs(sol.y[1])) < np.deg2rad(90)


# =========================================================================
# 2) CACHE -- ucitaj vec izracunate tocke iz prijasnjih pokretanja
# =========================================================================
CACHE_FILE = "rezultati_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    print(f"Ucitano {len(cache)} vec izracunatih tocaka iz '{CACHE_FILE}'.")
else:
    cache = {}
    print("Nema postojeceg cache-a, krecem od nule.")


def kljuc(th1, th2):
    return f"{int(round(th1))},{int(round(th2))}"


def stabilizes_cached(th1, th2):
    k = kljuc(th1, th2)
    if k in cache:
        return cache[k]
    rez = bool(stabilizes(th1, th2))
    cache[k] = rez
    return rez


# =========================================================================
# 3) MREZA POCETNIH UVJETA (theta1(0), theta2(0)) I TESTIRANJE SVAKOG
# =========================================================================

# =========================================================================
# 3) MREZA POCETNIH UVJETA -- SAMO NOVE RUBNE TRAKE
# =========================================================================
RASPON_TH1 = 55   # stupnjeva, x-os (theta1) -- konacni, ukupni raspon
RASPON_TH2 = 50   # stupnjeva, y-os (theta2)
STARI_RASPON_TH1 = 50   # dio theta1 raspona koji je vec u cacheu
KORAK = 1

th2_vals = np.arange(-RASPON_TH2, RASPON_TH2 + 1, KORAK)

th1_novi = list(np.arange(-RASPON_TH1, -STARI_RASPON_TH1, KORAK)) + \
           list(np.arange(STARI_RASPON_TH1 + 1, RASPON_TH1 + 1, KORAK))

ukupno = len(th1_novi) * len(th2_vals)
novih = 0
print(f"Racunam SAMO nove rubne tocke: theta1 u {th1_novi[:3]}...{th1_novi[-3:]} "
      f"({len(th1_novi)} vrijednosti) x theta2 ({len(th2_vals)} vrijednosti) "
      f"= {ukupno} tocaka...")

for i, th1 in enumerate(th1_novi):
    for th2 in th2_vals:
        bila_u_cacheu = kljuc(th1, th2) in cache
        stabilizes_cached(th1, th2)
        if not bila_u_cacheu:
            novih += 1
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"  theta1={th1} gotov ({i+1}/{len(th1_novi)})  "
          f"(ukupno novoizracunatih do sad: {novih})")

print(f"\nGotovo. Novoizracunato ovaj put: {novih} / {ukupno} tocaka.")
print(f"Cache sada sadrzi {len(cache)} tocaka ukupno, spremljen u '{CACHE_FILE}'.")

# =========================================================================
# 4) UCITAJ SVE TOCKE IZ CACHEA (stare + nove) ZA PUNI GRAF
# =========================================================================
th1_vals = np.arange(-RASPON_TH1, RASPON_TH1 + 1, KORAK)
rezultat = np.zeros((len(th2_vals), len(th1_vals)), dtype=bool)
nedostaje = []
for i, th1 in enumerate(th1_vals):
    for j, th2 in enumerate(th2_vals):
        k = kljuc(th1, th2)
        if k in cache:
            rezultat[j, i] = cache[k]
        else:
            nedostaje.append((th1, th2))

if nedostaje:
    print(f"\nUPOZORENJE: {len(nedostaje)} tocaka jos nije u cacheu "
          f"(nisu bile ni u starom ni u novom rasponu) -- graf ce za njih "
          f"pokazati crveno kao zadano.")

# =========================================================================
# 4) GRAF - PODRUCJE ATRAKCIJE
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 7))

th1_grid, th2_grid = np.meshgrid(th1_vals, th2_vals)
boje = np.where(rezultat, '#2ecc71', '#e74c3c')

ax.scatter(th1_grid, th2_grid, c=boje.ravel(), s=18, edgecolors='none')

ax.set_xlabel(r'$\theta_1(0)$ [deg]')
ax.set_ylabel(r'$\theta_2(0)$ [deg]')
ax.set_title('Područje atrakcije LQR regulatora (zeleno = stabilizira)')
ax.set_xticks(np.arange(-RASPON_TH1, RASPON_TH1 + 1, 5))
ax.set_yticks(np.arange(-RASPON_TH2, RASPON_TH2 + 1, 5))
ax.set_aspect('equal')
ax.grid(True, alpha=0.4)

fig.tight_layout()
fig.savefig('podrucje_atrakcije.png', dpi=150, bbox_inches='tight')
print("Graf spremljen kao 'podrucje_atrakcije.png'")