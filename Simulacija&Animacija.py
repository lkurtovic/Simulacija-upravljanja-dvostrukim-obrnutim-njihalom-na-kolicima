import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =========================================================================
# 1) FIZIKALNI PARAMETRI
# =========================================================================
m, m1, m2 = 1.0, 0.3, 0.2
l1, l2 = 0.3, 0.25
d1, d2, d3 = 0.1, 0.05, 0.05
g = 9.81

# =========================================================================
# 2) LINEARIZIRANI MODEL:  xdot = A x + B u
# =========================================================================
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

# =========================================================================
# 3) LQR REGULATOR
# =========================================================================
Q = np.diag([1., 100., 100., 1., 10., 10.])
R = np.array([[1.]])
P = solve_continuous_are(A, B, Q, R)
Kctrl = np.linalg.solve(R, B.T @ P).flatten()

# =========================================================================
# 4) PUNI NELINEARNI MODEL
# =========================================================================
def nonlinear_dynamics(t, x):
    q, th1, th2, qd, th1d, th2d = x
    u = -Kctrl @ x
    M = np.array([
        [m+m1+m2,               l1*(m1+m2)*np.cos(th1),   m2*l2*np.cos(th2)],
        [l1*(m1+m2)*np.cos(th1), l1**2*(m1+m2),            l1*l2*m2*np.cos(th1-th2)],
        [l2*m2*np.cos(th2),      l1*l2*m2*np.cos(th1-th2), l2**2*m2]
    ])
    f = np.array([
        l1*(m1+m2)*th1d**2*np.sin(th1) + m2*l2*th2d**2*np.sin(th2) + u,
        -l1*l2*m2*th2d**2*np.sin(th1-th2) + g*(m1+m2)*l1*np.sin(th1),
        l1*l2*m2*th1d**2*np.sin(th1-th2) + g*l2*m2*np.sin(th2)
    ]) - np.array([d1*qd, d2*th1d, d3*th2d])
    yddot = np.linalg.solve(M, f)
    return [qd, th1d, th2d, yddot[0], yddot[1], yddot[2]]

x0 = [0, np.deg2rad(1), np.deg2rad(-1), 0, 0, 0]
t_final = 8
fps = 30
t_eval = np.linspace(0, t_final, int(t_final*fps))
sol = solve_ivp(nonlinear_dynamics, [0, t_final], x0, t_eval=t_eval, max_step=0.01)
X = sol.y

# =========================================================================
# 5) GRAFOVI (q, theta1, theta2, u kroz vrijeme)
# =========================================================================
u_hist = -(Kctrl @ X)

fig1, axs = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
axs[0].plot(sol.t, X[0]); axs[0].set_ylabel('q [m]'); axs[0].grid(True)
axs[1].plot(sol.t, np.rad2deg(X[1])); axs[1].set_ylabel(r'$\theta_1$ [deg]'); axs[1].grid(True)
axs[2].plot(sol.t, np.rad2deg(X[2])); axs[2].set_ylabel(r'$\theta_2$ [deg]'); axs[2].grid(True)
axs[3].plot(sol.t, u_hist); axs[3].set_ylabel('u [N]'); axs[3].set_xlabel('t [s]'); axs[3].grid(True)
fig1.tight_layout()
fig1.savefig('odzivi.png', dpi=120)
print("Graf odziva spremljen kao 'odzivi.png'")

# =========================================================================
# 6) ANIMACIJA - KAMERA PRATI KOLICA
# =========================================================================
half_width = l1 + l2 + 0.3   # koliko sirok "prozor" oko kolica zelimo
cart_w, cart_h = 0.2, 0.1

fig2, ax = plt.subplots(figsize=(6, 6))
ax.set_ylim(-0.1, l1 + l2 + 0.3)
ax.set_aspect('equal')
ax.grid(True)

ground, = ax.plot([], [], 'k-', lw=1)
cart_patch = plt.Rectangle((0, 0), cart_w, cart_h, fc=[0.3, 0.3, 0.8])
ax.add_patch(cart_patch)
link1, = ax.plot([], [], 'k-', lw=2)
link2, = ax.plot([], [], 'k-', lw=2)
mass1, = ax.plot([], [], 'ro', ms=10, mfc='r')
mass2, = ax.plot([], [], 'go', ms=10, mfc='g')
time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def update(frame):
    q, th1, th2 = X[0, frame], X[1, frame], X[2, frame]
    x1, y1 = q + l1*np.sin(th1), l1*np.cos(th1)
    x2, y2 = x1 + l2*np.sin(th2), y1 + l2*np.cos(th2)

    # <<< KAMERA PRATI KOLICA: pomakni granice x-osi oko trenutne pozicije q
    ax.set_xlim(q - half_width, q + half_width)

    ground.set_data([q - half_width, q + half_width], [0, 0])
    cart_patch.set_xy((q - cart_w/2, -cart_h/2))
    link1.set_data([q, x1], [0, y1])
    link2.set_data([x1, x2], [y1, y2])
    mass1.set_data([x1], [y1])
    mass2.set_data([x2], [y2])
    time_txt.set_text(f't = {sol.t[frame]:.2f} s')
    return cart_patch, link1, link2, mass1, mass2, time_txt, ground


ani = animation.FuncAnimation(fig2, update, frames=len(sol.t),
                               interval=1000/fps, blit=False)

ani.save('animacija_njihala_kamera.gif', writer='pillow', fps=fps)
print("Animacija (kamera prati kolica) spremljena kao 'animacija_njihala_kamera.gif'")
