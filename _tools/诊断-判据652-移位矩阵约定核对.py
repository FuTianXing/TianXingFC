# -*- coding: utf-8 -*-
"""临时调试：绕数构型的连线变量是否为纯规范变换。"""
import numpy as np

N = 12
Sx = np.roll(np.eye(N), -1, axis=1)
Sy = np.roll(np.eye(N), -1, axis=1)
S1 = np.kron(Sx, np.eye(N))
S2 = np.kron(np.eye(N), Sy)
xarr = np.repeat(np.arange(N), N)
yarr = np.tile(np.arange(N), N)
P1 = np.diag((-1.0) ** xarr)
ONE = np.ones(N * N, dtype=complex)


def staggered(U1, U2):
    S1U = np.diag(U1) @ S1
    S2U = np.diag(U2) @ S2
    return (S1U - S1U.conj().T) / 2.0 + P1 @ (S2U - S2U.conj().T) / 2.0


def winding_theta(w, xc, yc):
    c45, s45 = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
    vx = (xarr - xc).astype(float)
    vy = (yarr - yc).astype(float)
    vx[np.abs(vx) < 1e-12] = 1e-12
    vp = c45 * vx + s45 * vy
    vq = -s45 * vx + c45 * vy
    return w * np.arctan2(vq, vp)


th = winding_theta(1, N // 2, N // 2)
th2 = th.reshape(N, N)
d1 = np.roll(th2, -1, axis=0) - th2
d2 = np.roll(th2, -1, axis=1) - th2
U1 = np.exp(1j * d1).reshape(-1)
U2 = np.exp(1j * d2).reshape(-1)
M = staggered(U1, U2)
Dfree = staggered(ONE, ONE)
G = np.diag(np.exp(1j * th))

print("max|d1| =", np.abs(d1).max(), " max|d2| =", np.abs(d2).max())
print("max|U1-1| =", np.abs(U1 - 1).max(), " max|U2-1| =", np.abs(U2 - 1).max())
print("max|M - G^dag Dfree G| =", np.abs(M - G.conj().T @ Dfree @ G).max())
print("max|M - G Dfree G^dag| =", np.abs(M - G @ Dfree @ G.conj().T).max())
# 逐条连线核对纯规范恒等式
Gd = np.exp(1j * th)
i1 = np.arange(N * N)
j1 = np.roll(i1, -N)  # n + e1：索引 i = x*N+y，x+1 即 i+N
j2 = np.array([(i // N) * N + (i % N + 1) % N for i in i1])
e1 = np.abs(U1 - Gd[j1] / Gd[i1]).max()
e2 = np.abs(U2 - Gd[j2] / Gd[i1]).max()
print("纯规范逐线核对：x 方向 %.3e，y 方向 %.3e" % (e1, e2))
print("格面通量原始值最大模 =", np.abs(
    d1 + np.roll(d2, -1, axis=0) - np.roll(d1, -1, axis=1) - d2).max())
sv1 = np.linalg.svd(M, compute_uv=False)
sv0 = np.linalg.svd(Dfree, compute_uv=False)
print("谱最大差 =", np.abs(sv1 - sv0).max())

print("---- 分块核对 ----")
D1free = (S1 - S1.T) / 2.0
D2free = (S2 - S2.T) / 2.0
S1U = np.diag(U1) @ S1
S2U = np.diag(U2) @ S2
D1U = (S1U - S1U.conj().T) / 2.0
D2U = (S2U - S2U.conj().T) / 2.0
print("D1U vs G^dag D1free G =", np.abs(D1U - G.conj().T @ D1free @ G).max())
print("D2U vs G^dag D2free G =", np.abs(D2U - G.conj().T @ D2free @ G).max())
print("D1U vs G D1free G^dag =", np.abs(D1U - G @ D1free @ G.conj().T).max())
print("D2U vs G D2free G^dag =", np.abs(D2U - G @ D2free @ G.conj().T).max())
print("D1U 反对易子与 P1：", np.abs(P1 @ D1U + D1U @ P1).max())
