# -*- coding: utf-8 -*-
"""验证 Z_3 L=2 的 [W_x, H] 与 W_x 键选择。"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

N, L = 3, 2
NE = 2 * L * L
D = N ** NE

def eid(t, i, j):
    return i * L + j if t == 'h' else L * L + i * L + j

stars = [[eid('h', i, j), eid('h', i, (j - 1) % L),
          eid('v', i, j), eid('v', (i - 1) % L, j)] for i in range(L) for j in range(L)]
plaqs = [[eid('h', i, j), eid('h', (i + 1) % L, j),
          eid('v', i, j), eid('v', i, (j + 1) % L)] for i in range(L) for j in range(L)]
stride = [N ** e for e in range(NE)]
w = np.exp(2j * np.pi / N)

H = lil_matrix((D, D), dtype=float)
for pi, pl in enumerate(plaqs):
    i, j = pi // L, pi % L
    signs = [+1, +1, -1, -1]
    for s in range(D):
        kk = sum(sg * ((s // stride[e]) % N) for e, sg in zip(pl, signs))
        H[s, s] -= 2.0 * np.cos(2 * np.pi * kk / N)
for st in stars:
    for s in range(D):
        for dk in (+1, -1):
            t = s
            for e in st:
                ke = (s // stride[e]) % N
                kn = (ke + dk) % N
                t += (kn - ke) * stride[e]
            H[t, s] -= 1.0
H = csr_matrix(H)

Wx = np.zeros(D, dtype=complex)
for s in range(D):
    sx = sum((s // stride[eid('v', i, 0)]) % N for i in range(L))
    Wx[s] = w ** sx

# [W_x, H] = Wx·H - H·Wx（Wx 对角）
comm = np.max(np.abs(Wx[:, None] * H.toarray() - H.toarray() * Wx[None, :]))
print('[W_x, H] max = %.2e' % comm)

# 基态空间 W_x 表示
wv, v = np.linalg.eigh(H.toarray())
E0 = wv[0]
deg = int((np.abs(wv - E0) < 1e-8).sum())
gs = v[:, :deg]
Wproj = np.zeros((deg, deg), dtype=complex)
for a in range(deg):
    for b in range(deg):
        Wproj[a, b] = np.conj(gs[:, a]) @ (Wx * gs[:, b])
ew = np.linalg.eigvalsh((Wproj + Wproj.conj().T) / 2)
print('简并 =', deg)
print('W_x 特征值(实部) =', np.round(ew, 4).tolist())
print('单位根实部 =', np.round(np.cos(2 * np.pi * np.arange(N) / N), 4).tolist())
