# -*- coding: utf-8 -*-
"""Z_3 L=2 基态诊断：A_v/B_p 本征值期望、对易性、基态结构。"""
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
for pl in plaqs:
    for s in range(D):
        kk = sum((s // stride[e]) % N for e in pl)
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

wv, v = np.linalg.eigh(H.toarray())
E0 = wv[0]
deg = int((np.abs(wv - E0) < 1e-8).sum())
print('E0 = %.6f, 简并 = %d' % (E0, deg))
gs = v[:, 0]

Av = [np.zeros((D, D), dtype=complex) for _ in stars]
for si, st in enumerate(stars):
    for s in range(D):
        t = s
        for e in st:
            ke = (s // stride[e]) % N
            kn = (ke + 1) % N
            t += (kn - ke) * stride[e]
        Av[si][t, s] = 1.0
Bp = [np.zeros((D, D), dtype=complex) for _ in plaqs]
for pi, pl in enumerate(plaqs):
    for s in range(D):
        kk = sum((s // stride[e]) % N for e in pl)
        Bp[pi][s, s] = w ** kk

for si in range(len(stars)):
    print('A_v[%d] 期望 = %.4f' % (si, np.conj(gs) @ (Av[si] @ gs)))
for pi in range(len(plaqs)):
    print('B_p[%d] 期望 = %.4f' % (pi, np.conj(gs) @ (Bp[pi] @ gs)))
for si in range(len(stars)):
    for pi in range(len(plaqs)):
        comm = np.max(np.abs(Av[si] @ Bp[pi] - Bp[pi] @ Av[si]))
        print('[A_%d, B_%d] max = %.2e' % (si, pi, comm))
