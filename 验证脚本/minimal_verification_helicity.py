# -*- coding: utf-8 -*-
"""G2 具名缺口之二（L5）：TT 两支的螺旋度 +2 与 -2 的显式确认。

判据（先立判据，再实现；每条都要求可证伪）：
  L5-1 谱：绕 k 的转动生成元（自旋部分作用在对称张量的两个指标上）在六维对称张量空间上
       的谱为 {+2, +1, 0, 0, -1, -2}，在 Frobenius 度量下自伴（故谱为实数）。指标约定或
       因子写错时，本征值不会落在这六个整数上。
  L5-2 不变性与限制谱：TT 投影与该生成元交换；限制到 TT 子空间后谱恰为 {+2, -2}
       （限制的迹为 0、行列式为 -4）。
  L5-3 偏振身份：TT 的两个本征矢在实基 (h_+, h_x) 下的分量满足 v_x = +- i v_+，即环形偏振
       基 (h_+ +- i h_x)/sqrt(2)，螺旋度本征态与圆偏振态是同一件事。
  L5-4 约束子空间身份：余核四维方向上，迹方向与沿 k 的无迹纵向-纵向方向给螺旋度 0，
       两条纵向-横向方向给 +-1（与 L4-2 的余核分解逐项对上）。
  L5-5 传播与简并：把纯螺旋度模式放在一维格点上按离散波动方程演化，其一，TT 性与螺旋度
       分量保持（反分量 <= 1e-12）；其二，两个螺旋度的相位推进相同（无色散差）。
       对照：给演化算子加入张量混合项（人为的灵敏度对照）后，反分量须长到 >= 1e-3。

单位：格距 1、c = 1、dt = 1（一维 Courant 数 1，单向解精确）。确定性，无随机数。
"""

import numpy as np

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
GRAM = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def sym_basis(a, b):
    M = np.zeros((3, 3))
    M[a, b] = 1.0
    M[b, a] = 1.0
    return M


def coeffs(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


def mat_from_coeffs(cv):
    M = np.zeros((3, 3), dtype=complex)
    for val, (a, b) in zip(cv, IDX):
        M[a, b] = val
        M[b, a] = val
    return M


def eps3(i, j, k):
    if len({i, j, k}) < 3:
        return 0.0
    order = [i, j, k]
    sign = 1.0
    for a in range(3):
        for b in range(a + 1, 3):
            if order[a] > order[b]:
                order[a], order[b] = order[b], order[a]
                sign = -sign
    return sign


def spin_generator(khat):
    W = np.zeros((6, 6), dtype=complex)
    for col, (a, b) in enumerate(IDX):
        h = sym_basis(a, b)
        out = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            Si = np.zeros((3, 3), dtype=complex)
            for j in range(3):
                for k in range(3):
                    Si[j, k] = -1j * eps3(i, j, k) * khat[i]
            out += Si @ h + h @ Si.T
        W[:, col] = coeffs(out)
    return W


def tt_projector(khat):
    d = np.eye(3) - np.outer(khat, khat)
    rows = []
    for (a, b) in IDX:
        S = sym_basis(a, b)
        rows.append(coeffs(d @ S @ d - 0.5 * d * np.trace(d @ S)))
    return np.array(rows).T


print("=" * 78)
print("L5-1 生成元在六维对称张量空间上的谱")
print("=" * 78)
TARGET = np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])
worst_spec = 0.0
worst_imag = 0.0
worst_self = 0.0
k_dirs = [np.array([0.0, 0.0, 1.0]),
          np.array([0.267, 0.535, 0.802]) / np.linalg.norm(np.array([0.267, 0.535, 0.802])),
          np.array([1.0, 0.0, 0.0])]
for khat in k_dirs:
    W = spin_generator(khat)
    ev = np.linalg.eigvals(W)
    worst_imag = max(worst_imag, float(np.max(np.abs(ev.imag))))
    worst_spec = max(worst_spec, float(np.max(np.abs(np.sort(ev.real) - TARGET))))
    worst_self = max(worst_self, float(np.max(np.abs(GRAM @ W - (GRAM @ W).conj().T))))
print("  三个 k 方向上的谱与 {+2,+1,0,0,-1,-2} 的最大偏差 = %.3e" % worst_spec)
print("  本征值虚部最大 = %.3e；Frobenius 度量下自伴偏差 = %.3e" % (worst_imag, worst_self))
check("L5-1", worst_spec <= 1e-12 and worst_imag <= 1e-12 and worst_self <= 1e-12,
      "谱为六个整数且生成元自伴")

print()
print("=" * 78)
print("L5-2 TT 不变性与限制谱")
print("=" * 78)
k_hat = np.array([0.0, 0.0, 1.0])
W = spin_generator(k_hat)
P = tt_projector(k_hat)
comm = float(np.max(np.abs(P @ W - W @ P)))
W_tt = P @ W @ P
ev_tt = np.linalg.eigvals(W_tt)
nz = ev_tt[np.abs(ev_tt) > 1e-8]
trace_tt = float(np.sum(nz).real)
det_tt = float(np.prod(nz).real)
print("  交换子 |P W - W P| = %.3e（期望 0）" % comm)
print("  TT 子空间上的非零本征值 = %s；迹 %.6f、行列式 %.6f（期望 0 与 -4）"
      % (np.round(np.sort(nz.real), 6), trace_tt, det_tt))
check("L5-2", comm <= 1e-14 and abs(trace_tt) <= 1e-10 and abs(det_tt + 4.0) <= 1e-8
      and float(np.max(np.abs(np.sort(nz.real) - np.array([-2.0, 2.0])))) <= 1e-10,
      "TT 投影与生成元交换，限制谱为 {+2,-2}")

print()
print("=" * 78)
print("L5-3 本征矢的偏振身份")
print("=" * 78)
vals, vecs = np.linalg.eig(W_tt)
h_plus = sym_basis(0, 0) - sym_basis(1, 1)
h_cross = sym_basis(0, 1)
worst_phase = 0.0
for n in range(6):
    if abs(vals[n]) < 1e-8:
        continue
    h = mat_from_coeffs(vecs[:, n])
    c_plus = 0.25 * np.sum(h * h_plus)
    c_cross = 0.25 * np.sum(h * h_cross)
    ratio = c_cross / c_plus
    worst_phase = max(worst_phase, min(abs(ratio - 1j), abs(ratio + 1j)))
print("  本征矢在 (h_+, h_x) 基下的分量比与 +-i 的偏差 = %.3e" % worst_phase)
check("L5-3", worst_phase <= 1e-12, "本征态即圆偏振态 (h_+ +- i h_x)/sqrt(2)")

print()
print("=" * 78)
print("L5-4 约束子空间的螺旋度身份")
print("=" * 78)
v1 = np.array([0.0, -1.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0])
named = [("迹", np.eye(3)),
         ("沿 k 的无迹纵向-纵向", np.outer(k_hat, k_hat) - np.eye(3) / 3.0)]
worst_zero = 0.0
for name, M in named:
    cv = coeffs(M)
    val = float(np.real(cv.conj() @ (W @ cv)) / float(cv.conj() @ cv))
    worst_zero = max(worst_zero, abs(val))
    print("  %s：螺旋度期望值 = %.3e" % (name, val))
cv1 = coeffs(np.outer(k_hat, v1) + np.outer(v1, k_hat))
cv2 = coeffs(np.outer(k_hat, v2) + np.outer(v2, k_hat))
Bmat = np.array([cv1, cv2]).T
ev_sub = np.sort(np.linalg.eigvals(np.linalg.pinv(Bmat) @ (W @ Bmat)).real)
print("  纵向-横向二维子空间上的本征值 = %s（期望 +-1）" % np.round(ev_sub, 6))
check("L5-4", worst_zero <= 1e-12
      and float(np.max(np.abs(ev_sub - np.array([-1.0, 1.0])))) <= 1e-10,
      "迹与无迹纵向-纵向为 0，纵向-横向为 +-1")

print()
print("=" * 78)
print("L5-5 传播与简并：平面波模式上的螺旋度守恒（含灵敏度对照）")
print("=" * 78)
NZ = 64
KZK = 2.0 * np.pi * 4.0 / NZ
gzz = np.arange(NZ)
mode = np.exp(1j * KZK * gzz)


def state_vector(target):
    for n in range(6):
        if abs(vals[n] - target) < 1e-8:
            v = vecs[:, n].copy()
            return v / np.sqrt(np.real(v.conj() @ (GRAM @ v)))
    raise RuntimeError("no eigenvector")


SP = state_vector(2.0)
SM = state_vector(-2.0)


def make_field(vec):
    field = np.zeros((6, NZ), dtype=complex)
    for idx in range(6):
        field[idx] = vec[idx] * mode
    return field


def content(field, vec):
    g = np.abs(vec.conj() @ (GRAM @ field))
    return float(np.linalg.norm(g))


def run(vec, steps=40, mix=0.0):
    field = make_field(vec)
    prev = make_field(vec * np.exp(-1j * KZK))
    for _ in range(steps):
        lap = np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 2.0 * field
        nxt = 2.0 * field - prev + lap
        if mix != 0.0:
            nxt[0] = nxt[0] + mix * field[1]
            nxt[1] = nxt[1] + mix * field[0]
        prev, field = field, nxt
    return field


def phase_advance(vec, steps=40, mix=0.0):
    f_init = make_field(vec)
    f_final = run(vec, steps=steps, mix=mix)
    return float(np.angle(f_final[0][0] / f_init[0][0]))


expect = float(((-40.0 * KZK) + np.pi) % (2.0 * np.pi) - np.pi)
for label, mix in (("各向同性演化", 0.0), ("加张量混合项（灵敏度对照）", 0.05)):
    field = run(SP, steps=40, mix=mix)
    a_plus = content(field, SP)
    a_minus = content(field, SM)
    tt_part = float(np.linalg.norm(P @ field))
    total = float(np.linalg.norm(field))
    adv = phase_advance(SP, mix=mix)
    print("  %s：+2 分量 %.6e，反分量相对幅度 %.3e，TT 占比 %.6f，相位推进 %+.6f rad（预测 %+.6f）"
          % (label, a_plus, a_minus / a_plus, tt_part / total, adv, expect))
    if mix == 0.0:
        check("L5-5a", a_minus / a_plus <= 1e-12 and abs(tt_part / total - 1.0) <= 1e-10
              and abs(adv - expect) <= 1e-12,
              "螺旋度与 TT 性保持（反分量 %.1e）" % (a_minus / a_plus))
    else:
        check("L5-5b", a_minus / a_plus >= 1e-3,
              "对照落在可测量的混合上（反分量 %.1e）" % (a_minus / a_plus))

adv_p = phase_advance(SP)
adv_m = phase_advance(SM)
gap = float((adv_p - adv_m + np.pi) % (2.0 * np.pi) - np.pi)
print("  两个螺旋度的相位推进 %+.9f 与 %+.9f rad，之差 %.3e（期望 0）" % (adv_p, adv_m, gap))
check("L5-5c", abs(gap) <= 1e-12, "两个螺旋度同速传播且无混合")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-10s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)
