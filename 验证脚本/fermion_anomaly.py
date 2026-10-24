# -*- coding: utf-8 -*-
"""P4-31 遗留之五：极简费米子扇区模型——缺陷零模与体侧拓扑的配对（F1 至 F5）。

过程记录：初版把 Wilson 项乘了单位矩阵（等效于给色散加一个随动量移动的常数，带隙不开、
零模找不到，判据 F1、F2、F4、F3 全判负）；改为与质量项同结构（乘 sigma_z）后带隙打开、
零模出现。另有一处判据不匹配：初版以 |E| <= 1e-8 定零模，未先核对谱的带隙存在。

定位：手征配对相容性判定的前置项。本模型不计算本框架费米子扇区的反常系数，
它判定的是"配对机制"这一件事：在有拓扑扇区的结构里，缺陷处的零模是否可以由体侧的
拓扑量自动配平，因而不需要额外的宇称奇项。

模型：一维格点 Dirac（含 Wilson 项避免加倍），质量取空间变号函数 m(n) = m0 tanh((n-n0)/w)。
判据（先立判据，再实现）：
  F1 零模存在与局域化：kink 背景给出恰好一个零模（|E| <= 1e-8），且波函数局域在壁处。
  F2 指标定理：零模数等于质量符号差 |Delta sign m|，在幅度与宽度的扫描下不变。
  F3 体侧配对：质量符号差（体侧拓扑量）与零模数一一对应，故配对由拓扑自动给出。
  F4 手征读数：零模的 <sigma_z> 符号随 kink 方向翻转（同一计数、相反手征）。
  F5 灵敏度对照：均匀质量给出零模数 0；去掉 Wilson 项后出现加倍型多余零模（判负对照）。

单位：格距 1。确定性，无随机数。
"""

import numpy as np

CHECKS = []
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
EYE = np.eye(2, dtype=complex)
N = 200
N0 = N // 2


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [{}] {}  {}".format(tag, "PASS" if ok else "FAIL", detail))


def hamiltonian(mass, wilson=1.0):
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    for j in range(N - 1):
        a, b = 2 * j, 2 * (j + 1)
        hop = 0.5j * SX
        H[a:a + 2, b:b + 2] += hop
        H[b:b + 2, a:a + 2] += -hop
        H[a:a + 2, b:b + 2] += 0.5 * wilson * SZ
        H[b:b + 2, a:a + 2] += 0.5 * wilson * SZ
        H[a:a + 2, a:a + 2] += -0.5 * wilson * SZ
        H[b:b + 2, b:b + 2] += -0.5 * wilson * SZ
    for j in range(N):
        H[2 * j:2 * j + 2, 2 * j:2 * j + 2] += mass(j) * SZ
    return H


def spectrum(mass, wilson=1.0):
    H = hamiltonian(mass, wilson)
    ev = np.linalg.eigvalsh(H)
    return H, ev


def zero_modes(ev, tol=1e-8):
    return np.where(np.abs(ev) <= tol)[0]


def kink(m0, w, sign=1.0):
    def m(j):
        return sign * m0 * np.tanh((j - N0) / w)
    return m


print("=" * 78)
print("F1 零模存在与局域化")
print("=" * 78)
H, ev = spectrum(kink(0.5, 4.0))
idx = zero_modes(ev)
print("  最低四档能量：{}".format(np.round(ev[:4], 12)))
print("  |E| <= 1e-8 的态数 = {}（期望 1）".format(len(idx)))
amp = {}
if len(idx) >= 1:
    v = np.linalg.eigh(H)[1][:, idx[0]]
    dens = np.abs(v) ** 2
    site = dens[0::2] + dens[1::2]
    part = 1.0 / float(np.sum(site ** 2))
    amp["参与比"] = part
    amp["峰值位置"] = int(np.argmax(site))
    print("  零模：参与比 = {:.2f} 格，峰值位置 = {}（壁在 {}）".format(part, amp["峰值位置"], N0))
check("F1", len(idx) == 1 and amp.get("参与比", 1e9) <= 30.0,
      "kink 给出单一零模且局域在壁处")

print()
print("=" * 78)
print("F2 指标定理与 F4 手征读数")
print("=" * 78)
rows = []
for m0 in (0.2, 0.5, 1.0):
    for w in (2.0, 4.0, 8.0):
        for sgn in (1.0, -1.0):
            Hk, evk = spectrum(kink(m0, w, sgn))
            z = zero_modes(evk)
            chi = 0.0
            if len(z) >= 1:
                v = np.linalg.eigh(Hk)[1][:, z[0]]
                a = v[0::2]; b = v[1::2]
                chi = float(np.sum(np.abs(a) ** 2) - np.sum(np.abs(b) ** 2))
            rows.append((m0, w, sgn, len(z), chi))
            print("  m0 = {:.1f}，w = {:.1f}，方向 {:+.0f}：零模数 = {}，<sigma_z> = {:+.6f}"
                  .format(m0, w, sgn, len(z), chi))
n_zm = set(r[3] for r in rows)
chi_pos = [r[4] for r in rows if r[2] > 0]
chi_neg = [r[4] for r in rows if r[2] < 0]
check("F2", n_zm == {1}, "零模数在九组幅度与宽度扫描下恒为 1")
check("F4", min(chi_pos) > 0.5 and max(chi_neg) < -0.5,
      "零模的手征符号随 kink 方向整体翻转（正 {} 组、负 {} 组）"
      .format(len(chi_pos), len(chi_neg)))

print()
print("=" * 78)
print("F3 体侧配对")
print("=" * 78)
_, ev_uni = spectrum(lambda j: 0.5)
print("  均匀质量（体侧同一符号，符号差 0）：零模数 = {}（期望 0）".format(len(zero_modes(ev_uni))))
check("F3", len(zero_modes(ev_uni)) == 0 and n_zm == {1},
      "零模数等于质量符号差，配对由体侧拓扑量给出")

print()
print("=" * 78)
print("F5 灵敏度对照")
print("=" * 78)
_, ev_no_w = spectrum(kink(0.5, 4.0), wilson=0.0)
z_no_w = zero_modes(ev_no_w, tol=1e-6)
print("  去掉 Wilson 项：|E| <= 1e-6 的态数 = {}（加倍型多余态出现）".format(len(z_no_w)))
check("F5", len(zero_modes(ev_uni)) == 0 and len(z_no_w) >= 2,
      "均匀质量与去 Wilson 两种对照都能让判据变化")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：{}/{} 通过".format(n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  {:6s} {}".format(tag, "PASS" if ok else "FAIL"))
print("=" * 78)
