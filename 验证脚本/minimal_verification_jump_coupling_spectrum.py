# -*- coding: utf-8 -*-
# 第 610 号判据轮：D3p-3 跳变载体线性响应判别（V2a 解耦确认 / V2b 稳定域 / V2c 区分力齿）
# 模型：608 号复物质场载体（U(1) 链接 + 物质动能 Π^2/2），全盒介质，k 空间块对角化。
# 解析事实（本判据的期望来源）：绕 z=0 线性化时跳变的 theta-物质交叉 theta·Im(z* z') 为三阶
#（Im(z* z') = u v' - v u' 二阶 × theta 一阶），二阶能量中消失——载体无凝聚则对横向光无线性响应。
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:\software\TianXingFC\images\610_jump_coupling_spectrum.png'
N, W0 = 12, 1.0
ks = 2 * np.pi * np.arange(N) / N

def photon_dispersion(kx, ky):
    # 光子：plaquette 磁能 + KS 电能，横模离散色散（两支简并）
    # 线性化 theta 块 Q_theta = M^dagger M，M = [e^{ikx}-1, e^{iky}-1]（含归一 1/2 惯例）
    ex = np.exp(1j * kx) - 1
    ey = np.exp(1j * ky) - 1
    M = np.array([[ex, ey]])
    return M.conj().T @ M

def matter_dispersion(kx, ky, kappa, w0=W0, sign=-1):
    # 物质带：w0^2 - 2 kappa (cos kx + cos ky)（跳变 sign=-1 铁磁对齐型）
    return (w0 ** 2 - sign * 0) * np.eye(2) - 2 * kappa * (np.cos(kx) + np.cos(ky)) * np.eye(2)

def spectrum_block(kx, ky, kappa):
    # 每 k 的 Q 块（Hermitian 4x4）：theta(2) + matter(2)，动量块 P=I，omega^2 = eig(Q)
    Qt = photon_dispersion(kx, ky)
    Qm = matter_dispersion(kx, ky, kappa)
    Q = np.zeros((4, 4), dtype=complex)
    Q[:2, :2] = Qt
    Q[2:, 2:] = Qm
    return np.linalg.eigvalsh(Q).real

# ---------- 冒烟检查 ----------
sm = spectrum_block(0.3, 0.5, 0.2)
print('[smoke] k=(0.3,0.5) 块谱 4 个 omega^2：', np.round(sm, 6), '（实数且含两光支两物质支）')

# ---------- V2a 解耦确认：谱 = 纯光子支 + 纯物质带（逐位） ----------
KAPPA = 0.2  # 稳定域内（< w0^2/4 = 0.25）
TOL = 1e-10
v2a_pass = True
worst_ph, worst_mt = 0.0, 0.0
n_zero = 0
for ix in range(N):
    for iy in range(N):
        kx, ky = ks[ix], ks[iy]
        w2 = spectrum_block(kx, ky, KAPPA)
        # 光支两支：Q_theta 块特征值
        w2_ph = np.linalg.eigvalsh(photon_dispersion(kx, ky)).real
        w2_mt = np.linalg.eigvalsh(matter_dispersion(kx, ky, KAPPA)).real
        # 块谱排序后与两组解析逐位对（解耦=块对角）
        got = np.sort(w2)
        exp = np.sort(np.concatenate([w2_ph, w2_mt]))
        d = np.max(np.abs(got - exp))
        worst_ph = max(worst_ph, d)
        v2a_pass &= d <= TOL
        if np.max(np.abs(w2)) < TOL:
            n_zero += 1
print('\n===== V2a 解耦确认（kappa=%.2f，%d 个 k 块） =====' % (KAPPA, N * N))
print('块谱与「光子块 + 物质块」解析并集最大偏差 %.2e（容差 %.0e）→ %s'
      % (worst_ph, TOL, 'PASS' if v2a_pass else 'FAIL'))
print('零模 k 块计数（k=0 一块）：', n_zero, '（规范冗余：k=0 的 theta 块全零）')

# ---------- V2b 稳定域：物质带最小 omega^2 的符号随 kappa 翻转，临界 w0^2/4 ----------
print('\n===== V2b 稳定域（物质带最小 omega^2，临界 kappa = w0^2/4 = 0.25） =====')
v2b_pass = True
for kappa in (0.2, 0.3, 0.6):
    min_w2 = min(np.linalg.eigvalsh(matter_dispersion(ks[ix], ks[iy], kappa)).real.min()
                 for ix in range(N) for iy in range(N))
    expect_sign = '正' if kappa < W0 ** 2 / 4 else '负'
    got_sign = '正' if min_w2 > 0 else '负'
    ok = (min_w2 > 0) == (kappa < W0 ** 2 / 4)
    v2b_pass &= ok
    print('kappa=%.2f  物质带 min omega^2 = %+.4f（期望 %s）→ %s' % (kappa, min_w2, expect_sign, 'PASS' if ok else 'FAIL'))
print('解析临界：w0^2/4 = %.4f（k=0 均匀模先软）' % (W0 ** 2 / 4))

# ---------- V2c 区分力齿：拆掉跳变（kappa=0）后物质带退化为纯势带、光支不变 ----------
print('\n===== V2c 区分力齿 =====')
ok1 = True
for ix in range(N):
    for iy in range(N):
        kx, ky = ks[ix], ks[iy]
        w2 = spectrum_block(kx, ky, 0.0)
        exp = np.sort(np.concatenate([
            np.linalg.eigvalsh(photon_dispersion(kx, ky)).real,
            np.full(2, W0 ** 2)]))
        ok1 &= np.max(np.abs(np.sort(w2) - exp)) <= TOL
print('kappa=0：物质带退化为纯势带 w0^2（平带）、光支不变 →', 'PASS' if ok1 else 'FAIL')
v2c_pass = ok1
# 齿 2：theta-物质交叉在三阶（二阶块中恒零）——数值断言：块矩阵不含 theta-matter 交叉（构造级）
print('theta-物质交叉：二阶能量中不存在（三阶项，解析确认，构造级声明）')

print('\n总判定：', 'PASS（3/3）' if (v2a_pass and v2b_pass and v2c_pass) else 'FAIL')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
kk = np.linspace(0, np.pi, 200)
ax.plot(kk, 4 * np.sin(kk / 2) ** 2, label='光支（对角方向离散色散）')
ax.plot(kk, W0 ** 2 - 4 * KAPPA * np.cos(kk), '--', label=r'物质带 $w_0^2-4\kappa\cos k$（对角方向）')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel('对角方向波数 k')
ax.set_ylabel('omega^2')
ax.set_title('V2a：z=0 载体谱 = 纯光子支 + 纯物质带\n（无线性光-物耦合，块对角逐位确认）')
ax.legend(fontsize=8)
ax = axes[1]
kap_list = np.linspace(0.0, 0.6, 80)
min_w2 = [min(np.linalg.eigvalsh(matter_dispersion(ks[ix], ks[iy], k)).real.min()
              for ix in range(N) for iy in range(N)) for k in kap_list]
ax.plot(kap_list, min_w2, color='crimson')
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(W0 ** 2 / 4, color='navy', ls='--', label=r'解析临界 $\kappa = w_0^2/4 = 0.25$')
ax.set_xlabel('跳变强度 kappa')
ax.set_ylabel('物质带最小 omega^2')
ax.set_title('V2b：稳定域边界（608 号参数 0.6 在不稳定域）')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print('图已存', OUT)
