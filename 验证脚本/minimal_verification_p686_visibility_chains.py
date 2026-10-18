# -*- coding: utf-8 -*-
r'''可见度口径的两条链对齐（判据 686 号）
=====================================================================
背景：库内存在两条独立的可见度链，此前未对过口径。

  链 A（记录机制侧，第 391 条）
    V_A = exp(-Delta^2/4) = exp(-D/2),  D = n_diff * G^2 / 2
    出身：相干态重叠，逐键因子化，指数 1/4 = (1/2)*(1/2) 两半各有出处。
    这是「偏离到可见度」的换算，D 是相空间距离。

  链 B（退相干侧，第 520 条 / 满周升格 §4.1）
    V_B(t) = (1 + 2 Gamma_coh t)^(-1/2)
    出身：高斯相位扩散的系综平均。

本轮问题：两条链是否同一对象？若是，D 与 Gamma_coh*t 应能互表；
若否，须写明各自适用条件与不可互代的边界。

理论期望（先立）：
  E1  链 A 是指数型（ln V 线性于 D），链 B 是代数型（ln V 非线性于 t）。
      二者不是同一函数，故不能在任意时刻互代。
  E2  小量极限下二者可对齐到一阶：D -> 0 时 V_A = 1 - D/2 + O(D^2)；
      Gamma t -> 0 时 V_B = 1 - Gamma t + O(t^2)。故对齐条件是
      D/2 = Gamma_coh t，即 D = 2 Gamma_coh t。
  E3  对齐只在一阶成立。以 D = 2Gt 代入后，两式在 D = 1 处差
      约 5%，D = 2 处差约 12%，D 增大后发散。
  E4  两链的物理对象不同：A 量的是「寄存器偏离的历史残差」（空间侧，
      一次读数的位移），B 量的是「相对相位的累积扩散」（时间侧）。
      二者可对齐是体系内两条独立通道在短时极限的相容，不是同一机制的两种写法。

判据：
  C1  两链函数形式不同（不是同一函数）：在 D = 2Gt 映射下，V_A 与 V_B
      在 t 上最大相对差 >= 0.1（取 Gamma*t 覆盖 [0, 5]）。
  C2  一阶对齐：D -> 0 时 V_A 与 V_B 的一阶系数相同（都是 -1）。
      |V_A(eps) - V_B(eps)| / eps <= 1e-3，eps = 1e-4。
  C3  对齐式 D = 2*Gamma_coh*t 在小量档成立：取 D <= 0.01，两式最大相对差
      <= 1e-2（容差预算：二阶项 (D/2)^2/2 在 D=0.01 处给 1.25e-5）。
  C4  偏离随 D 增长：D = 0.5、1、2、4 四档的相对差单调增，且 D=2 档 >= 0.1。
  C5  链 A 的逐键因子化（沿用第 391 条口径）：n_diff 档下 V_A 等于
      逐键乘积，偏差 <= 1e-12。确认链 A 内部自洽，不是本轮新结论。
  C6  恒真反例（EP5）：把 V_B 替身为 exp(-Gamma t)（即链 A 的形式）后，
      C1 必须 FAIL。若仍 PASS 则 C1 不读函数形状，整轮无效。
  C7  EP3 数值半宽：C3 的容差与浮点噪声的比值，确认不是机器噪声。

可逆性等级：
  C1、C2、C4 判据稳健（代数恒等，换参数不翻）。
  C3 判据敏感（小量档阈值决定容差，改档会翻）。
  C5 判据稳健（沿用既有结论）。C6 恒真对照。C7 读数。

参数与确定性：无随机数（纯代数），确定性。
图存 images/p686_visibility_chains.png。
====================================================================='''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PASS, FAIL = 'PASS', 'FAIL'
results = []


def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')


# ---------------- 两条链 ----------------
def v_chain_a(D):
    """链 A（第 391 条）：相干态重叠，D 为相空间距离。"""
    return np.exp(-0.5 * np.asarray(D, dtype=float))


def v_chain_b(Gt):
    """链 B（第 520 条）：高斯相位扩散的系综平均。"""
    return (1.0 + 2.0 * np.asarray(Gt, dtype=float)) ** (-0.5)


# ---------------- C1 两链不是同一函数 ----------------
Gt = np.linspace(0.0, 5.0, 400)
D_map = 2.0 * Gt          # 由 C2 的一阶对齐式给出
va = v_chain_a(D_map)
vb = v_chain_b(Gt)
denom = np.maximum(np.abs(vb), 1e-12)
dev = np.max(np.abs(va - vb) / denom)
report('C1 两链函数形式不同', dev >= 0.1,
       f'在 D=2Gt 映射下最大相对差 {dev:.3f}（阈值 0.1）')

# ---------------- C2 一阶对齐 ----------------
eps = 1e-4
va1 = v_chain_a(eps)
vb1 = v_chain_b(eps / 2.0)      # D = 2Gt -> Gt = D/2
diff = abs(va1 - vb1) / eps
report('C2 一阶系数相同（均为 -1）', diff <= 1e-3,
       f'eps={eps:g}：V_A={va1:.12f} V_B={vb1:.12f}，差/eps={diff:.2e}')

# ---------------- C3 小量档对齐 ----------------
Ds = np.array([1e-4, 1e-3, 1e-2])
va3 = v_chain_a(Ds)
vb3 = v_chain_b(Ds / 2.0)
dev3 = np.max(np.abs(va3 - vb3) / vb3)
report('C3 小量档对齐成立', dev3 <= 1e-2,
       f'D<=1e-2 最大相对差 {dev3:.2e}（预算 1.25e-5 的二阶项量级）')

# ---------------- C4 偏离随 D 增长 ----------------
Dlist = np.array([0.5, 1.0, 2.0, 4.0])
va4 = v_chain_a(Dlist)
vb4 = v_chain_b(Dlist / 2.0)
dev4 = np.abs(va4 - vb4) / vb4
mono = np.all(np.diff(dev4) > 0)
report('C4 偏离随 D 单调增长且 D=2 档 >= 0.1', bool(mono) and dev4[2] >= 0.1,
       '; '.join(f'D={d:g} 差 {e:.3f}' for d, e in zip(Dlist, dev4)))

# ---------------- C5 链 A 逐键因子化 ----------------
ok = True
det = []
for n_diff in [1, 2, 4, 6]:
    G = 1.0
    D_tot = n_diff * G ** 2 / 2.0
    v_tot = v_chain_a(D_tot)
    v_prod = v_chain_a(G ** 2 / 2.0) ** n_diff
    rel = abs(v_tot - v_prod) / v_tot
    ok &= rel <= 1e-12
    det.append(f'n={n_diff} 差 {rel:.1e}')
report('C5 链 A 逐键因子化（沿用第 391 条）', ok, '; '.join(det))

# ---------------- C6 恒真反例 ----------------
vb_exp = np.exp(-Gt)
dev_bad = np.max(np.abs(va - vb_exp) / vb_exp)
report('C6 恒真反例（exp 替身须使 C1 FAIL）', dev_bad < 0.1,
       f'exp 替身对 V_A 的最大相对差 {dev_bad:.2e}（须 <0.1 才算替身确实混同）')

# ---------------- C7 浮点噪声比 ----------------
noise = np.finfo(float).eps
ratio = 1e-2 / noise
print(f'[READ] C7 容差/浮点噪声 = {ratio:.3e}（远大于 1，C3 不可能是机器噪声）')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
tt = np.linspace(0, 5, 400)
ax.plot(tt, v_chain_b(tt), '-', lw=1.8, label=r'链 B：$(1+2\Gamma t)^{-1/2}$')
ax.plot(tt, v_chain_a(2 * tt), '--', lw=1.8, label=r'链 A：$e^{-D/2}$，$D=2\Gamma t$')
ax.plot(tt, np.exp(-tt), 'k:', lw=1.0, label=r'$e^{-\Gamma t}$')
ax.set_xlabel(r'$\Gamma t$')
ax.set_ylabel('可见度')
ax.set_title('同一映射下两链的形状不同')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(Dlist, dev4, 'o-', lw=1.6)
ax.axhline(0.1, color='r', ls=':', lw=1.0, label='0.1 参考线')
ax.set_xlabel(r'$D = 2\Gamma t$')
ax.set_ylabel('两链相对差')
ax.set_title('偏离随 D 增长（短时对齐、长时发散）')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()

out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p686_visibility_chains.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
