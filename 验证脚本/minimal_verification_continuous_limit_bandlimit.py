# -*- coding: utf-8 -*-
r"""
minimal_verification_continuous_limit_bandlimit.py

[连续极限与带限轮] 尼奎斯特界与晶格带边各管什么：可分辨与存在的分野

定位
----
对应开放问题 P3-12 判据 (iii)（连续极限与带限）与 P3-26（尼奎斯特界与晶格带边的调和）。
两句待核的话：其一，格距趋零时阶梯读数还原连续螺旋；其二，可分辨频率上界（半个普朗克频率）
与晶格色散带边（$2\sqrt{3}/t_P$）是两个不同来源的界限，需说明谁管谁。
本脚本不改动基准文件；不含新科学断言。

模型
----
晶格色散取《元相子振荡晶格的色散关系》§3 的精确式（单位 $l_P = 1$、$t_P = 1$、$c = l_P/t_P = 1$，故 $\omega_P = 2\pi$）：
  $\omega^2(\mathbf{k}) = \dfrac{4}{t_P^2}\sum_{a=x,y,z}\sin^2\dfrac{k_a l_P}{2}$，
布里渊区 [111] 角点 $k_a = \pi/l_P$ 给出 $\omega_{\max} = 2\sqrt{3}/t_P = (\sqrt{3}/\pi)\,\omega_P \approx 0.5513\,\omega_P$。
采样：读数发生在离散节点 $t_n = n\,t_P$（物理时间），采样率 $\omega_s = 2\pi/t_P = \omega_P$，
尼奎斯特上界 $\omega_N = \omega_s/2 = \pi/t_P \approx 0.5\,\omega_P$。
阶梯与连续：$Z_{\text{粒}}(t) = l_P\lfloor \omega_P t/2\pi\rfloor$ 对连续 $Z_{\text{连续}}(t) = c\,t$（因 $l_P\omega_P/2\pi = c$）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 晶格带边：沿 [111] 方向数值求 $\omega(\mathbf{k})$ 的极大值 = $2\sqrt{3}/t_P$（相对误差 $<10^{-12}$）；
   换算 $\omega_{\max}/\omega_P = \sqrt{3}/\pi = 0.5513 \pm 0.001$（与色散关系 §3 的 0.55 一致）。
J2 尼奎斯特界：$\omega = \omega_P$ 的采样序列恒为常数（最大偏差 $<10^{-12}$，混叠到直流）；
   $\omega = 0.5\,\omega_P$ 的采样序列与常数序列显著不同（振幅 >= 1.0）。
J3 混叠映射：$\omega \in (\omega_N, \omega_P)$ 的采样序列与 $\omega' = \omega_P - \omega$ 逐一相同（最大偏差 $<10^{-12}$），三对例子。
J4 存在但不可分辨（P3-26 调和）：$\omega = 0.52\,\omega_P$ 满足 $\omega < \omega_{\max}$（存在），
   且其采样序列与 $\omega' = 0.48\,\omega_P$ 完全相同（不可分辨）；不可分辨频段宽度 $= (2\sqrt{3}-\pi)/t_P$，
   占带边 $9.31\%\pm0.05\%$，两界之比 $2\sqrt{3}/\pi = 1.1027 \pm 0.001$。
J5 连续极限：阶梯对连续直线的最大偏差 = $l_P$（精确）；细化 $l_P \to l_P/2^k$、$t_P \to t_P/2^k$（保持 $c$）下
   该偏差按 $2^{-k}$ 缩小（五点双对数斜率 $-1.0 \pm 0.05$）。
J6 低能极限：$\omega(\mathbf{k})$ 对 $c|\mathbf{k}|$ 的相对偏差随 $k \to 0$ 按 $k^2$ 收敛（双对数斜率 $2.0 \pm 0.1$），
   即首阶偏离就是洛伦兹破坏的 $(kl_P)^2$ 签名。

运行
----
python 验证脚本/minimal_verification_continuous_limit_bandlimit.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/continuous_limit_bandlimit.png
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []
LP, TP = 1.0, 1.0
WP = 2 * np.pi / TP
C = LP / TP
WN = WP / 2


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def omega(kvec):
    """晶格色散（色散关系 §3 精确式）。kvec 以 1/l_P 为单位。"""
    k = np.asarray(kvec, dtype=float)
    return np.sqrt(4.0 / TP ** 2 * np.sum(np.sin(k * LP / 2.0) ** 2))


print("== J1 晶格带边 ==")
k_corner = np.array([np.pi / LP] * 3)
w_corner = omega(k_corner)
w_max_expect = 2 * np.sqrt(3) / TP
rel = abs(w_corner - w_max_expect) / w_max_expect
ratio_band = w_corner / WP
print(f"  ω([111] 角点) = {w_corner:.12f}（期望 2√3/t_P = {w_max_expect:.12f}，相对误差 {rel:.2e}）")
print(f"  ω_max/ω_P = {ratio_band:.6f}（期望 √3/π = {np.sqrt(3)/np.pi:.6f}）")
check("J1a 带边数值", rel < 1e-12, f"相对误差 {rel:.2e}（阈值 1e-12）")
check("J1b 带边换算", abs(ratio_band - np.sqrt(3) / np.pi) < 1e-3, f"ω_max/ω_P = {ratio_band:.6f}（期望 0.5513 ± 0.001）")

print("== J2 尼奎斯特界 ==")
K = 64
n = np.arange(K)
s_wp = np.cos(WP * n * TP)
s_half = np.cos(0.5 * WP * n * TP)
d_wp = np.max(np.abs(s_wp - 1.0))
amp_half = np.max(np.abs(s_half - np.mean(s_half)))
print(f"  ω = ω_P 的采样序列对常数的最大偏差 = {d_wp:.2e}（期望 < 1e-12，混叠到直流）")
print(f"  ω = 0.5ω_P 的序列振幅 = {amp_half:.6f}（期望 >= 1.0，可分辨）")
check("J2a 整频混叠", d_wp < 1e-12, f"最大偏差 {d_wp:.2e}")
check("J2b 半频可分辨", amp_half >= 1.0, f"振幅 {amp_half:.6f}")

print("== J3 混叠映射 ==")
pairs = [(0.55, 0.45), (0.70, 0.30), (0.90, 0.10)]
devs = []
for f1, f2 in pairs:
    a = np.cos(f1 * WP * n * TP)
    b = np.cos(f2 * WP * n * TP)
    devs.append(np.max(np.abs(a - b)))
print(f"  三对频率（占 ω_P 比例）（0.55,0.45）、(0.70,0.30)、(0.90,0.10) 的序列最大偏差 = {['%.2e' % d for d in devs]}")
check("J3 混叠映射", max(devs) < 1e-12, f"最大偏差 {max(devs):.2e}（阈值 1e-12）")

print("== J4 存在但不可分辨 ==")
w_test = 0.52 * WP
w_alias = WP - w_test
exists = w_test < w_corner
same = np.max(np.abs(np.cos(w_test * n * TP) - np.cos(w_alias * n * TP)))
gap = (w_max_expect - WN)
gap_frac = gap / w_corner
ratio_two = w_max_expect / WN
print(f"  ω = 0.52ω_P = {w_test:.6f}，带边 = {w_corner:.6f}，存在性 = {exists}")
print(f"  与 ω' = {w_alias/WP:.2f}ω_P 的序列最大偏差 = {same:.2e}")
print(f"  不可分辨频段宽度 = {gap:.6f}/t_P，占带边 {gap_frac*100:.2f}%，两界之比 = {ratio_two:.6f}")
check("J4a 频段存在", exists and gap > 0, f"带边 {w_corner:.4f} > 尼奎斯特 {WN:.4f}，频段宽 {gap:.4f}")
check("J4b 不可分辨", same < 1e-12, f"与 0.48ω_P 的序列偏差 {same:.2e}")
check("J4c 两界之比", abs(ratio_two - 2 * np.sqrt(3) / np.pi) < 1e-3, f"比值 {ratio_two:.6f}（期望 1.1027 ± 0.001），占带边 {gap_frac*100:.2f}%")

print("== J5 连续极限 ==")
T = 20.0 * TP
ks = list(range(0, 5))
sup_dev = []
for k in ks:
    lp, tp = LP / 2 ** k, TP / 2 ** k
    wp = 2 * np.pi / tp
    c = lp / tp
    t = np.linspace(0, T, 200001)
    stair = lp * np.floor(wp * t / (2 * np.pi))
    sup_dev.append(np.max(np.abs(stair - c * t)))
slope5 = np.polyfit(ks, np.log2(sup_dev), 1)[0]
print(f"  细化 k = {ks}：最大偏差 = {['%.6f' % s for s in sup_dev]}（期望 l_P/2^k = {['%.6f' % (1/2**k) for k in ks]}）")
print(f"  以 2 为底的对数斜率 = {slope5:.4f}（期望 -1.0）")
check("J5a 单格偏差 = l_P", abs(sup_dev[0] - LP) < 1e-6, f"k=0 偏差 {sup_dev[0]:.6f}（期望 {LP:.6f}）")
check("J5b 细化收敛", abs(slope5 + 1.0) <= 0.05, f"斜率 {slope5:.4f}（期望 -1.0 ± 0.05）")

print("== J6 低能极限 ==")
kks = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
reldev = []
for kk in kks:
    kv = np.array([kk, kk, kk]) / np.sqrt(3)   # 沿 [111] 方向，|k| = kk
    reldev.append(abs(omega(kv) / (C * kk) - 1))
slope6 = np.polyfit(np.log(kks), np.log(reldev), 1)[0]
print(f"  相对偏差 = {['%.3e' % r for r in reldev]}，双对数斜率 = {slope6:.4f}（期望 2.0）")
check("J6 低能极限", abs(slope6 - 2.0) <= 0.1, f"斜率 {slope6:.4f}（期望 2.0 ± 0.1）")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0][0]
    kk = np.linspace(0, np.pi / LP, 400)
    ww = [omega([k / np.sqrt(3)] * 3) for k in kk]
    ax.plot(kk, ww, color="#2563eb", lw=2, label="ω(k) 沿 [111]")
    ax.axhline(WN, color="#0f766e", ls="--", lw=1.4, label=f"尼奎斯特界 ω_P/2 = {WN:.3f}")
    ax.axhline(w_corner, color="#be123c", ls=":", lw=1.6, label=f"晶格带边 2√3/t_P = {w_corner:.3f}")
    ax.fill_between([0, np.pi / LP], WN, w_corner, color="#fde68a", alpha=0.45, label="存在但不可分辨")
    ax.set_title("色散与两条界限")
    ax.set_xlabel("|k| （1/l_P）")
    ax.set_ylabel("ω")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    t = np.linspace(0, 8 * TP, 2000)
    ax.plot(t, np.cos(0.52 * WP * t), color="#2563eb", lw=1.6, label="ω = 0.52ω_P")
    ax.plot(t, np.cos(0.48 * WP * t), color="#be123c", lw=1.0, ls="--", label="ω = 0.48ω_P（混叠伙伴）")
    ax.plot(n[:17] * TP, np.cos(0.52 * WP * n[:17] * TP), "o", color="#1f2328", ms=4, label="采样点（t_n = n t_P）")
    ax.set_title("同一组采样点，两条不同频率")
    ax.set_xlabel("t （t_P）")
    ax.set_ylabel("读数")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.loglog([2 ** k for k in ks], sup_dev, "o-", color="#b45309", label="最大偏差")
    ax.loglog([2 ** k for k in ks], [1 / 2 ** k for k in ks], "--", color="#9aa1b0", label="l_P/2^k")
    ax.set_title("连续极限：细化下偏差按格距缩小")
    ax.set_xlabel("细化倍数 2^k")
    ax.set_ylabel("阶梯对直线的最大偏差")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.loglog(kks, reldev, "o-", color="#7c3aed", label="相对偏差")
    ax.set_title(f"低能极限：相对偏差 ∝ k^{slope6:.1f}")
    ax.set_xlabel("|k| （1/l_P）")
    ax.set_ylabel("|ω/(c k) − 1|")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/continuous_limit_bandlimit.png", dpi=160)
    print("  图已保存：images/continuous_limit_bandlimit.png")
except Exception as exc:
    print(f"  绘图跳过：{exc}")

passed = sum(1 for _, ok in CHECKS if ok)
print(f"== 判据合计 {passed}/{len(CHECKS)} 通过，用时 {time.time()-T0:.1f} s ==")
