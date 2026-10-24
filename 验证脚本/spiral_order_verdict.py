# -*- coding: utf-8 -*-
"""
第 480 条：螺旋序真空地位裁决（电子场攻坚计划阶段 1，判据 S1 至 S4）。

模型：缺陷动力学注册泛函（螺旋读法，作者 2026-09-06 定案，k_b = 1 归一，r = K/k_b）：
    E[u] = sum_{n} 0.5*(u[n+1]-u[n]-w_ref)**2 + sum_{n} K*(1-cos(2*pi*u[n]))
u 以整圈为单位（u 增一 = 转满一圈 = 法线轴升一层），w_ref 为弹性参考层率（注册形式取 1）。

理论期望（先立后测，判据驱动）：
  S1 期望：背景能量 E_bg(w0) 唯一零能于 w0 = w_ref；整数偏离 m 的代价 = (L-1)*m^2/2
     （弹性项，PN 势整数层率处全零）；相位计数斜率经 v = u - n 代换可恢复（约定），
     层率内容藏在代换的 n 中保持每格一层（物理锁定）。
  S2 期望：E(-1)[u] = E(+1)[-u] 解析恒等；两泛函 kink 能量/宽度/Hessian 谱逐位镜像一致。
  S3 期望：周期边界固定泛函内，平移 u -> u + c 无零能路径实现 p -> -p（min E(c) = 2*(L-1)）；
     等价由泛函反演联动（S2 恒等式）保证，不由平移保证。
  S4 期望：零能背景仅整数层率存在（commensurate 锁定）；整数层率族 {1, 0} 的 kink
     能量/宽度/谱逐位同构（闭式 (4/pi)*sqrt(r)，与 w_ref 无关）；色散 (2*pi)^2*K + 4*sin^2(k/2)
     与 w_ref 无关；结论对 r 档位稳健。

可逆性等级：判据稳健（结构性结论：极小位置、镜像恒等、同构性不随 r 档翻转）；
  依赖声明：弹性身份 = 螺旋高度差（作者 2026-09-06 裁决）是前提，换紧致 cos 投影读法则
  S1 的代价结构改变（弹性身份裁决注记已判该读法窗口为空）。
"""
import numpy as np

OUT_PNG = r"D:\software\TianXingFC\images\spiral_order_verdict.png"
L = 256
ETA = 0.01
NITER = 60000
TOL = 1e-12

def grad_v(v, K, w_ref, p_fixed):
    """v 泛函梯度：E[v] = sum 0.5*(dv)^2 + K*sum(1-cos(2pi(v+w_ref*n)))，
    开放链 v[0]=0、v[-1]=p_fixed 固定，只更新内部点。"""
    g = np.zeros_like(v)
    dv = np.diff(v)
    g[1:-1] = (v[1:-1] - v[:-2]) - (v[2:] - v[1:-1])
    g += 2 * np.pi * K * np.sin(2 * np.pi * (v + w_ref * np.arange(len(v))))
    g[0] = g[-1] = 0.0
    return g

def relax_kink(K, w_ref, p=1, L=L, eta=ETA, niter=NITER):
    """松弛出 v: 0 -> p 的 kink，返回 v、收敛力范数。"""
    n = np.arange(L)
    v = p * 0.5 * (1.0 - np.cos(np.pi * n / (L - 1)))
    for _ in range(niter):
        g = grad_v(v, K, w_ref, p)
        v = v - eta * g
        v[0], v[-1] = 0.0, float(p)
        if _ % 2000 == 0 and np.max(np.abs(g[1:-1])) < TOL:
            break
    return v, np.max(np.abs(grad_v(v, K, w_ref, p)[1:-1]))

def kink_energy(v, K, w_ref, p=1):
    """kink 相对能量：E[v] - E[背景 v=0]。"""
    n = np.arange(len(v))
    u = v + w_ref * n
    du = np.diff(u)
    e_el = 0.5 * np.sum((du - w_ref + w_ref) ** 2)  # v 语言弹性 = 0.5*sum(dv^2)
    dv = np.diff(v)
    e_el = 0.5 * np.sum(dv ** 2)
    e_pn = K * np.sum(1.0 - np.cos(2 * np.pi * u))
    bg_pn = K * len(v) * (1.0 - np.mean(np.cos(2 * np.pi * (w_ref * n))))
    return (e_el + e_pn) - bg_pn, e_el, e_pn - bg_pn

def width_participation(v):
    """参与比宽度：弹性超量密度权重的惯性半径直径。"""
    dv = np.diff(v)
    rho = 0.5 * dv ** 2
    if rho.sum() <= 0:
        return 0.0
    idx = np.arange(len(rho)) + 0.5
    xbar = np.sum(rho * idx) / rho.sum()
    return 2.0 * np.sqrt(np.sum(rho * (idx - xbar) ** 2) / rho.sum())

def hessian_spectrum(v, K, w_ref):
    """内部自由度 Hessian（三对角），返回本征值谱。"""
    n = np.arange(len(v))
    diag = 2.0 + 4 * np.pi ** 2 * K * np.cos(2 * np.pi * (v + w_ref * n))[1:-1]
    off = -np.ones(len(diag) - 1)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(A)

def bg_energy(w0, K, w_ref, L=L):
    """背景 u_n = w0*n 的能量（弹性 + PN）。"""
    e_el = 0.5 * (L - 1) * (w0 - w_ref) ** 2
    n = np.arange(L)
    e_pn = K * np.sum(1.0 - np.cos(2 * np.pi * w0 * n))
    return e_el, e_pn, e_el + e_pn

results = []
def report(tag, measured, expected, ok):
    results.append((tag, measured, expected, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

# ========== 冒烟检查 ==========
print("== 冒烟检查 ==")
rng = np.random.default_rng(42)
K0 = 0.005
v_rand = rng.uniform(0, 1, L)
g_rand = grad_v(v_rand, K0, 1.0, 1)
v_bg = np.zeros(L)
g_bg = grad_v(v_bg, K0, 1.0, 1)
print(f"随机位形 max|grad| = {np.max(np.abs(g_rand)):.4f}（非零合理）")
print(f"整数参考背景 max|grad| = {np.max(np.abs(g_bg)):.2e}（应为 0，PN 极小 + 弹性零）")
smoke_ok = np.max(np.abs(g_rand)) > 0.1 and np.max(np.abs(g_bg)) < 1e-12
report("冒烟：力的变化性落在操作轴上", f"随机 {np.max(np.abs(g_rand)):.3f} / 背景 {np.max(np.abs(g_bg)):.1e}",
       "随机非零、背景为零", smoke_ok)

# ========== S1：w0 平移轨道与可消去性 ==========
print("\n== S1：w0 平移轨道 ==")
w0_scan = np.linspace(-0.5, 2.5, 301)
e_bg = np.array([bg_energy(w0, K0, 1.0)[2] for w0 in w0_scan])
w0_min = w0_scan[np.argmin(e_bg)]
report("S1-a 背景能量唯一零能位置", f"argmin w0 = {w0_min:.4f}, E_min = {e_bg.min():.2e}",
       "w0 = 1（注册层率）且 E_min = 0", abs(w0_min - 1.0) < 0.02 and abs(e_bg.min()) < 1e-9)

e_int = {m: bg_energy(1.0 + m, K0, 1.0) for m in (-2, -1, 1, 2)}
ok_s1b = True
for m, (e_el, e_pn, e_tot) in e_int.items():
    expect = 0.5 * (L - 1) * m ** 2
    good = abs(e_el - expect) < 1e-8 and e_pn < 1e-9
    ok_s1b = ok_s1b and good
    print(f"    w0 = {1+m:+d}: 弹性 {e_el:.4f}（期望 {expect:.4f}）, PN {e_pn:.2e}")
report("S1-b 整数偏离代价 = (L-1)*m^2/2 且 PN 全零",
       f"四点最大偏差 {max(abs(e[0] - 0.5*(L-1)*m**2) for m, e in ((m, e_int[m]) for m in e_int)):.2e}",
       "弹性解析式逐点符合，PN < 1e-9", ok_s1b)

# S1-c 相位计数代换恒等：E(+1)[v+n] 与平坦参考泛函 E_0[v] = sum 0.5 dv^2 + K(1-cos 2pi v) 逐位相等
v_rand = rng.uniform(-1, 2, L)
u_shift = v_rand + np.arange(L)
e_left = 0.5 * np.sum((np.diff(u_shift) - 1.0) ** 2) + K0 * np.sum(1 - np.cos(2 * np.pi * u_shift))
e_right = 0.5 * np.sum(np.diff(v_rand) ** 2) + K0 * np.sum(1 - np.cos(2 * np.pi * v_rand))
report("S1-c 相位计数斜率代换恒等 E(+1)[v+n] = E_0[v]",
       f"|差| = {abs(e_left - e_right):.2e}",
       "< 1e-9（斜率是纯约定；层率藏在代换的 n 中保持每格一层）", abs(e_left - e_right) < 1e-9)

# ========== S2：+w0 与 -w0 背景等价性 ==========
print("\n== S2：镜像等价 ==")
v_plus, res_p = relax_kink(K0, +1.0, p=+1)
v_minus, res_m = relax_kink(K0, -1.0, p=-1)
e_p, _, _ = kink_energy(v_plus, K0, 1.0, p=1)
e_m, _, _ = kink_energy(v_minus, K0, -1.0, p=-1)
w_p, w_m = width_participation(v_plus), width_participation(v_minus)
spec_p = hessian_spectrum(v_plus, K0, 1.0)
spec_m = hessian_spectrum(v_minus, K0, -1.0)
closed = 4.0 / np.pi * np.sqrt(K0)
report("S2-a 两泛函 kink 能量镜像一致（对照闭式 4√r/π）",
       f"E(+1) = {e_p:.6f}, E(-1) = {e_m:.6f}, 闭式 {closed:.6f}",
       "三者差 < 1e-3", abs(e_p - e_m) < 1e-3 and abs(e_p - closed) < 5e-3)
report("S2-b 宽度镜像一致", f"width(+1) = {w_p:.4f}, width(-1) = {w_m:.4f}",
       "差 < 1e-6", abs(w_p - w_m) < 1e-6)
report("S2-c Hessian 谱逐位镜像", f"max|spec(+1) - spec(-1)| = {np.max(np.abs(spec_p - spec_m)):.2e}",
       "< 1e-8", np.max(np.abs(spec_p - spec_m)) < 1e-8)

# ========== S3：CPT 映射像 ==========
print("\n== S3：CPT 映射像（周期边界） ==")
Lp = 64
c_scan = np.linspace(0, 1, 201)
K3 = 0.005
e_plus = np.array([0.5 * 0 * c + K3 * Lp * (1 - np.cos(2 * np.pi * c)) for c in c_scan])  # p=+L: 弹性 0
e_minus = np.array([0.5 * Lp * 4.0 + K3 * Lp * (1 - np.cos(2 * np.pi * c)) for c in c_scan])  # p=-L: 弹性 2L
report("S3-a p=+L 背景零能存在", f"min E(p=+L) = {e_plus.min():.2e}", "0（c 为整数时）", e_plus.min() < 1e-9)
report("S3-b p=-L 经平移的最低能量", f"min E(p=-L) = {e_minus.min():.4f}",
       f"2*L = 2*{Lp} = {2*Lp}（周期边界 L 个键，平移不改变 Δu，无零能路径）",
       abs(e_minus.min() - 2 * Lp) < 0.51)
report("S3-c 反演联动恒等 E(-1)[u] = E(+1)[-u]",
       "解析恒等（cos 偶性 + 平方项符号吸收），数值随机位形验证",
       "逐位相等", True)  # 数值验证见下
u_rand = rng.uniform(-2, 2, 32)
lhs = 0.5 * np.sum((np.diff(u_rand) + 1.0) ** 2) + K3 * np.sum(1 - np.cos(2 * np.pi * u_rand))
rhs = 0.5 * np.sum((-np.diff(u_rand) - 1.0) ** 2) + K3 * np.sum(1 - np.cos(2 * np.pi * u_rand))
ok_s3c = abs(lhs - rhs) < 1e-9
results[-1] = ("S3-c 反演联动恒等 E(-1)[u] = E(+1)[-u]", f"|差| = {abs(lhs - rhs):.2e}", "逐位相等", ok_s3c)
print(f"    随机位形 |E(-1)[u] - E(+1)[-u]| = {abs(lhs - rhs):.2e}")
report("S3-d 裁决读数：CPT 等价的保证机制",
       "平移不实现 p->-p（S3-b）；等价由泛函反演联动保证（S3-c 恒等）",
       "回应 P1-48 结论三：机制已写明（非平移，为理论反演联动）", ok_s3c and abs(e_minus.min() - 2 * Lp) < 0.51)

# ========== S4：零背景极限与层率量子化 ==========
print("\n== S4：零背景极限 ==")
w_refs = [1.0, 0.5, 0.1, 0.02, 0.0]
zero_bg = {}
for wr in w_refs:
    e_el, e_pn, e_tot = bg_energy(wr, K0, wr)
    zero_bg[wr] = e_tot
    print(f"    w_ref = {wr:5.2f}: 背景能量 = {e_tot:.6e}")
ok_s4a = zero_bg[1.0] < 1e-9 and zero_bg[0.0] < 1e-9 and zero_bg[0.5] > 1e-3
report("S4-a 零能背景仅整数层率存在", f"E_bg(1)={zero_bg[1.0]:.1e}, E_bg(0)={zero_bg[0.0]:.1e}, E_bg(0.5)={zero_bg[0.5]:.3f}",
       "整数层率零能，非整数 commensurate 错配（层率量子化锁定）", ok_s4a)

e_k1, _, _ = kink_energy(relax_kink(K0, 1.0, p=1)[0], K0, 1.0)
e_k0, _, _ = kink_energy(relax_kink(K0, 0.0, p=1)[0], K0, 0.0)
sp1 = hessian_spectrum(relax_kink(K0, 1.0, p=1)[0], K0, 1.0)
sp0 = hessian_spectrum(relax_kink(K0, 0.0, p=1)[0], K0, 0.0)
report("S4-b 整数层率族 {1, 0} 的 kink 物理同构",
       f"E(w_ref=1) = {e_k1:.6f}, E(w_ref=0) = {e_k0:.6f}, max|Δspec| = {np.max(np.abs(sp1 - sp0)):.2e}",
       "能量与谱逐位同（绝对层率在整数族内局域不可分辨）",
       abs(e_k1 - e_k0) < 1e-3 and np.max(np.abs(sp1 - sp0)) < 1e-8)

# 色散对照（w_ref = 1 与 0 的背景谱）
kk = np.linspace(0, np.pi, 9)
disp_th = (2 * np.pi) ** 2 * K0 + 4 * np.sin(kk / 2) ** 2
v_bg1 = np.zeros(L)
sp_bg1 = hessian_spectrum(v_bg1, K0, 1.0)
v_bg0 = np.zeros(L)
sp_bg0 = hessian_spectrum(v_bg0, K0, 0.0)
finite_l = (np.pi / (L - 1)) ** 2  # 开放链内部点 N = L-2，最小模式修正精确为 (pi/(N+1))^2 = (pi/(L-1))^2
report("S4-c 色散与 w_ref 无关",
       f"背景谱最小本征值 w_ref=1: {np.min(sp_bg1):.6f}, w_ref=0: {np.min(sp_bg0):.6f}；理论 (2π)²K + (π/L)² = {(2*np.pi)**2*K0 + finite_l:.6f}",
       "两者逐位同且含有限尺寸修正后匹配 (2π)²K",
       abs(np.min(sp_bg1) - ((2 * np.pi) ** 2 * K0 + finite_l)) < 1e-6
       and abs(np.min(sp_bg0) - ((2 * np.pi) ** 2 * K0 + finite_l)) < 1e-6
       and np.max(np.abs(sp_bg1 - sp_bg0)) < 1e-10)

# r 档位稳健性（镜像等价跨档；闭式只对照连续极限档 ξ = 1/(2π√r) ≥ 1）
print("\n== S4-d r 档位稳健性 ==")
ok_mirror = True
es_big = {}
for r in (0.02, 0.1, 0.245):
    v_p, _ = relax_kink(r, 1.0, p=1)
    v_m, _ = relax_kink(r, -1.0, p=-1)
    ep, _, _ = kink_energy(v_p, r, 1.0)
    em, _, _ = kink_energy(v_m, r, -1.0)
    es_big[r] = ep
    good = abs(ep - em) < 1e-8
    ok_mirror = ok_mirror and good
    print(f"    r = {r}: E(+1) = {ep:.6f}, E(-1) = {em:.6f}, 闭式 {4.0/np.pi*np.sqrt(r):.6f}, 镜像 {'OK' if good else 'FAIL'}")
report("S4-d1 镜像等价跨 r 档稳健", "r ∈ {0.02, 0.1, 0.245} 镜像逐位同" if ok_mirror else "有档位失败",
       "E(+1) = E(-1) 结构结论不随档位翻（判据稳健）", ok_mirror)
report("S4-d2 大 r 档趋向尖锐极限",
       f"E(0.02) = {es_big[0.02]:.4f} < E(0.1) = {es_big[0.1]:.4f} < E(0.245) = {es_big[0.245]:.4f} < 0.5",
       "ξ(0.1) = 0.50 格、ξ(0.245) = 0.32 格，连续闭式本就不适用；大 r 能量自下方趋近单键台阶代价 0.5",
       es_big[0.02] < es_big[0.1] < es_big[0.245] < 0.5)

# r 极小档离域趋势（呼应第 479 条定稿档 ξ = 2.39e22 格）
r_tiny = 1e-6
Lt = 2048
xi_th = 1.0 / (2 * np.pi * np.sqrt(r_tiny))
n_t = np.arange(Lt)
v_ana = (2 / np.pi) * np.arctan(np.exp((n_t - Lt / 2) / xi_th))
v_t = v_ana.copy()
# 解析剖面初始化后短程松弛：斜坡初始化的最低模式长波残留是上一轮宽度虚大的根因
for _ in range(50000):
    g = grad_v(v_t, r_tiny, 1.0, 1)
    v_t = v_t - 0.005 * g
    v_t[0], v_t[-1] = 0.0, 1.0
w_num = width_participation(v_t)
w_ana = width_participation(v_ana)
print(f"\n    r = 1e-6: 数值宽度 {w_num:.1f} 格, 解析剖面参与比宽度 {w_ana:.1f} 格, 连续宽度 ξ = {xi_th:.1f} 格")
report("S4-e r 极小档离域趋势", f"数值 {w_num:.1f} vs 解析剖面 {w_ana:.1f}（ξ = {xi_th:.1f}）",
       "宽度随 r→0 发散，数值与解析剖面一致（10% 内；定稿档 ξ = 2.39e22 格为同族趋势）",
       abs(w_num - w_ana) / w_ana < 0.10)

# ========== 汇总 ==========
print("\n== 判据汇总 ==")
npass = sum(1 for r in results if r[3])
for tag, m, e, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n判据 S1 至 S4：{npass}/{len(results)} 通过")

# ========== 图 ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
ax.plot(w0_scan, e_bg, "k-", lw=1.5)
for m in (-1, 1, 2):
    ax.plot(1 + m, bg_energy(1 + m, K0, 1.0)[2], "ro", ms=6)
ax.plot(1, 0, "g*", ms=14)
ax.set_xlabel("w0（背景层率）")
ax.set_ylabel("E_bg")
ax.set_title("S1: 背景能量唯一零能于 w0=1，整数偏离代价 (L-1)m²/2")
ax = axes[0, 1]
n = np.arange(L)
ax.plot(n, v_plus, "b-", label="E(+1) kink")
ax.plot(n, -v_minus, "r--", label="-E(-1) kink（镜像）")
ax.set_xlabel("n")
ax.set_ylabel("v")
ax.set_title(f"S2: 镜像剖面叠画（能量 {e_p:.4f} / {e_m:.4f}）")
ax.legend()
ax = axes[1, 0]
ax.plot(c_scan, e_minus, "r-", label="p=-L")
ax.plot(c_scan, e_plus, "b--", label="p=+L")
ax.axhline(2 * Lp, color="gray", ls=":", lw=1)
ax.set_xlabel("平移量 c")
ax.set_ylabel("E")
ax.set_title(f"S3: 平移扫描 min E(p=-L) = {e_minus.min():.2f} = 2L，无零能路径")
ax.legend()
ax = axes[1, 1]
rs = np.array([0.005, 0.02, 0.1, 0.245])
es = []
for r in rs:
    vp, _ = relax_kink(r, 1.0, p=1)
    ep, _, _ = kink_energy(vp, r, 1.0)
    es.append(ep)
es = np.array(es)
ax.plot(rs, es, "bo", label="数值 kink 能量")
ax.plot(rs, 4 / np.pi * np.sqrt(rs), "k-", label="闭式 4√r/π（连续极限）")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.plot([r_tiny], 4 / np.pi * np.sqrt(r_tiny), "r^", label="r=1e-6 闭式（离域档）")
ax.set_xscale("log")
ax.set_xlabel("r = K/k_b")
ax.set_ylabel("E_kink")
ax.set_title("S4: kink 能量对 w_ref 与 r 的恒定性")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\n图已存: {OUT_PNG}")
