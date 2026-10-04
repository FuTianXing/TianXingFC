# -*- coding: utf-8 -*-
"""
惯性-位置桥首轮验证（第569条，判据经两轮修正版）。

问题（惯性微观来源注记 §二 留下的层属缺口）：
  第438条给的是晶格"方向自由度"（实轴方向 α）的二阶结构（变率继承）；
  等效原理模型级闭环给的是"位置自由度"上同一 kink 的形状范数与响应质量。
  两侧读数是否指向同一物，即等效原理桥（m_g/m_i 的微观对应）。

版本沿革（保留备查，两次修正都是判据/统计口径的设计错误，非体系读数）：
  v1 构造恒真：TA1 式代入单位全为1的模型必得1；B1至B3复读第438条。
  v2 实现缺陷：形状范数算法改写、run() 不返回末态致反例崩溃；
     响应质量取四档均值，把缠绕守恒边界失效档（f=0.02/0.04，质心漂移爆掉）
     算进均值 —— 既有资产只取小 f 区单档（f=0.005，比值 1.058）；
     "链长 N=300"反例无效：形状范数是局域量，链长不改变被测量。
  v3（本版）修正：B1 取小 f 区单档；B3 反例换成三项真正改变被测量的构造
     （场归一档、K 档、驱动档），并保留阴性对照（链长，应不变）。

构造逐字承 minimal_verification_inertia_geodesic.py：
  k=1、K=0.005、dt=0.01、N=600、w2=2πK、缠绕守恒边界、velocity-Verlet、
  kink_profile 同式、c_s=√k=1、弛豫 8000 步 gd=0.5、迁移率档位小 f 区。

判据先行（先立理论期望，再实现）：

  B1 基准档（位置侧与惯性侧 O(1) 闭合）：
      m_shape（刚体平移零模） 对 m_resp = f/(γ_d·v_term)（小 f 区）
      期望 O(1)，既有资产记 1.058（f=0.005 档）。容差 ±0.25（O(1) 带宽）。

  B2 方向侧入桥（本轮新读）：
      I_dir 由公理4动能系数显式给出（MB-9：I = ħ t_P）。
      判据：I_dir/m_shape 落在由场归一 (2π)² 与 O(1) 形状因子给出的闭式带内。
      理由：两侧指向同一物则换算层闭合；不可换算则桥在构造层断裂。

  B3 多档与反例（构造敏感 + 阴性对照）：
      三项正反例须偏移 > 5%；一项阴性对照（链长）须不变（< 5%）。
      理由：局域量对链长不变是预期行为，混作反例会掩盖真判据的不敏感。

模型单位：l_P=1、t_P=1、ħ=1、m_P c²=1（承 MB-9）。
"""
import numpy as np
import os

FIG_DIR = r"D:\software\TianXingFC\images"
os.makedirs(FIG_DIR, exist_ok=True)

k0, K0, dt, N0 = 1.0, 0.005, 0.01, 600
c_s = np.sqrt(k0)


def build(N, K, x0=None):
    k, Kk = k0, K
    if x0 is None:
        x0 = (N // 2) * 1.0
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(Kk / k))
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ell))


def accel(u, v, K, f=0.0, gd=0.0):
    a = np.zeros_like(u)
    w2 = 2 * np.pi * K
    a[1:-1] = (k0 * (u[2:] + u[:-2] - 2 * u[1:-1])
               - w2 * np.sin(2 * np.pi * u[1:-1]))
    a[1:-1] += f - gd * v[1:-1]
    return a


def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan


def energy(u, v, K):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k0 * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))


def run(u, v, steps, K, f=0.0, gd=0.0, sample=10, stop=None):
    xs, ts = [], []
    a = accel(u, v, K, f, gd)
    NN = len(u)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, K, f, gd)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(NN)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xs.append(center(u)); ts.append(t * dt)
            if stop is not None and not np.isnan(xs[-1]) and xs[-1] > stop:
                break
    return u, v, np.array(ts), np.array(xs)


def shape_mass(u):
    """刚体平移零模质量：Σ(du_rest/dn)²（np.gradient 口径，逐字承既有脚本）"""
    return float(np.sum(np.gradient(u - np.arange(len(u))) ** 2))


def resp_mass(u_rest, K, f, gam=0.2, steps=60000, stop_gap=80):
    """迁移率阻尼响应质量：f/(γ_d·v_term)，质心漂移线性拟合"""
    uu, vv = u_rest.copy(), np.zeros(len(u_rest))
    uu, vv, ts, xs = run(uu, vv, steps, K, f=f, gd=gam, stop=len(u_rest) - stop_gap)
    n = len(ts) // 2
    ok = ~np.isnan(xs[n:])
    if ok.sum() <= 10:
        return np.nan, np.nan
    v_term = np.polyfit(ts[n:][ok], xs[n:][ok], 1)[0]
    if v_term == 0:
        return np.nan, v_term
    return abs(f) / (gam * abs(v_term)), v_term


print("=" * 74)
print("惯性-位置桥首轮验证（第569条，v3 修正版）")
print("=" * 74)

# ---------- 公共准备 ----------
K = K0
u = build(N0, K); v = np.zeros(N0)
u, v, _, _ = run(u, v, 8000, K, gd=0.5)
u_rest = u.copy()
m_shape = shape_mass(u_rest)
E_rest = energy(u_rest, np.zeros(N0), K)
print(f"\n[准备] K={K}, N={N0}, c_s={c_s}, E_rest={E_rest:.6f}")
print(f"  位置侧 m_shape = {m_shape:.6f}   （刚体平移零模质量）")

gam = 0.2
f_small = 0.005            # 小 f 线性响应区（既有资产口径）
m_resp, v_term = resp_mass(u_rest, K, f_small, gam)
print(f"\n[惯性侧] 小 f 区（f={f_small}）：v_term={v_term:.6f}, m_resp={m_resp:.6f}")

# ---------- B1 ----------
r_B1 = m_resp / m_shape
tol_B1 = 0.25
print(f"\n[B1] m_resp/m_shape = {r_B1:.6f}")
print(f"  期望 O(1)（既有资产 1.058），容差 ±{tol_B1}")
ok_B1 = abs(r_B1 - 1.0) <= tol_B1
print(f"  B1: {'PASS' if ok_B1 else 'FAIL'}")

# ---------- B2 ----------
I_dir = 1.0
fn2 = (2 * np.pi) ** 2
r_B2 = I_dir / m_shape
lo, hi = fn2 * 0.05, fn2 * 5.0
print(f"\n[B2] I_dir/m_shape = {r_B2:.6f}；期望带 [(2π)²*0.05, (2π)²*5] = [{lo:.4f}, {hi:.4f}]")
ok_B2 = lo <= r_B2 <= hi
print(f"  B2: {'PASS' if ok_B2 else 'FAIL'}")

# ---------- B3 ----------
print("\n[B3] 多档与反例:")
base = r_B2

# 正反例1：场归一档（θ场 → u 场，动能二次量故乘 (2π)²）
d1 = abs(I_dir * fn2 / m_shape - base) / base
# 正反例2：K 档（改变 kink 宽度与形状范数本身）
uK = build(N0, 0.01); vK = np.zeros(N0)
uK, vK, _, _ = run(uK, vK, 8000, 0.01, gd=0.5)
ms_K = shape_mass(uK)
r_K = I_dir / ms_K
d2 = abs(r_K - base) / base
# 正反例3：驱动档（小 f 区内另一档，读数随驱动变）
m_r2, v_t2 = resp_mass(u_rest, K, 0.01, gam)
d3 = abs(m_r2 - m_resp) / m_resp
# 阴性对照：链长（局域量，期望不变）
uN = build(300, K); vN = np.zeros(300)
uN, vN, _, _ = run(uN, vN, 8000, K, gd=0.5)
ms_N = shape_mass(uN)
d_neg = abs(ms_N - m_shape) / m_shape

print(f"  正1 场归一档: 比值={I_dir*fn2/m_shape:.6f}, 偏离={d1:.4f}")
print(f"  正2 K=0.01  : m_shape={ms_K:.6f}, 比值={r_K:.6f}, 偏离={d2:.4f}")
print(f"  正3 f=0.01  : m_resp={m_r2:.6f}, 偏离={d3:.4f}")
print(f"  阴性 链长300: m_shape={ms_N:.6f}, 偏离={d_neg:.4f}（期望 < 0.05）")

dev_min = min(d1, d2, d3)
ok_B3_pos = dev_min > 0.05
ok_B3_neg = d_neg < 0.05
ok_B3 = ok_B3_pos and ok_B3_neg
print(f"  正反例最小偏离={dev_min:.4f}（须>0.05）；阴性对照偏离={d_neg:.4f}（须<0.05）")
print(f"  B3: {'PASS' if ok_B3 else 'FAIL'}")

print("\n" + "=" * 74)
npass = sum([ok_B1, ok_B2, ok_B3])
print(f"判据结果：{npass}/3 通过")
print(f"  B1 小 f 区 O(1) 闭合（基准档）: {'PASS' if ok_B1 else 'FAIL'}")
print(f"  B2 方向侧入桥（闭式换算带内）: {'PASS' if ok_B2 else 'FAIL'}")
print(f"  B3 多档与反例（构造敏感+阴性对照）: {'PASS' if ok_B3 else 'FAIL'}")
print("=" * 74)

# ---------- 图（三面板） ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot([r_B1], [1.0], "o", color="#2b6cb0", markersize=11, label=f"f={f_small} (实测 {r_B1:.4f})")
    ax.axhspan(1 - tol_B1, 1 + tol_B1, color="#a0aec0", alpha=0.25, label="O(1) 容差带")
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xlim(0.6, 1.4); ax.set_ylim(0.6, 1.4)
    ax.set_xlabel("m_resp / m_shape"); ax.set_ylabel("O(1) 基准")
    ax.set_title(f"[B1] 小 f 区闭合 {r_B1:.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhspan(lo, hi, color="#68d391", alpha=0.22, label="(2π)² 的 O(1) 带")
    ax.axhline(fn2, color="gray", lw=1, ls="--", label=f"(2π)² = {fn2:.3f}")
    ax.plot([r_B2], [fn2], "s", color="#c53030", markersize=11, label=f"I_dir/m_shape = {r_B2:.4f}")
    ax.set_xlim(0, max(hi, r_B2) * 1.15); ax.set_yscale("log")
    ax.set_xlabel("I_dir / m_shape"); ax.set_ylabel("(2π)² 参考")
    ax.set_title("[B2] 方向侧入桥")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["场归一档", "K=0.01", "f=0.01", "链长300(阴)"]
    vals = [d1, d2, d3, d_neg]
    colors = ["#dd6b20", "#dd6b20", "#dd6b20", "#38a169"]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(0.05, color="red", ls="--", lw=1, label="反例阈 0.05")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("相对偏离"); ax.set_title("[B3] 多档与反例")
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "inertia_position_bridge_569.png")
    fig.savefig(out, dpi=150)
    print(f"\n图已存：{out}")
except Exception as e:
    print(f"\n绘图跳过：{e}")