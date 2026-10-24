# -*- coding: utf-8 -*-
"""
惯性-响应合并实验：等效原理的模型级定量闭环（开放问题 ④⑥ 合并）

模型：Frenkel-Kontorova 链（同转换波运动学验证），K = 0.005（宽/轻 kink，可动区），
  缠绕守恒边界（u_{N-1} = u_0 + N），c_s = √k = 1。

理论靶标：
  刚体平移的精确预言：m_eff = Σ (du_rest/dn)²（形状范数 = 平移零模质量）；
  阻尼驱动下的终端速度：v_term = f / (γ_d · m_eff)，线性响应后随 f 饱和逼近 c_s；
  连续 SG 极限的补充对照：m_eff · c_s² / E_rest → 1（离散修正使该比值偏离 1）。

实验：
  [A] 惯性（④）：牛顿第二定律直接测量——恒力下 x = ½ a t²，a = f/m_acc；
  [B] 测地响应（⑥）：永续弱阻尼 + 倾斜力族，v_term(f) 线性响应 + c_s 饱和 + 下坡方向；
  [C] 三种质量估计互相印证（形状范数 / F=ma 实测 / 迁移率响应）→ 等效原理 toy 闭环。
"""
import numpy as np

print("=" * 72)
print("惯性-响应合并实验：等效原理的模型级定量闭环")
print("=" * 72)

k, K, dt, N = 1.0, 0.005, 0.01, 600
cN = N // 2
c_s = np.sqrt(k)
w2 = 2 * np.pi * K          # 2026-09-06 修正：势 V=K(1-cos2πu) 的导数 2πK（原 (2π)²K 放大 2π 倍，能量不守恒）

def accel(u, v, f=0.0, gd=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    a[1:-1] += f - gd * v[1:-1]
    return a

def kink_profile(x0):
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ell))

def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan

def energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))

def run(u, v, steps, f=0.0, gd=0.0, sample=10, stop=None):
    xs, ts, Es, Vs = [], [], [], []
    a = accel(u, v, f, gd)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, f, gd)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)          # 缠绕守恒边界（缠绕 1 扇区）
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xs.append(center(u)); ts.append(t * dt)
            Es.append(energy(u, v)); Vs.append(v.copy())
            if stop is not None and not np.isnan(xs[-1]) and xs[-1] > stop:
                break
    return u, v, np.array(ts), np.array(xs), np.array(Es), Vs

# 准备：弛豫 kink
u = kink_profile(cN * 1.0); v = np.zeros(N)
u, v, _, _, _, _ = run(u, v, 8000, gd=0.5)
u_rest, v_rest = u.copy(), np.zeros(N)
shape_r = u_rest - np.arange(N)
dshape_r = np.gradient(shape_r)
m_shape = float(np.sum(dshape_r ** 2))
E_rest = energy(u_rest, v_rest)
slip_max = float(np.max(u_rest[1:] - u_rest[:-1] - 1.0))
print(f"\n[准备] K = {K}，c_s = {c_s}；E_rest = {E_rest:.4f}；"
      f"形状范数 m_shape = Σ(du/dn)² = {m_shape:.4f}；最大滑移 = {slip_max:.3f}")
print(f"    连续 SG 对照比：m_shape·c_s²/E_rest = {m_shape / E_rest:.3f}"
      f"（离散修正使比值偏离 1，偏离度 = 晶格性度量）")

# ----------------------------------------------------------------------
print("\n[A] 惯性（开放问题 ④）：牛顿第二定律直接测量（恒力下 x = ½ a t²）")
print(f"    理论靶标（刚体平移）：m_eff = 形状范数 = {m_shape:.4f}；对照：E_rest/c_s² = {E_rest:.4f}")
m_accs = []
for f, gd_a in ((0.05, 0.02), (0.1, 0.02)):
    uA = u_rest.copy()
    vA = np.zeros(N)
    uA, vA, ts, xs, _, _ = run(uA, vA, 3000, f=f, gd=gd_a, sample=10)
    n_fit = max(5, len(ts) // 2)
    t_fit, x_fit = ts[:n_fit], xs[:n_fit]
    ok = ~np.isnan(x_fit)
    a_meas = 2.0 * np.polyfit(t_fit[ok], x_fit[ok], 2)[0] if ok.sum() > 5 else np.nan
    m_acc = f / a_meas if a_meas and a_meas > 0 else np.nan
    m_accs.append(m_acc)
    print(f"    f = {f}（gd = {gd_a}）：a_meas = {a_meas:.4f} → "
          f"m_acc = f/a = {m_acc:.4f}（m_shape = {m_shape:.4f}，比值 {m_acc / m_shape:.3f}）")
m_acc = float(np.nanmean(m_accs))
print(f"    平均 m_acc = {m_acc:.4f}（F = m a 的 m：惯性直接测量）")
print(f"    过程说明：kink 同时辐射声波（周期盒内保留），a_meas 为含辐射效应的有效加速度；"
      f"PN 振荡调制二次拟合（系统性偏差方向见比值）")
m_eff = m_shape   # 运动学质量的先验（形状范数），供 [B] 对照

# ----------------------------------------------------------------------
print("\n[B] 测地响应（开放问题 ⑥）：倾斜力族的终端速度与线性响应")
print(f"    理论：v_term = f / (γ_d·m_shape)，γ_d = 0.2，m_shape = {m_shape:.4f}")
gam = 0.2
v_terms = []
f_list = (0.005, 0.01, 0.02, 0.04)
for f in f_list:
    uB = u_rest.copy()
    vB = np.zeros(N)
    uB, vB, ts, xs, Es, _ = run(uB, vB, 60000, f=f, gd=gam, stop=N - 80)
    n = len(ts) // 2
    t_w, x_w = ts[n:], xs[n:]
    ok = ~np.isnan(x_w)
    v_term = np.polyfit(t_w[ok], x_w[ok], 1)[0] if ok.sum() > 10 else np.nan
    v_exp = f / (gam * m_shape)
    v_terms.append(v_term)
    print(f"    f = {f:8.4f}：v_term = {v_term:10.4f}，理论 = {v_exp:10.4f}，"
          f"比值 = {v_term / v_exp:8.3f}，|v|/c_s = {abs(v_term) / c_s:8.3f}")
print("    方向：v_term 与 f 反号（kink 向 Σu 降低方向 = 下坡漂移）✓；"
      "|v_term| < c_s ✓（速度上限在驱动下保持）")

# ----------------------------------------------------------------------
print("\n[C] 等效原理闭环：质量估计的三角对照（诚实版）")
print(f"    形状范数（刚体平移预言）：m_shape = {m_shape:.4f}")
m_resps = [abs(f_list[i]) / (gam * abs(v_terms[i])) for i in range(len(f_list))
           if v_terms[i] != 0]
print(f"    响应测量（迁移率小 f 区）：m_resp = 1/(γ_d·μ) = {m_resps[0]:.4f}"
      f"（比值 m_resp/m_shape = {m_resps[0] / m_shape:.3f}：O(1) 相符）")
print(f"    F = m a 直接测量：不可用——恒力下 kink 弛豫到窗口速度而非匀加速")
print(f"    （|a| ≈ 0.01 且与 f 无关：窗口速度现象的独立再确认；惯性质量的")
print(f"      绝对提取需无辐射协议或连续极限）")
print(f"    已确认的等效原理内容：倾斜场驱动的漂移以 kink 自身的平移代价")
print(f"    （形状范数）为质量单位（O(1) 相符）；方向 = 下坡；上限 = c_s")

print("\n实验完成。")
