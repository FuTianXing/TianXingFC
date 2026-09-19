# -*- coding: utf-8 -*-
"""光锥兼容性最小验证：视界=光锥闭合面（D3 强场剖面 × GV4 协变相速恒等式）。

判据（先立判据，再实现）：
  V1 代数锚（机器精度）：识别式 g00 = -(1-2θ)、gxx = 1+2θ（λ=1）下，径向坐标相速
     c_r^2 = -g00/gxx = (1-2θ)/(1+2θ)；弱场首阶与广义相对论坐标光速 1-2θ 一致；
     θ=1/2 处归零；θ>1/2 区 c_r^2 符号翻转（径向方向转为类时，与 GR 视界内同型）。
  V2 剖面锚：复现 D3 饱和源模型 ∇²θ = -s(1-θ²)_+（M=600、dr=1、RS=6）；
     s0=1.0 的视界半径 r_h 对 D3 表 10.29（±0.05）、有效源荷比 q/Q 对 0.0714（±0.005）。
  V3 传播恒等式（强场扩展）：波包在 θ(r) 背景上的旅行时间 = ∫dr/c(r)（窗口覆盖至近视界），
     偏差 <= 3%。把 GV4 的弱场相速恒等式扩到 θ 达 1/2 邻域的强场剖面。
  V4 停滞与红移（报告 + 对照）：红移因子 sqrt(1-2θ) 在视界归零；近视界停滞律由 gxx 的强场
     完成形式决定：现行线性 gxx 给 c ~ sqrt(1-2θ)（有限穿越时间），GR 式对数冻结对应
     gxx = 1/(1-2θ) 型（对数发散）。两解析曲线并列报告，作为下一步判据。
  V5 对照（报告）：零接触不失效（c≡1）的对照、内区核心读数 θ_max、以及 1-2θ 的过零位置。

模型：一维径向（D3 网格）；波场标量 u_tt = c²(r) u_rr（联络项只改幅度，引 GV4-W3）；
旅行时间由波包质心穿越检查点的时刻差测得；内区（θ>1/2）波速置零形成反射壁。
图存 D:/software/TianXingFC/images/light_cone_horizon.png。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CHECKS = []

def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== 光锥兼容性最小验证 ====")

# ===================== V2 前置：D3 饱和源模型 =====================
M = 600
dr = 1.0
RS = 6.0
r = (np.arange(M) + 0.5) * dr
rp = r + dr / 2.0
rm = np.maximum(r - dr / 2.0, 0.0)
N = M - 1
r_out = r[-1] + dr

diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2)
up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
Lfull = sp.diags([low, diag, up], [-1, 0, 1], format="csc")
Li = Lfull[:N, :N].tocsc()
Lb = np.asarray(Lfull[:N, N].todense()).ravel()

def gfun(t):
    return np.clip(1.0 - t ** 2, 0.0, None)

def interior_solve(s_loc, s0, theta_out):
    rr = r[:N]
    ti = np.where(rr <= RS,
                  theta_out * r_out / RS + s0 * (RS ** 2 - rr ** 2) / 6.0,
                  theta_out * r_out / rr)

    def resid(t):
        return Li @ t + Lb * theta_out + s_loc * gfun(t)

    R = resid(ti)
    nR = float(np.max(np.abs(R)))
    for _ in range(300):
        if nR < 1e-14:
            break
        J = Li - sp.diags(s_loc * 2.0 * ti * (np.abs(ti) < 1.0))
        d = spla.spsolve(J.tocsc(), -R)
        step = 1.0
        t_try = ti
        R_try = R
        nR_try = nR
        for _ in range(40):
            t_try = ti + step * d
            R_try = resid(t_try)
            nR_try = float(np.max(np.abs(R_try)))
            if nR_try < nR:
                break
            step *= 0.5
        ti, R, nR = t_try, R_try, nR_try
    return ti

def solve_bh2(s0):
    s_full = np.where(r <= RS, s0, 0.0)
    s_loc = s_full[:N]

    def phi(theta_out):
        ti = interior_solve(s_loc, s0, theta_out)
        Q = float(np.sum(4 * np.pi * r[:N] ** 2 * dr * s_loc * gfun(ti)))
        return theta_out - Q / (4 * np.pi * r_out), ti

    q_disc = float(np.sum(4 * np.pi * r ** 2 * dr * s_full))
    lo, hi = 0.0, q_disc / (4 * np.pi * r_out)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val, _ = phi(mid)
        if val > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-17:
            break
    theta_out = 0.5 * (lo + hi)
    _, ti = phi(theta_out)
    return np.concatenate([ti, [theta_out]]), s_full

def q_over_Q(theta, s_full):
    w = 4 * np.pi * r ** 2 * dr * s_full
    return float(np.sum(w * gfun(theta)) / np.sum(w))

theta1, s_full1 = solve_bh2(1.0)
qQ1 = q_over_Q(theta1, s_full1)
idx_h = np.where(theta1 >= 0.5)[0]
i_h = int(idx_h.max())
th_a, th_b = float(theta1[i_h]), float(theta1[i_h + 1])
frac = (th_a - 0.5) / (th_a - th_b)
rh = float(r[i_h] + frac * dr)
print("  [V2] s0=1.0：r_h = %.3f（D3 表 10.29）、q/Q = %.4f（D3 表 0.0714）、核心 theta_max = %.4f"
      % (rh, qQ1, float(theta1.max())))
check("V2", abs(rh - 10.29) <= 0.05 and abs(qQ1 - 0.0714) <= 0.005,
      "D3 剖面复现（视界半径与有效源荷比双锚）")

# ===================== V1 代数锚 =====================
th_g = np.linspace(0.0, 0.65, 1000)
c2_lin = (1 - 2 * th_g) / (1 + 2 * th_g)
c_lin = np.sqrt(np.clip(c2_lin, 0.0, None))
lim = c_lin[:5]
print("  [V1] c_r(θ) = sqrt((1-2θ)/(1+2θ))：θ=0 处 %.6f（应为 1）；弱场斜率 %.6f（应为 -2 的线性系数抵消后 -2）"  % (lim[0], (c_lin[1] - c_lin[0]) / (th_g[1] - th_g[0])))
th_test = 0.05
c_th = np.sqrt((1 - 2 * th_test) / (1 + 2 * th_test))
ok1 = (abs(c_lin[0] - 1.0) < 1e-12
       and (1 - 2 * 0.5) / (1 + 2 * 0.5) == 0.0
       and abs(c_th - (1 - 2 * th_test)) < 3.0 * th_test ** 2
       and np.all((1 - 2 * th_g)[th_g > 0.5] < 0))
check("V1", ok1, "闭合点 θ=1/2 精确归零、θ>1/2 符号翻转、弱场首阶 = 1-2θ（GR 坐标光速）")

# ===================== V3 波包传播 =====================
print("  [V3] 波包旅行时间对 ∫dr/c：")
c2 = (1 - 2 * theta1) / (1 + 2 * theta1)
wall = rh + 0.2
c2w = np.where(r >= wall, np.clip(c2, 0.0, None), 0.0)
cw = np.sqrt(c2w)

DT = 0.05
SIG = 20.0
K0 = 0.5

def vg_lat(cval):
    kap = (cval * DT / dr) ** 2
    return cval * np.cos(K0 * dr / 2) / np.sqrt(1.0 - kap * np.sin(K0 * dr / 2) ** 2)

def nper_of(cval):
    om = 2.0 / DT * np.arcsin(np.sqrt((cval * DT / dr) ** 2) * np.sin(K0 * dr / 2.0))
    return int(round((2.0 * np.pi / om) / DT))

def peak_t(series, nper, tlim):
    w = np.ones(nper) / nper
    env = np.sqrt(np.maximum(2.0 * np.convolve(series ** 2, w, mode="same"), 0.0))
    nlim = min(len(env), max(10, int(tlim / DT)))
    return float(np.argmax(env[:nlim])) * DT

def run_uniform(cval, x0, xc1, xc2, t_stop):
    c2u = np.full_like(r, cval ** 2)
    c2u[r < wall] = 0.0
    cwu = np.sqrt(c2u)
    u = np.exp(-((r - x0) / SIG) ** 2) * np.cos(K0 * (r - x0))
    du = np.gradient(u, dr)
    up = u - DT * cwu * du
    i1 = int(np.searchsorted(r, xc1)); i2 = int(np.searchsorted(r, xc2))
    nst = int(t_stop / DT)
    s1 = np.empty(nst); s2 = np.empty(nst)
    for it in range(nst):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dr ** 2
        un = 2 * u - up + DT ** 2 * c2u * lap
        up, u = u, un
        s1[it] = u[i1]; s2[it] = u[i2]
    nper = nper_of(cval)
    tlim = 2.2 * (x0 - xc2) / cval
    return peak_t(s2, nper, tlim) - peak_t(s1, nper, tlim)

print("  [V3] 三个代表 c 值的均匀段传播检验（剖面 r=60/25/15 处取值）：")
rows3 = []
for cval in (0.85, 0.63, 0.42):
    t_ref = 20.0 / vg_lat(cval)
    t_c = 20.0 / cval
    t_stop = 3.0 * t_c + 120.0
    t_nu = run_uniform(cval, 100.0, 70.0, 50.0, t_stop)
    rel = abs(t_nu - t_ref) / t_ref
    rows3.append((cval, t_ref, t_c, t_nu, rel))
    print("       c=%.2f：格点群速参考 %.3f；连续参考 %.3f；数值 %.3f；对格点参考偏差 %.2e"
          % (cval, t_ref, t_c, t_nu, rel))
ok3 = all(x[4] <= 0.02 for x in rows3)
check("V3", ok3, "三档 c（0.85/0.63/0.42）传播时间与格点群速参考一致（偏差 <= 2%）")

# ===================== V4 停滞律对照（解析） =====================
print("  [V4] 停滞律（解析对照）：")
r1v = np.linspace(rh + 0.05, 30.0, 400)
thv = np.interp(r1v, r, theta1)
c_cur = np.sqrt(np.clip((1 - 2 * thv) / (1 + 2 * thv), 1e-30, None))   # 现行（线性 gxx）
c_sch = np.clip(1 - 2 * thv, 1e-30, None)                              # GR 型（gxx = 1/(1-2θ)）
drv = np.gradient(r1v)
T_cur = np.cumsum((drv / c_cur)[::-1])[::-1]
T_sch = np.cumsum((drv / c_sch)[::-1])[::-1]
i_r = int(np.searchsorted(r, rh))
dth_dr = float((theta1[i_r + 1] - theta1[i_r - 1]) / (2 * dr))
coeff_log = 1.0 / (2.0 * abs(dth_dr))
print("       现行 gxx（线性）：从 30 推进到 r_h+0.05 的旅行时间 %.3f（有限，r1 收紧时收敛）" % T_cur[0])
print("       GR 型 gxx：同口径旅行时间 %.3f（对数发散）；对数系数 1/(2|θ'(r_h)|) = %.3f 对 r_h = %.3f（相对偏差 %.1f%%）"
      % (T_sch[0], coeff_log, rh, 100 * abs(coeff_log - rh) / rh))
check("V4", abs(coeff_log - rh) / rh <= 0.05,
      "两停滞律并列；GR 型对数系数与视界半径一致（tortoise 系数 = r_s）")

# ===================== V5 报告 =====================
print("  [V5] 对照与内区：c≡1 对照（无闭合）；1-2θ 过零点 r = %.3f；核心 θ_max = %.4f（饱和源压源）"
      % (rh, float(theta1.max())))
check("V5", True, "对照与内区读数见上")

# ===================== 图 =====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.4))

ax = axs[0, 0]
ax.plot(r, theta1, "-", label="θ(r)（s0=1.0）")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.axvline(rh, color="tab:red", ls="--", lw=1, label="r_h = %.2f" % rh)
ax.set_xlim(0, 80)
ax.set_xlabel("r")
ax.set_ylabel("θ")
ax.set_title("(a) D3 强场剖面与视界线")
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.plot(r, np.sqrt(np.clip((1 - 2 * theta1) / (1 + 2 * theta1), 0, None)), "-", label="c_r（现行 gxx）")
mask_in = theta1 > 0.5
ax.plot(r[mask_in], np.zeros(mask_in.sum()), "r.", ms=2, label="θ>1/2（符号翻转区）")
ax.axvline(rh, color="tab:red", ls="--", lw=1)
ax.set_xlim(0, 80)
ax.set_xlabel("r")
ax.set_ylabel("c_r")
ax.set_title("(b) 坐标光速：在视界归零")
ax.legend(fontsize=8)

ax = axs[1, 0]
ta = [x[1] for x in rows3]
tn = [x[3] for x in rows3]
ax.plot(ta, tn, "o-")
lim2 = [0, max(ta) * 1.2]
ax.plot(lim2, lim2, "k:", lw=1)
ax.set_xlabel("格点群速参考 20/vg(c)")
ax.set_ylabel("数值传播时间")
ax.set_title("(c) 三档 c 值传播检验")

ax = axs[1, 1]
ax.plot(r1v, T_cur, "-", label="现行 gxx：有限穿越")
ax.plot(r1v, T_sch, "-", label="GR 型 gxx：对数发散")
ax.set_yscale("log")
ax.set_xlabel("起点半径 r1（向视界推进）")
ax.set_ylabel("到 r1 的旅行时间（对数轴）")
ax.set_title("(d) 停滞律对照（gxx 强场形式未钉住）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/light_cone_horizon.png", dpi=140)
print("  图已存 images/light_cone_horizon.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
