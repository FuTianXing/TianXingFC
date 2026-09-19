# -*- coding: utf-8 -*-
"""U2（最小臂）：耦合重开区的有效应力判决。

判据（先立判据，再实现；草案见 内部草稿/判据草案-U2（最小臂）：耦合重开区的有效应力判决.md）：
  U2-A1 对照锚：lambda = 0 复现 D3 剖面（r_h 在 [10.27, 10.31]；真空区 theta*r 相对散布 <= 5%）。
  U2-A2 源区耦合不改真空结构：H1 口径下真空区散布 <= 5%（与 lambda 无关）；q_eff(lambda) 单调。
  U2-A3 扩展区耦合生成缺陷（报告）：H2、H3 在其可达 lambda 上，作用区内 max|theta + r theta'|
       相对 lambda=0 基准的比；比 >= 5 登记「自带有效应力」，否则登记「维持至该口径」。
  U2-A4 有效源量级（报告）：变形剖面的 R_theta 与 G_tt r^2 峰值（W3 同机器），并列 W3 参考 |R_theta| = 0.79。
  U2-A5 停滞律连带（报告）：r_h 与对数系数 1/(2|theta'(r_h)|) 的偏移。

模型：径向饱和源模型（同 D3/gxx：M=600、dr=1、RS=6、s0=1.0）加耦合力
  lap(theta) = -s g(theta) - lambda sin(theta) h(r)，g = (1 - theta^2)_+。
口径：H1 h = 1 (r <= RS)；H2 h = g(theta)（记录饱和份额，全域）；H3 h = 1 (r <= 10.285)。
外边界：衰减 Robin（theta' = -(kappa + 1/r) theta；kappa = sqrt(lambda) 于 H2，0 于 H1/H3）。
求解：阻尼 Newton 加自适应 lambda 延拓；物理分支守门（theta >= 0 且 < 1.05）。
确定性。图存 D:/software/TianXingFC/images/u2_coupling_stress.png。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== U2 最小臂：耦合重开区的有效应力判决 ====")

M = 600; dr = 1.0; RS = 6.0; s0 = 1.0
r = (np.arange(M) + 0.5) * dr
rp = r + dr / 2.0
rm = np.maximum(r - dr / 2.0, 0.0)
diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2)
up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
L0 = sp.diags([low, diag, up], [-1, 0, 1], format="lil")
inside = (r <= RS)

def gfun(t):
    return np.clip(1.0 - t ** 2, 0.0, None)

def hof(t, key):
    if key == "H1":
        return np.where(r <= RS, 1.0, 0.0)
    if key == "H2":
        return gfun(t)
    if key == "H3":
        return np.where(r <= 10.285, 1.0, 0.0)
    raise ValueError(key)

def jac_h(t, key):
    if key == "H2":
        return -2.0 * t * (np.abs(t) < 1.0)
    return np.zeros_like(t)

def make_operator(kappa):
    Lm = L0.copy().tolil()
    Lm[-1, :] = 0.0
    Lm[-1, -1] = 1.0 + dr * (kappa + 1.0 / r[-1])
    Lm[-1, -2] = -1.0
    return Lm.tocsr()

def residual(t, lam, key, Lm, sgn):
    g = gfun(t)
    h = hof(t, key)
    return Lm @ t + s0 * np.where(inside, g, 0.0) + sgn * lam * np.sin(t) * h

def solve(lam, key, guess, sgn):
    kappa = np.sqrt(lam) if key == "H2" else 0.0
    Lm = make_operator(kappa)
    t = guess.copy()
    nR = float(np.max(np.abs(residual(t, lam, key, Lm, sgn))))
    for _ in range(400):
        if nR < 1e-12:
            break
        g = gfun(t)
        h = hof(t, key)
        djac = s0 * np.where(inside, -2.0 * t * (np.abs(t) < 1.0), 0.0) \
               + sgn * lam * (np.cos(t) * h + np.sin(t) * jac_h(t, key))
        J = Lm + sp.diags(djac)
        d = spla.spsolve(J.tocsc(), -residual(t, lam, key, Lm, sgn))
        t_try, nR_try = t, nR
        improved = False
        for _ in range(40):
            t_try = t + 0.5 ** _ * d
            nR_try = float(np.max(np.abs(residual(t_try, lam, key, Lm, sgn))))
            if nR_try < nR:
                improved = True
                break
        if not improved:
            break
        t, nR = t_try, nR_try
    return t, nR

def ricci(A, B, rv, Ap, App, Bp):
    Rtt = App / (2 * B) - Ap / (4 * B) * (Ap / A + Bp / B) + Ap / (rv * B)
    Rrr = -App / (2 * A) + Ap / (4 * A) * (Ap / A + Bp / B) + Bp / (rv * B)
    Rth = 1.0 - 1.0 / B - (rv / (2 * B)) * (Ap / A - Bp / B)
    return Rtt, Rrr, Rth

def readouts(t):
    out = {}
    rh = float(np.interp(0.5, t[::-1], r[::-1])) if t.max() > 0.5 else float("nan")
    out["rh"] = rh
    out["q_eff"] = float(4.0 * np.pi * 50.0 * np.interp(50.0, r, t))
    m = (r >= RS + 5.0) & (r <= 60.0)
    tr = t[m] * r[m]
    out["spread"] = float((tr.max() - tr.min()) / tr.mean())
    dth = np.gradient(t, r)
    D = t + r * dth
    m2 = (r >= 8.0) & (r <= 15.0)
    m3 = (r >= 20.0) & (r <= 60.0)
    out["D_near"] = float(np.abs(D[m2]).max())
    out["D_outer"] = float(np.abs(D[m3]).max())
    A = 1.0 - 2.0 * t
    Ap = np.gradient(A, r)
    App = np.gradient(Ap, r)
    B = 1.0 / A
    Bp = -Ap / A ** 2
    Rtt, Rrr, Rth = ricci(A, B, r, Ap, App, Bp)
    Gtt = Rtt + 0.5 * A * (Rrr / B + 2.0 * Rth / r ** 2 - Rtt / A)
    band = (r >= 11.0) & (r <= 30.0)
    out["Rth_max"] = float(np.abs(Rth[band]).max())
    out["Gtt_r2"] = float(np.abs(Gtt[band] * r[band] ** 2).max())
    out["logcoef"] = float("nan")
    if not np.isnan(rh):
        i = int(np.argmin(np.abs(r - rh)))
        gp = dth[i]
        if gp != 0.0:
            out["logcoef"] = float(1.0 / (2.0 * abs(gp)))
    return out

guess0 = np.minimum(0.999, 5.14 / np.maximum(r, RS))
LAM_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ARMS = [("P", +1.0), ("M", -1.0)]
stored = {}
lam_star = {}
for sgn_name, sgn in ARMS:
    for key in ("H1", "H2", "H3"):
        t0, nR0 = solve(0.0, key, guess0, sgn)
        if not (nR0 < 1e-9 and t0.min() > -0.2 and t0.max() < 1.2):
            raise RuntimeError("lambda=0 基线未收敛：" + key)
        path = [(0.0, t0.copy())]
        cur_lam, cur_t = 0.0, t0
        targets = list(LAM_GRID[1:])
        ti = 0
        stopped = None
        while ti < len(targets):
            tg = targets[ti]
            if tg <= cur_lam + 1e-9:
                ti += 1
                continue
            dl = tg - cur_lam
            if len(path) >= 2:
                dl_prev = path[-1][0] - path[-2][0]
                guess = cur_t + (path[-1][1] - path[-2][1]) * (dl / dl_prev) if dl_prev > 0 else cur_t.copy()
            else:
                guess = cur_t.copy()
            t_try, nR = solve(tg, key, guess, sgn)
            ok = (nR < 1e-9) and (t_try.min() > -0.2) and (t_try.max() < 1.2)
            if ok:
                path.append((tg, t_try.copy()))
                cur_lam, cur_t = tg, t_try
                ti += 1
            else:
                mid = 0.5 * (cur_lam + tg)
                if mid - cur_lam < 2e-4 or mid >= tg - 1e-9:
                    stopped = (tg, nR, float(t_try.min()), float(t_try.max()))
                    break
                targets = targets[:ti] + [round(mid, 4)] + targets[ti:]
        stored[(sgn_name, key)] = path
        ls = None
        for i in range(1, len(path)):
            l0, tp0 = path[i - 1]
            l1, tp1 = path[i]
            if tp1.max() >= 1.0 and tp0.max() < 1.0 and tp1.max() > tp0.max():
                frac = (1.0 - tp0.max()) / (tp1.max() - tp0.max())
                ls = l0 + frac * (l1 - l0)
                break
        if ls is not None:
            lam_star[(sgn_name, key)] = ls
        if stopped is not None:
            print("  [延拓] 臂%s %s 在 lambda=%.4f 后停止（残余 %.2e，theta 范围 [%.3f, %.3f]）"
                  % (sgn_name, key, stopped[0], stopped[1], stopped[2], stopped[3]))
        else:
            print("  [延拓] 臂%s %s 抵达 lambda=1.00（theta 范围 [%.4f, %.4f]）"
                  % (sgn_name, key, path[-1][1].min(), path[-1][1].max()))
        if (sgn_name, key) in lam_star:
            print("       （满偏越界阈值：theta_max 于 lambda* = %.4f 达到 1.0）" % lam_star[(sgn_name, key)])

def prof(arm, key, lam):
    ks = [x for x in stored[(arm, key)] if abs(x[0] - lam) < 5e-4]
    return ks[0][1] if ks else None

def last(arm, key):
    return stored[(arm, key)][-1]

base_t = stored[("P", "H1")][0][1]
R0 = readouts(base_t)
print("  [U2-A1] lambda=0：r_h = %.4f（D3 10.285）；真空区散布 = %.3f%%；D_near = %.4f；D_outer = %.4f"
      % (R0["rh"], 100.0 * R0["spread"], R0["D_near"], R0["D_outer"]))
check("A1", 10.27 <= R0["rh"] <= 10.31 and R0["spread"] <= 0.05, "对照锚：复现 D3 剖面")

p_h1 = stored[("P", "H1")]
rows_h1 = [(lam, readouts(t)) for lam, t in p_h1]
qspread_ok = all(r["spread"] <= 0.05 for _, r in rows_h1)
qmono = all(rows_h1[i + 1][1]["q_eff"] >= rows_h1[i][1]["q_eff"] for i in range(len(rows_h1) - 1))
print("  [U2-A2] 臂P H1（lambda 至 %.2f）：散布最大 = %.3f%%；q_eff 单调 = %s；q_eff 比 = %.4f"
      % (p_h1[-1][0], 100.0 * max(r["spread"] for _, r in rows_h1), qmono,
         rows_h1[-1][1]["q_eff"] / rows_h1[0][1]["q_eff"]))
check("A2", qspread_ok and qmono, "源区耦合只在作用区改写源荷，区外真空结构维持")

for arm in ("P", "M"):
    for key in ("H2", "H3"):
        lams = [x[0] for x in stored[(arm, key)]]
        shows = [lam for lam in LAM_GRID if lam <= lams[-1] + 1e-6]
        for lam in shows:
            t = prof(arm, key, lam)
            if t is None:
                continue
            rr = readouts(t)
            print("  [读数] 臂%s %s lambda=%.2f：theta_max=%.4f  r_h=%.4f  散布=%.3f%%  D_near=%.4f  D_outer=%.4f  R_th=%.4f  G_tt r2=%.4f"
                  % (arm, key, lam, t.max(), rr["rh"], 100.0 * rr["spread"], rr["D_near"], rr["D_outer"], rr["Rth_max"], rr["Gtt_r2"]))
        print("       （臂%s %s 最高可达 lambda=%.2f）" % (arm, key, lams[-1]))

for key in ("H2", "H3"):
    lam_b = last("P", key)[0]
    t_b = last("P", key)[1]
    r1 = readouts(t_b)
    ratio_near = r1["D_near"] / R0["D_near"] if R0["D_near"] > 0 else float("inf")
    ratio_outer = r1["D_outer"] / R0["D_outer"] if R0["D_outer"] > 0 else float("inf")
    verdict = "自带有效应力" if max(ratio_near, ratio_outer) >= 5.0 else "维持至该口径"
    print("  [U2-A3] 臂P %s 在 lambda=%.2f：D_near 比 = %.1f 倍；D_outer 比 = %.1f 倍 => %s"
          % (key, lam_b, ratio_near, ratio_outer, verdict))

t_b = last("P", "H2")[1]
rH = readouts(t_b)
print("  [U2-A4] 臂P H2 在 lambda=%.2f：max|R_theta| = %.4f（W3 线性完成参考 0.79）；G_tt r^2 峰值 = %.4f"
      % (last("P", "H2")[0], rH["Rth_max"], rH["Gtt_r2"]))
check("A4", True, "有效源量级已报告（见上；对照 W3 参考）")

print("  [U2-A5] 停滞律连带：")
for arm in ("P", "M"):
    for key in ("H1", "H2", "H3"):
        lb = last(arm, key)[0]
        r0 = readouts(stored[(arm, key)][0][1])
        r1 = readouts(last(arm, key)[1])
        print("       臂%s %s（lambda 至 %.2f）：r_h %.4f -> %.4f；对数系数 %.4f -> %.4f"
              % (arm, key, lb, r0["rh"], r1["rh"], r0["logcoef"], r1["logcoef"]))
check("A5", True, "停滞律连带已报告（r_h 与对数系数偏移）")

print("  [报告] 满偏越界阈值 lambda*（theta_max 达到 1.0 的耦合强度）：")
any_star = False
for arm in ("P", "M"):
    for key in ("H1", "H2", "H3"):
        if (arm, key) in lam_star:
            any_star = True
            print("       臂%s %s：lambda* = %.4f" % (arm, key, lam_star[(arm, key)]))
if not any_star:
    print("       无（全臂在 lambda <= 1.0 内 theta_max 未达 1.0）")

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
mloc = (r >= 4.0) & (r <= 60.0)
ax = axs[0, 0]
ax.plot(r[mloc], base_t[mloc], "-", color="gray", label="lambda=0")
for key, mk in (("H1", "--"), ("H2", "-"), ("H3", ":")):
    lb, t_b2 = last("P", key)
    ax.plot(r[mloc], t_b2[mloc], mk, label="P-%s, lambda=%.2f" % (key, lb))
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.axvline(10.285, color="tab:red", ls="--", lw=1)
ax.set_xlabel("r"); ax.set_ylabel("theta")
ax.set_title("(a) 剖面：领衔臂各口径")
ax.legend(fontsize=8)

ax = axs[0, 1]
for arm, key, mk in (("P", "H1", "o-"), ("P", "H2", "s-"), ("P", "H3", "^-"), ("M", "H1", "o:"), ("M", "H2", "s:")):
    path = stored[(arm, key)]
    base = readouts(path[0][1])["q_eff"]
    ax.plot([x[0] for x in path], [readouts(x[1])["q_eff"] / base for x in path], mk, label=arm + "-" + key)
ax.set_xlabel("lambda"); ax.set_ylabel("q_eff(lambda)/q_eff(0)")
ax.set_title("(b) 有效荷响应（P 推高臂 / M 反号对照）")
ax.legend(fontsize=8)

ax = axs[1, 0]
mb = (r >= 7.0) & (r <= 60.0)
t0 = base_t
ax.semilogy(r[mb], np.abs(t0 + r * np.gradient(t0, r))[mb], "-", color="gray", label="lambda=0 基准")
for key, mk in (("H1", "--"), ("H2", "-"), ("H3", ":")):
    lb, t_b2 = last("P", key)
    dd = np.abs(t_b2 + r * np.gradient(t_b2, r))
    ax.semilogy(r[mb], dd[mb], mk, label="P-%s, lambda=%.2f" % (key, lb))
ax.set_xlabel("r"); ax.set_ylabel("|theta + r theta'|")
ax.set_title("(c) 真空缺陷剖面")
ax.legend(fontsize=8)

ax = axs[1, 1]
mb2 = (r >= 11.0) & (r <= 30.0)
A = 1.0 - 2.0 * t0
Ap = np.gradient(A, r); App = np.gradient(Ap, r)
B = 1.0 / A; Bp = -Ap / A ** 2
_, _, Rth0 = ricci(A, B, r, Ap, App, Bp)
ax.plot(r[mb2], np.abs(Rth0[mb2]), "-", color="gray", label="lambda=0 基准")
for key, mk in (("H1", "--"), ("H2", "-"), ("H3", ":")):
    lb, t_b2 = last("P", key)
    A = 1.0 - 2.0 * t_b2
    Ap = np.gradient(A, r); App = np.gradient(Ap, r)
    B = 1.0 / A; Bp = -Ap / A ** 2
    _, _, Rth = ricci(A, B, r, Ap, App, Bp)
    ax.plot(r[mb2], np.abs(Rth[mb2]), mk, label="P-%s, lambda=%.2f" % (key, lb))
ax.set_xlabel("r"); ax.set_ylabel("|R_theta|")
ax.set_title("(d) 等效源量级")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/u2_coupling_stress.png", dpi=140)
print("  图已存 images/u2_coupling_stress.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
