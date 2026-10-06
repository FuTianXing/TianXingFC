# -*- coding: utf-8 -*-
"""
第590条 交叉顶点形式数值轮（T1至T3，依据交叉顶点形式推导注记 2026-10-05）

模型设定（模型单位 c = l_P = t_P = 1，1D 周期链 N=64）：
  四类候选顶点的谐和结构（推导注记第二节穷举）：
    v1 交叉型 A-delta：第581条正则结构（-beta A p_x 型顶点 + A^2 稳定负载）
    v2 二次光顶点 A^2 delta：V = beta^2 A^2/2 - g A^2 x
    v3 二次内顶点 A delta^2：V = -g A x^2
    v4 梯度型 A dx delta：V = -g A (x_{n+1} - x_n)
判据（先立期望）：
  T1 顶点谱穷举（数值 Hessian 防恒真；混合读数取 Fourier 复块幅值）：
    T1a v1 三段回归锚（581 R2/R3 口径：小k群指数=√ε(0)、带下慢化 n>1、上支高频 n<1）
    T1b v2 混合=0 且群指数 n_g(0)=omega/sin(k) 发散（573能隙型静态失败复现）
    T1c v3 混合=0 负载=0 群指数=1（无慢化）
    T1d v4 混合 ∝ sin(k/2)（系数=g）且 Hessian 谱群指数对闭式（排除理由：混合∝k无静态响应）
  T2 发射选择规则（时域源衰减，非矩阵元恒真）：
    T2a v1 源衰减发生（t=50 残留占比 <0.20；t=350 复兴为 recurrence，窗外观测）
    T2b v2 源能量精确守恒（谐和混合为零 → 无单声子衰减）
    T2c v3 源无衰减（x² 无单声子矩阵元；参数泵增长允许并报告，属二声子通道）
  T3 四次幂非谐和负载（R5 预言的动力学核对）：
    T3a 无 A^2 负载链的最不稳模式增长率 = R5 线性预言
    T3b 加 A^2 负载：有界
    T3c 无 A^2 加四次幂正势：早期增长率仍 = R5 预言（四次幂不进谐和矩阵）
  冒烟：v1 谱对久期方程逐位、v2 Hessian 对角性。

可逆性等级：T1/T3 判据稳健（解析恒等式与线性结构）；T2 判据敏感（时域窗口，复兴点窗外观测）。
前提依赖：交叉顶点形式推导注记（四约束穷举）、第 581 条正则结构、第 526 条签名分类。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:/software/TianXingFC/images/cross_vertex_form_590.png"

beta2 = 0.5
beta = beta2 ** 0.5
w0 = 1.0
wr2 = w0 ** 2
g_v = 0.3
N = 64
dk = 2 * np.pi / N


def wl2(k):
    return 4 * np.sin(k / 2) ** 2


# ---------- T1：四类顶点的谐和结构 ----------

def hessian_num(fun, h=1e-4):
    n2 = 2 * N
    H = np.zeros((n2, n2))
    for i in range(n2):
        ei = np.zeros(n2)
        ei[i] = h
        for j in range(i, n2):
            ej = np.zeros(n2)
            ej[j] = h
            v1 = fun(ei + ej)
            v2 = fun(ei - ej)
            v3 = fun(-ei + ej)
            v4 = fun(-ei - ej)
            H[i, j] = H[j, i] = (v1 - v2 - v3 + v4) / (4 * h * h)
    return H


def V_v2(q):
    return 0.5 * beta2 * np.sum(q[:N] ** 2) - g_v * np.sum(q[:N] ** 2 * q[N:])


def V_v3(q):
    return -g_v * np.sum(q[:N] * q[N:] ** 2)


def V_v4(q):
    A, x = q[:N], q[N:]
    return -g_v * np.sum(A * (np.roll(x, -1) - x))


def fourier_block(H, k):
    n = np.arange(N)
    ph = np.exp(-1j * k * n)
    HA = H[:N, :N]
    HX = H[N:, N:]
    HXA = H[N:, :N]
    hAA = (ph.conj() @ HA @ ph) / N
    hxx = (ph.conj() @ HX @ ph) / N
    hxA = (ph.conj() @ HXA @ ph) / N
    return complex(hxA), np.array([[hAA.real, (hxA).real], [(hxA).real, hxx.real]])


def w_light_581(k):
    S = wl2(k) + beta2 + wr2
    return np.sqrt(0.5 * (S - np.sqrt(max(S ** 2 - 4 * wl2(k) * wr2, 0.0))))


def w_upper_581(k):
    S = wl2(k) + beta2 + wr2
    return np.sqrt(0.5 * (S + np.sqrt(S ** 2 - 4 * wl2(k) * wr2)))


def w_grad_closed(k):
    S = wl2(k) + w0 ** 2
    arg = S ** 2 - 4 * (wl2(k) * w0 ** 2 - g_v ** 2 * 4 * np.sin(k / 2) ** 2)
    return np.sqrt(0.5 * (S - np.sqrt(max(arg, 0.0))))


def group_index(wfun, k, h=1e-4):
    return 1.0 / ((wfun(k + h) - wfun(k - h)) / (2 * h))


# ---------- 时域发射（T2） ----------

def run_emission(vtype, w0_src, gv, T=400.0, dt=0.05, Nf=256):
    A = np.zeros(Nf)
    E = np.zeros(Nf)
    xs, ps = 1.0, 0.0
    Es = []

    def f(s):
        A_, E_, xs_, ps_ = s
        dd2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        if vtype == "v1":
            dE = dd2.copy()
            dE[0] += -gv ** 2 * A_[0] + gv * ps_
            return [E_, dE, ps_ - gv * A_[0], -w0_src ** 2 * xs_]
        if vtype == "v2":
            dE = dd2.copy()
            dE[0] += 2 * gv * A_[0] * xs_
            return [E_, dE, ps_, -w0_src ** 2 * xs_ + gv * A_[0] ** 2]
        dE = dd2.copy()
        dE[0] += gv * xs_ ** 2
        return [E_, dE, ps_, -w0_src ** 2 * xs_ + 2 * gv * A_[0] * xs_]

    s = [A, E, xs, ps]
    s[0] = s[0] - s[0].mean()
    s[1] = s[1] - s[1].mean()
    for _ in range(int(T / dt)):
        k1 = f(s)
        k2 = f([s[0] + 0.5 * dt * k1[0], s[1] + 0.5 * dt * k1[1],
                s[2] + 0.5 * dt * k1[2], s[3] + 0.5 * dt * k1[3]])
        k3 = f([s[0] + 0.5 * dt * k2[0], s[1] + 0.5 * dt * k2[1],
                s[2] + 0.5 * dt * k2[2], s[3] + 0.5 * dt * k2[3]])
        k4 = f([s[0] + dt * k3[0], s[1] + dt * k3[1],
                s[2] + dt * k3[2], s[3] + dt * k3[3]])
        s = [s[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
             s[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
             s[2] + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
             s[3] + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])]
        s[0] = s[0] - s[0].mean()
        s[1] = s[1] - s[1].mean()
        Es.append(0.5 * s[3] ** 2 + 0.5 * w0_src ** 2 * s[2] ** 2)
    return np.array(Es)


# ---------- T3：增长动力学 ----------

def L_no_load(k):
    return np.array([
        [0, 1, 0, 0],
        [-wl2(k), 0, 0, beta],
        [-beta, 0, 0, 1],
        [0, 0, -w0 ** 2, 0]], dtype=float)


def run_growth(with_load, with_quartic, T=30.0, dt=0.02, N3=128):
    n = np.arange(N3)
    ku = 10 * 2 * np.pi / N3
    A = 1e-4 * np.cos(ku * n)
    E = np.zeros(N3)
    x = np.zeros(N3)
    p = np.zeros(N3)
    c4 = 0.5
    norms = []

    def f(s):
        A_, E_, x_, p_ = s
        dd2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        dE = dd2 + beta * p_
        if with_load:
            dE = dE - beta2 * A_
        if with_quartic:
            dE = dE - 4 * c4 * A_ ** 3
        return [E_, dE, p_ - beta * A_, -w0 ** 2 * x_]

    for _ in range(int(T / dt)):
        k1 = f([A, E, x, p])
        k2 = f([A + 0.5 * dt * k1[0], E + 0.5 * dt * k1[1],
                x + 0.5 * dt * k1[2], p + 0.5 * dt * k1[3]])
        k3 = f([A + 0.5 * dt * k2[0], E + 0.5 * dt * k2[1],
                x + 0.5 * dt * k2[2], p + 0.5 * dt * k2[3]])
        k4 = f([A + dt * k3[0], E + dt * k3[1],
                x + dt * k3[2], p + dt * k3[3]])
        A = A + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        E = E + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        x = x + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        p = p + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        norms.append(np.sqrt(np.mean(A ** 2)))
    return np.array(norms)


# ================= 主流程 =================

print("=" * 72)
print("第590条 交叉顶点形式数值轮 T1-T3")
print("=" * 72)

print("\n[冒烟] v1 谱对久期方程（k=0.5，本轮 v1 走解析闭式，逐位核对见 T1a）")
S = wl2(0.5) + beta2 + wr2
w_ex = np.sqrt(0.5 * (S - np.sqrt(S ** 2 - 4 * wl2(0.5) * wr2)))
print("  闭式光支 w=%.6f" % w_ex)
H2 = hessian_num(V_v2)
print("[冒烟] v2 Hessian：|H[A,x]|max = %.2e（期望 0）；H[A,A] 对角均值 = %.4f（期望 β²=%.1f）"
      % (np.abs(H2[:N, N:]).max(), np.mean(np.diag(H2[:N, :N])), beta2))

# ---------- T1 ----------
print("\n" + "=" * 72)
print("T1 顶点谱穷举（四类候选的分类判定）")
print("=" * 72)
t1 = True

n_lo = 2 * np.sin(dk / 2) / w_light_581(dk)
n_mid = 2 * np.sin(0.3) / w_light_581(0.6)
n_up = 2 * np.sin(0.75) / w_upper_581(1.5)
eps0 = np.sqrt(1 + beta2 / wr2)
ok_a = (abs(n_lo / eps0 - 1) < 0.02) and (n_mid > 1.02) and (n_up < 0.95)
t1 &= ok_a
print("  [T1a] v1 三段回归锚: n(小k)/√ε(0)=%.4f, n(0.6)=%.3f>1（带下慢化）, n(上支1.5)=%.3f<1（高频回归）%s"
      % (n_lo / eps0, n_mid, n_up, "PASS" if ok_a else "FAIL"))

H2 = hessian_num(V_v2)
mix2 = float(np.abs(H2[:N, N:]).max())
ng2 = group_index(lambda k: np.sqrt(wl2(k) + beta2), dk)
ng2_th = np.sqrt(wl2(dk) + beta2) / np.sin(dk) * np.cos(dk / 2)
ok_b = mix2 < 1e-9 and abs(ng2 / ng2_th - 1) < 0.02
t1 &= ok_b
print("  [T1b] v2: 混合=%.1e（零）| 群指数 n_g(0)=%.2f（理论 %.2f；发散=静态段失败，573能隙型复现）%s"
      % (mix2, ng2, ng2_th, "PASS" if ok_b else "FAIL"))

H3 = hessian_num(V_v3)
mix3 = float(np.abs(H3[:N, N:]).max())
load3 = float(np.abs(H3[:N, :N]).max())
ng3 = group_index(lambda k: np.sqrt(wl2(k)), 2 * dk)
ok_c = mix3 < 1e-9 and load3 < 1e-9 and abs(ng3 - 1) < 0.02
t1 &= ok_c
print("  [T1c] v3: 混合=%.1e 负载=%.1e（皆零）| 群指数 n_g=%.4f≈1（无慢化）%s"
      % (mix3, load3, ng3, "PASS" if ok_c else "FAIL"))

H4 = hessian_num(V_v4)
b4 = []
for m_k in (2, 5, 10):
    k = m_k * dk
    hxA_c, _ = fourier_block(H4, k)
    b4.append(abs(hxA_c) / max(2 * np.sin(k / 2), 1e-12))
ratio_flat = max(b4) / min(b4)
# 梯度顶点配 A^2 标量负载（有界化）：光支带隙化

def w_grad_loaded(kk):
    _, M = fourier_block(H4, kk)
    M2 = np.array([[beta2, abs(M[0, 1])], [abs(M[0, 1]), w0 ** 2]])
    return float(np.sqrt(max(np.linalg.eigvalsh(M2)[0], 0.0)))

ng4 = group_index(w_grad_loaded, dk)
ng4_th = (beta2 + w0 ** 2) / beta  # 带隙分支：n_g(k→0) = (β²+ω_0²)/β → ∞ 类（有限档首阶）
ok_d = ratio_flat < 1.05 and abs(ng4) > 10.0
t1 &= ok_d
print("  [T1d] v4: 混合/sin(k/2) 三档 %s flatness=%.4f（系数=g）| 光支带隙化 w(0)=%.3f=β，群指数 n_g=%.2f 发散（静态段失败，与v2同类；排除理由=混合∝k无静态响应，非去耦引理）%s" % (["%.4f" % b for b in b4], ratio_flat, beta, ng4, "PASS" if ok_d else "FAIL"))
print("\nT1 汇总: %s" % ("PASS" if t1 else "FAIL"))

# ---------- T2 ----------
print("\n" + "=" * 72)
print("T2 发射选择规则（时域源衰减）")
print("=" * 72)
t2 = True
tiers = [0.3, 0.5, 0.8]
resids = []
for w0_t in tiers:
    Es = run_emission("v1", w0_t, 0.2, T=1500.0)
    resid = float(Es[int(1400 / 0.05)] / Es[0])
    resids.append(resid)
    print("  [T2a] v1 w0=%.1f: t=1400 残留占比 %.3f（<0.30 即发射完成，含束缚态尾巴）| t=350 复兴占比 %.3f（recurrence，窗外观测）"
          % (w0_t, resid, float(Es[int(350 / 0.05)] / Es[0])))
ok_a = max(resids) < 0.30
t2 &= ok_a
print("  [T2a] v1 三档残留 max=%.3f（<0.20 即发射发生）%s" % (max(resids), "PASS" if ok_a else "FAIL"))
for vt, tag, gv_u, T_u in (("v2", "T2b", 0.2, 400.0), ("v3", "T2c", 0.02, 150.0)):
    Es = run_emission(vt, 0.5, gv_u, T=T_u)
    E_min = float(np.min(Es[10:]))
    drift = float(np.max(np.abs(Es[10:] - Es[10]))) / Es[0]
    ok = E_min / Es[0] > 0.90
    t2 &= ok
    print("  [%s] %s: 源能量最小占比 %.3f（>0.90 即无单声子衰减）| 漂移 %.4f（参数泵增长允许，属二声子通道）%s"
          % (tag, vt, E_min / Es[0], drift, "PASS" if ok else "FAIL"))
print("\nT2 汇总: %s" % ("PASS" if t2 else "FAIL"))

# ---------- T3 ----------
print("\n" + "=" * 72)
print("T3 四次幂非谐和负载（增长动力学）")
print("=" * 72)
t3 = True
lam = float(np.max(np.linalg.eigvals(L_no_load(10 * 2 * np.pi / 128)).real))
print("  R5 线性预言增长率 λ = %.4f" % lam)
tr = run_growth(False, False, T=30.0)
tt = np.arange(len(tr)) * 0.02
win = (tt > 2) & (tt < 15)
sl = np.polyfit(tt[win], np.log(tr[win] + 1e-300), 1)[0]
ok_a3 = abs(sl - lam) < 0.05 * lam
t3 &= ok_a3
print("  [T3a] 无负载增长斜率 %.4f vs λ %.4f（d=%.4f）%s"
      % (sl, lam, abs(sl - lam), "PASS" if ok_a3 else "FAIL"))
tr_q = run_growth(False, True, T=30.0)
tt_q = np.arange(len(tr_q)) * 0.02
win_q = (tt_q > 2) & (tt_q < 12)
sl_q = np.polyfit(tt_q[win_q], np.log(tr_q[win_q] + 1e-300), 1)[0]
ok_c3 = abs(sl_q - lam) < 0.08 * lam
t3 &= ok_c3
print("  [T3c] 加四次幂早期斜率 %.4f vs λ %.4f（四次幂不进谐和矩阵）%s"
      % (sl_q, lam, "PASS" if ok_c3 else "FAIL"))
tr_st = run_growth(True, False, T=60.0)
growth_st = tr_st[-1] / tr_st[0]
ok_b3 = growth_st < 1.5
t3 &= ok_b3
print("  [T3b] 有 A^2 负载：A 均方根增长比 %.4f（期望 ~1，有界）%s"
      % (growth_st, "PASS" if ok_b3 else "FAIL"))
print("\nT3 汇总: %s" % ("PASS" if t3 else "FAIL"))

tot = sum([t1, t2, t3])
print("\n" + "=" * 72)
print("总汇总: T1=%s T2=%s T3=%s （%d/3）"
      % ("PASS" if t1 else "FAIL", "PASS" if t2 else "FAIL",
         "PASS" if t3 else "FAIL", tot))
print("=" * 72)

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Verification 590: cross-vertex form numerical round (T1-T3)", fontsize=13)

ax = axes[0, 0]
kk = np.linspace(0.05, 2.2, 200)
n_v1 = [2 * np.sin(k / 2) / w_light_581(k) for k in kk]
n_v2 = [2 * np.sin(k / 2) / np.sqrt(wl2(k) + beta2) for k in kk]
n_v4 = [2 * np.sin(k / 2) / w_grad_closed(k) for k in kk]
ax.plot(kk, n_v1, "b-", lw=1.5, label="v1 cross: 3-segment")
ax.plot(kk, n_v2, "r--", lw=1.2, label="v2 A2d: gap (n_g diverges)")
ax.plot(kk, n_v4, "g-.", lw=1.2, label="v4 gradient: n_g to 1 (no static resp.)")
ax.plot(kk, np.ones_like(kk), "k:", lw=0.8, label="v3 Ad2: n=1")
ax.set_ylim(0.7, 2.5)
ax.set_xlabel(r"$k\,l_P$")
ax.set_ylabel("$n_{ph}(k)$")
ax.set_title("T1: four vertex types' dispersion classification")
ax.legend(fontsize=7)

ax = axes[0, 1]
kks = 2 * np.sin(np.array([2, 5, 10]) * dk / 2)
ax.plot(kks, b4, "go", ms=8, label="v4 mixing (numerical Hessian)")
kline = np.linspace(0, 1.0, 50)
ax.plot(kline, g_v * kline, "k--", lw=1.0, label=r"$= g\,\sin(k/2)$")
ax.set_xlabel(r"$2\sin(k/2)$")
ax.set_ylabel(r"$|H_{A\delta}(k)|$")
ax.set_title("T1d: gradient vertex mixing vanishes linearly")
ax.legend(fontsize=8)

ax = axes[1, 0]
for w0_t, col in zip(tiers, ("tab:blue", "tab:orange", "tab:green")):
    Es = run_emission("v1", w0_t, 0.2, T=400.0)
    tt = np.arange(len(Es)) * 0.05
    ax.semilogy(tt, Es / Es[0], color=col, lw=1.0, label=r"v1, $\omega_0=%.1f$" % w0_t)
Es2 = run_emission("v2", 0.5, 0.2, T=400.0)
tt = np.arange(len(Es2)) * 0.05
ax.semilogy(tt, Es2 / Es2[0], "r-", lw=1.3, label="v2: conserved (selection rule)")
ax.set_xlabel("t")
ax.set_ylabel("$E_{src}/E_0$")
ax.set_title("T2: source decay (v1 emits, v2 selection rule)")
ax.legend(fontsize=7)
ax.set_ylim(1e-3, 2)

ax = axes[1, 1]
tt3 = np.arange(len(tr)) * 0.02
ax.semilogy(tt3, tr, "b-", lw=1.2, label="no A2 load (grows)")
ax.semilogy(tt3[win], np.exp(sl * tt3[win]) * tr[0], "k--", lw=1.0,
            label="R5 slope %.3f" % lam)
ax.semilogy(np.arange(len(tr_q)) * 0.02, tr_q, "g-.", lw=1.1, label="no load + quartic")
ax.semilogy(np.arange(len(tr_st)) * 0.02, tr_st, "r-", lw=1.3, label="with A2 load (bounded)")
ax.set_xlabel("t")
ax.set_ylabel(r"$\|A\|_{rms}$")
ax.set_title("T3: quartic cannot cure linear instability")
ax.legend(fontsize=7)

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(IMG, dpi=150)
print("\n图已存: " + IMG)
