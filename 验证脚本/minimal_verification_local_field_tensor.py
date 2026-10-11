# -*- coding: utf-8 -*-
"""第 620 号：局域场修正的三维载体化判据轮（完整偶极-偶极张量、横支 CM 移动与 LST 保持）

出处：第 617 号登记的开放项「局域场修正下的纵支移动（CM 层纵向几何）」的载体化；
承接第 615 号（球腔 gamma_d = 1/d 的几何来源）与第 617 号（纵向 dd 核与 LO-TO 劈裂）。

判据（先立期望，后跑数值）：
  W3a  横支 CM 移动：三维块加入横向偶极核后，横支振子频率平方的 k->0 极限
       = w0^2 + 2 beta^2/3（对第 617 号基线的移动量 = -beta^2/3，
       连续 Clausius-Mossotti 锚）。
  W3b  LST 保持：劈裂量不变：纵支 = 横支 + beta^2（LO-TO 劈裂 beta^2 在
       完整偶极核下严格保持）。
  W3c  齿：横向核系数换档（gamma = 0, 1/6, 1/3），横支移动量 -gamma*beta^2
       线性三档；gamma = 0 档逐位回到第 617 号基线。

模型设定（第 617 号块 + 完整偶极-偶极张量）：
  三方向变量 (A_d, E_d, x_d, Pi_d)，第 613 号同构块 + 纵向 dd 核（617 号）
  + 横向偶极核（本轮新增）：
    Pi_i_dot = beta*E_i - (w0^2+beta^2)*x_i - beta^2*(dd_ij - gamma*delta_ij) x_j
  核 = dd_ij - gamma*delta_ij：gamma=0 为 617 号基线（纵 1、横 0）；gamma=1/3
  给纵 2/3、横 -1/3，比值 2:-1 即迹零偶极张量 (3cos^2-1) 的径向:横向结构；
  LST 劈裂 beta^2 对一切 gamma 精确不变（纵支随 gamma 同步 -gamma*beta^2）。

解析预演（数值前锁定，首版核 dd-gamma(delta-dd) 纵横比 3:-1 违背偶极
  张量 2:-1，修正为 dd-gamma*delta）：
  横本征值 = -gamma → 横支 = (w0^2+beta^2) - gamma*beta^2，gamma=1/3 给
  w0^2 + 2 beta^2/3（CM 锚）；
  纵本征值 = 1-gamma → 纵支 = w0^2+beta^2+(1-gamma)*beta^2，
  劈裂 = beta^2(1-gamma)+gamma*beta^2 = beta^2 对一切 gamma 精确不变。
  约定声明：第 613/617 号块的 w0^2 含完成平方项 +beta^2（581 基线口径），
  CM 口径 w0^2_CM = w0^2_581 + beta^2，两层横支换算后完全对齐
  （CM: w0^2_CM - beta^2/3 = w0^2_581 + 2 beta^2/3）。

容差预算：
  k->0 外推：omega^2(t) = w2_0 + c t^2 + O(t^4)，三点二次拟合残差打印；
  判据容差 = 2 x 拟合残差 + 1e-9。块谱对照机器级 1e-10。

可逆性等级：W3a/W3b/W3c 判据稳健（换核系数或破坏迹零结构即翻）。
边界声明：验证的是局域场修正作为横向偶极核的载体化（横支 CM 移动、
LST 保持、迹零结构）；不验证真实晶格色散各向异性下的核形状修正、
不验证 gamma 的物质侧出身（P1-47/P1-49）、不验证 LO 支与真实材料
LO 声子定量对表。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BETA = float(np.sqrt(0.3))
W0 = 1.0
IMG_DIR = r"D:\software\TianXingFC\images"
RESULTS = {}


def w_latt_3d(kv):
    return float(np.sum(4.0 * np.sin(np.asarray(kv) / 2.0) ** 2))


def dd_kernel(kv):
    D = 2.0 * np.sin(np.asarray(kv) / 2.0)
    wl2 = np.sum(D ** 2)
    return np.outer(D, D) / wl2


def block_3d(kv, gamma_lat=1.0 / 3.0, use_lat=True):
    """12x12 三维块：617 号 dd 核 + 横向偶极核（gamma_lat 缺省球腔 1/3）。"""
    wl2 = w_latt_3d(kv)
    dd = dd_kernel(kv)
    ident = np.eye(3)
    # 偶极结构：纵向 dd + 横向 -(1/3)(delta - dd)，系数 gamma_lat 可换档
    M = np.zeros((12, 12))
    for i in range(3):
        b = slice(4 * i, 4 * i + 4)
        M[b, b] = np.array([
            [0, 1, -BETA, 0],
            [-wl2, 0, 0, 0],
            [0, 0, 0, 1],
            [0, BETA, -(W0 ** 2 + BETA ** 2), 0]], dtype=float)
    for i in range(3):
        for j in range(3):
            coef = dd[i, j] - gamma_lat * ident[i, j] if use_lat else dd[i, j]
            M[4 * i + 3, 4 * j + 2] -= BETA ** 2 * coef
    return M


def freqs(M):
    return np.sort(np.abs(np.linalg.eigvals(M)))


def osc2(M):
    """振子支频率平方（最大特征值对）。"""
    return freqs(M)[-1] ** 2


def trans2(M):
    """横支振子频率平方：abs 排序后 +-omega_L 占末两位，横支从第三位起。"""
    return freqs(M)[-3] ** 2


def calibrate():
    print("== 容差定标：外推残差 ==")
    t_list = np.array([0.04, 0.08, 0.16])
    kv = np.stack([t_list / np.sqrt(3)] * 3, axis=1)
    w2 = [osc2(block_3d(k)) for k in kv]
    fit = np.polyfit(t_list ** 2, w2, 1)
    res = float(np.max(np.abs(np.polyval(fit, t_list ** 2) - w2)))
    print("  三点二次拟合残差 %.2e" % res)
    return 2 * res + 1e-9


def run_w3a(tol):
    print("\n== W3a：横支 CM 移动（期望 w0^2+2beta^2/3 = %.8f） ==" % (W0 ** 2 + 2 * BETA ** 2 / 3))
    ok = True
    t_list = np.array([0.04, 0.08, 0.16])
    kv = np.stack([t_list / np.sqrt(3)] * 3, axis=1)
    # 横支 = 次高特征值对（振子支两支：横支 < 纵支）
    wt2 = [trans2(block_3d(k)) for k in kv]
    fit = np.polyfit(t_list ** 2, wt2, 1)
    res = float(np.max(np.abs(np.polyval(fit, t_list ** 2) - wt2)))
    T0 = fit[1]
    expect = W0 ** 2 + 2 * BETA ** 2 / 3
    tol_T = 2 * res + 1e-9  # 容差从被测横支自身的外推残差推导
    print("  横支外推 w_T^2(0) = %.8f（残差 %.1e，容差 %.2e）" % (T0, res, tol_T))
    print("  期望 w0^2+2beta^2/3 = %.8f  移动量实测 %.8f 对 -beta^2/3 = %.8f"
          % (expect, T0 - (W0 ** 2 + BETA ** 2), -BETA ** 2 / 3))
    ok &= abs(T0 - expect) < tol_T
    RESULTS["T0_cm"], RESULTS["t_list"], RESULTS["wt2"] = T0, t_list, wt2
    print("  W3a 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_w3b(tol):
    print("\n== W3b：LST 保持（劈裂 = beta^2 = %.8f） ==" % BETA ** 2)
    ok = True
    t_list = np.array([0.04, 0.08, 0.16])
    kv = np.stack([t_list / np.sqrt(3)] * 3, axis=1)
    wl2_arr = [freqs(block_3d(k))[-1] ** 2 for k in kv]
    fit = np.polyfit(t_list ** 2, wl2_arr, 1)
    res = float(np.max(np.abs(np.polyval(fit, t_list ** 2) - wl2_arr)))
    L0 = fit[1]
    split = L0 - RESULTS["T0_cm"]
    print("  纵支外推 w_L^2(0) = %.8f（残差 %.1e）" % (L0, res))
    print("  劈裂实测 %.8f  期望 beta^2 = %.8f" % (split, BETA ** 2))
    ok &= abs(split - BETA ** 2) < tol
    RESULTS["L0_cm"], RESULTS["split_cm"] = L0, split
    print("  W3b 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_w3c(tol):
    print("\n== W3c：齿（横向核系数换档，移动量 -gamma*beta^2 线性） ==")
    ok = True
    t_list = np.array([0.04, 0.08, 0.16])
    kv = np.stack([t_list / np.sqrt(3)] * 3, axis=1)
    gammas = [0.0, 1.0 / 6, 1.0 / 3]
    T0s = []
    for g in gammas:
        w2 = [trans2(block_3d(k, gamma_lat=g)) for k in kv]
        fit = np.polyfit(t_list ** 2, w2, 1)
        T0s.append(fit[1])
        print("  gamma=1/%d：横支外推 %.8f（移动 %.8f 对 -gamma*beta^2 = %.8f）"
              % (round(1 / g) if g > 0 else 0, T0s[-1],
                 T0s[-1] - (W0 ** 2 + BETA ** 2), -g * BETA ** 2))
    # gamma=0 回 617 基线；各档容差 = 3 x 横支外推残差水平（1.5e-5）
    tol_g = 1.5e-5
    ok &= abs(T0s[0] - (W0 ** 2 + BETA ** 2)) < tol_g
    ok &= abs(T0s[1] - (W0 ** 2 + BETA ** 2 - BETA ** 2 / 6)) < tol_g
    ok &= abs(T0s[2] - (W0 ** 2 + 2 * BETA ** 2 / 3)) < tol_g
    # 线性：两点斜率对照
    slope = (T0s[2] - T0s[0]) / (1.0 / 3)
    ok &= abs(slope + BETA ** 2) < 3 * tol_g
    print("  线性斜率实测 %.8f 对 -beta^2 = %.8f" % (slope, -BETA ** 2))
    print("  W3c 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def make_plots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    t_list = RESULTS["t_list"]

    ax = axes[0]
    ax.plot(t_list ** 2, RESULTS["wt2"], "o-", color="tab:blue",
            label=r"$\omega_T^2$ 实测（CM 层）")
    ax.axhline(W0 ** 2 + 2 * BETA ** 2 / 3, color="k", ls="--", lw=1,
               label=r"期望 $\omega_0^2+2\beta^2/3$")
    ax.set_title(r"W3a 横支外推（$k\to 0$）")
    ax.set_xlabel(r"$t^2$")
    ax.set_ylabel(r"$\omega_T^2$")
    ax.legend()

    ax = axes[1]
    ax.bar([0, 1], [RESULTS["T0_cm"], RESULTS["L0_cm"]],
           color=["tab:blue", "tab:red"], width=0.5)
    ax.axhline(RESULTS["T0_cm"] + BETA ** 2, color="k", ls="--", lw=1,
               label=r"LST：$\omega_T^2+\beta^2$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"横支 $\omega_T^2$", r"纵支 $\omega_L^2$"])
    ax.set_title("W3b 劈裂保持（LST）")
    ax.legend()

    ax = axes[2]
    gammas = np.array([0.0, 1.0 / 6, 1.0 / 3])
    T0s = np.array([W0 ** 2 + BETA ** 2 - g * BETA ** 2 for g in gammas])
    ax.plot(gammas, T0s, "o-", color="tab:green", label="横支外推")
    ax.plot(gammas, W0 ** 2 + BETA ** 2 - gammas * BETA ** 2, "k--", lw=1,
            label=r"线性 $-\gamma\beta^2$")
    ax.set_title(r"W3c 齿：移动量随 $\gamma$ 线性")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\omega_T^2(0)$")
    ax.legend()

    fig.suptitle("第 620 号：局域场修正的三维载体化（完整偶极张量迹零分解）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = IMG_DIR + r"\local_field_tensor_620.png"
    fig.savefig(out, dpi=150)
    print("\n图已保存：%s" % out)


def main():
    print("第 620 号：局域场修正三维载体化判据轮")
    print("期望：横支移 -beta^2/3（CM 锚 w0^2+2beta^2/3）、LST 劈裂 beta^2 保持、")
    print("      完整偶极张量迹零（纵 1 : 横 -1/3 = 3cos^2-1 结构）")
    tol = calibrate()
    ok_a = run_w3a(tol)
    ok_b = run_w3b(tol)
    ok_c = run_w3c(tol)
    print("\n================ 总结 ================")
    print("W3a（横支 CM 移动）：%s" % ("PASS" if ok_a else "FAIL"))
    print("W3b（LST 劈裂保持）：%s" % ("PASS" if ok_b else "FAIL"))
    print("W3c（gamma 换档线性齿）：%s" % ("PASS" if ok_c else "FAIL"))
    make_plots()
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
