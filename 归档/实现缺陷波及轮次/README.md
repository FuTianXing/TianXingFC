# 实现缺陷波及轮次（归档）

**归档日期**：2026-10-24
**归档依据**：第 808 号轮定位到三处装置级实现缺陷，第 812 号轮在订正后的装置上测得动源档 16/16 全通过。本目录收存读数受这两轮波及、或读数已被这两轮作废的轮次产物。
**处置方式**：只读保存，内容未改一字。`git log --follow <路径>` 可查完整历史，`验证脚本/README.md` 的对应条目保留并标注归档位置。

## 三处装置级实现缺陷（第 808 号定位）

1. **3D 侧 Christoffel 求导下标错位**：`D[(i,j,k)] = d1_keep(gam[i,j], k, eps)` 的键语义与 `D[(mu,nu,sig)]` 的使用不一致，导致求导沿错轴进行。
2. **4D 侧 Ricci 标量收缩式漏指标**：`R = Ric00 + ...` 的对角和写法，正确式应按逆度规全收缩 `R = g^{mu nu} R_{mu nu}`。
3. **4D 侧 `R_{mu nu}` 的 a2 分项反号**：`a1 - a2` 应为 `a1 + a2`。

缺陷 1 使三维里奇张量整体错，缺陷 2 与 3 使四维侧标量与里奇张量错。三者叠加使第 762 至 804 号多轮读到的「3+1 侧与 4D 直接构造之差」成为伪影。

## 本目录收录的轮次

| 轮号 | 脚本 | 注记 | 处置 |
| --- | --- | --- | --- |
| 777 | `minimal_verification_3plus1_rebuild_moving.py` | 库内登记的 3+1 关系式的独立检验与符号订正 | 读数作废，符号订正改读 |
| 780 | `minimal_verification_ricci3d_spherical_check.py` | 三维里奇张量的 x 分量疑点的排除 | 读数作废 |
| 781 | `minimal_verification_moving_shift_3plus1.py` | 动源档 d_0 项的正确形式与差额的归零核对 | 读数作废，「结构性信息缺失」诊断作废 |
| 784 | `minimal_verification_relations_itemwise.py` | 三条 3+1 关系的逐条核对 | 读数作废 |
| 788 | `minimal_verification_g0i_adm.py` | G_0i 侧的 ADM 关系式的正确形式 | 读数作废 |
| 792 | `minimal_verification_gij_adm_terms.py` | G_ij 侧的完整 ADM 分解的逐项定位 | 读数作废 |
| 793 | `minimal_verification_r3_compatibility.py` | R3 的数值相容性的闭合 | 核心结论已由第 794 号撤回，随本目录归档 |
| 794 | `minimal_verification_gamma_forms.py` | 三种 gamma 形式的 R3 对照 | 读数作废 |
| 801 | `minimal_verification_gij_source_localization.py` | G_ij 侧差额的来源定位 | 定量读数整轮撤回 |
| 802 | `minimal_verification_k_window_ratio_scan.py` | 驻留窗口档上的比值重测 | 与本线同源，随同轮次归档 |
| 803 | `minimal_verification_gij_gconst_localization.py` | N 常数构型上残余差额的来源 | 定量读数整轮撤回 |
| 804 | `minimal_verification_gij_proportionality.py` | 差额与 gamma 径向曲率严格正比 | 定量读数整轮撤回，比例约 2.4 是缺陷指纹 |

对应的通俗版注记一并收存于本目录，文件名带「通俗版」三字。

## 未归档的相关轮次

第 773、776 号（动源构型差额随 v 线性增长、差额不属任何单项）读数同样受缺陷波及，但其注记内的部分方法论结论（判据量须相对被解释的量来度量、细分十项的排除思路）仍成立，且第 776 号抓到的「重建式 d_0 项写错」指向与第 808 号缺陷 3 同源，故留作诊断线索，未移入本目录。

第 762、765、767、770 号读数取自另一套装置（一维径向算子），不含上述三处指纹，其结论（差额归零、不依赖球对称）仍成立，未归档。

## 权威结论

订正后的装置上：静场档 3+1 侧与 4D 侧逐位相容（`G_ij` 侧残余 5.551e-17）；动源档三口径残余中 ADM lapse 档最小，shift 项随 gamma 梯度线性增长（对数斜率 1.0669 与 1.0412）。见第 812 号注记与通俗版。
