# `_tools/` 索引

本目录存放 agent 任务临时脚本（批量替换、数据处理、一次性生成工具）。与 `验证脚本/`（理论验证材料：判据 + 注记 + 脚本）的分工见 `AGENTS.md`「数值验证脚本规范」。

清理判据只有一条：该文件是否被工作区文档按 `_tools/<文件名>` 点名引用。零引用的一次性脚本（尤其修改替换类）在对应任务收尾后及时删除，纪律见 `AGENTS.md`「临时脚本清理规范」。`验证脚本/` 下的 `minimal_verification_*.py` 是永久资产，严禁删除。

## 常驻检查器

| 脚本 | 用途 | 用法与退出码 |
|---|---|---|
| `check_latex_tears.py` | 扫描 LaTeX 撕裂。向 md 写入含 LaTeX 的内容时若用普通字符串而非 raw string，`\ne`、`\nabla`、`\langle` 的 `\n` 与 `\rangle` 的 `\r` 会被当作转义符执行，命令被吃掉前两字符，断口处留下余部。判据为双条件交叉：行首为已知命令余部，且该行 `$` 计数为奇数。 | `python _tools/check_latex_tears.py [--all]`。默认跳过 `_tools/`（其中 `.bak` 是修复前快照，理应保留损坏）。0 无残留，1 有残留。 |
| `check_cross_refs.py` | 开放问题总览引用图谱：条目数与唯一键、前向引用、单向引用对、零入引条目、裸引用「第 N 条」。 | `python _tools/check_cross_refs.py [路径] [--full]`。0 无异常，1 检出编号重复或裸引用。 |

两者建议纳入每次批量改稿后的自检，与 `AGENTS.md`「修改后必查 em-dash、禁句式、账字」并列。

## 被文档引用的溯源脚本

以下脚本在 `验证脚本/` 注记中标注为「未固化」，即验证结论已写入注记，脚本本体仍留在本目录，且注记与 `验证脚本/README.md` 按 `_tools/<文件名>` 路径引用。移动或改名之前须先更新引用方。

| 脚本 | 引用方 |
|---|---|
| `scar_probe_topo.py`、`scar_probe_topo_q.py` | `验证脚本/README.md` 第 13 条；同上注记 |
| `elastic_identity_test.py` | `验证脚本/注记-复合模型：粒子 = 均匀缠绕背景上的 FK kink 位错与宽度参数区.md` |
| `explore_Pbar_doga.py`、`explore_Pbar_doga2.py`、`explore_Pbar_doga3.py` | `内部草稿/从混沌到兴盛：大爆炸作为天行力校准的相变.md`（P4-26 证据脚注）；`注记-温度定义：相位无序度的能量化标度.md`；`温度攻坚成果总览.md`；`开放问题总览` |
| `explore_complementary_temperature.py`、`explore_complementary_temperature2.py`、`explore_complementary_temperature3.py`、`explore_endpoint_budget.py` | `注记-温度定义的波粒互补扩展：双侧激发度与等相干平衡.md`；`温度攻坚成果总览.md`；`评审报告/评审-温度定义的波粒互补扩展：双侧激发度与等相干平衡`；`开放问题总览` |
| `quantum_bridge_a_s1.py` | `总结报告-引力机制整改与张量桥分段ab攻坚：2026-09-10.md`；`注记-张量桥分段a S1探测结果：纯规范陷阱重现与判据修正.md`；`注记-张量桥分段a攻坚计划：SO(3)标架3+1定量装配与史瓦西判据.md` |
| `review_preflight.py` | `评审报告/温度的波粒互补定义-评审报告-2026-09-08.md` |
| `gen_spiral_ladder_explain.py` | 生成 `可视化/螺旋阶梯几何_信息图.html`，改动该信息图时需复用 |
| `gen_kepu_illustrations.py` | 生成 `images/kepu_spiral_staircase.png` 与 `images/kepu_three_doors.png`，供 `科普文章/一条螺旋线等了几十年：最简单的骨架为什么最晚被发现.md` 使用；改动插图时需复用（含中文字体断言） |
| `_backup_微观根基_pre_revise3.md` | `评审报告/修改报告-爱因斯坦场方程的微观根基-2026-09-10-第三轮.md`。该快照与当前文件及 git 全部历史提交均不等价，是第三轮修订前状态的唯一副本 |

**2026-09-16 已清理**（提交 `20436aa`）：`scar_probe_single.py`、`scar_probe_lift.py`、`merge_t4_u1.py`、`merge_t4_u1_q.py`、`z2_l3_check.py`、`t5_quantum_width.py`、`composite_kink_width.py` 七只按引用溯源清除（均曾入库，可从历史提交找回）；相关注记已改指常驻脚本：状态图与宽度扫描升为 `验证脚本/minimal_verification_state_graph.py` 与 `minimal_verification_composite_width.py`，T4、T5 由 `sector_dynamics` 接续，量子层 $Z_N$ 部分由 `zn_sectors` 接续。

## 功能可复用但暂无文档引用

| 文件 | 说明 |
|---|---|
| `preview_server.js` | 工作区根目录通用静态预览服务器，端口 8766，做 `可视化/` 时可用 |
| `diag_z3.py`、`diag_z3_wx.py` | 与 `z2_l3_check.py`、`merge_t4_u1_q.py` 同属纠缠侧承载攻坚的姊妹脚本，尚未被文档点名 |

## `_stage0/`

开放问题总览重构（schema 化、关闭项回迁、去重、长条目分级列表化）的进行中执行材料，含修复脚本与各轮修复前备份。引用方为 `内部草稿/优化方案-开放问题总览重构：2026-09-11.md`。其中 `fix_stage0.py`、`fix_glossary.py`、`fix_note_entangle.py`、`validate_stage0.py` 的判据（行首为 LaTeX 命令余部且 `$` 计数为奇数，再核对断口前一行是否以半截公式结尾）具通用性，可复用。该重构收尾前不得清理。

## `_tmp/`

既定暂存位，已在 `.gitignore` 排除。存放脚本运行期产物（stdout 日志、迭代截图）。

## 历史清理

2026-09-11 清理 112 个零引用文件：62 个根级脚本（图片生成 API 调试族 19、2026-09-09 批量修订族 29、定名与登记族 13、其他 1）与 `_tmp/` 下 50 个运行产物。全部移入 `_tools/_trash_20260911/`（112 个文件，10.5 MB，已在 `.gitignore` 排除），未执行删除。62 个脚本均在 git 索引中，可从历史提交找回；`_tmp/` 内容从未入库。备份保留至作者确认后可删。
2026-09-18 清空 `_tools/_trash_20260911/`（作者指令：清理）：112 个文件、10.46 MB 已删除；其中 62 个根级脚本在 git 索引中，可从历史提交找回，`_tmp/` 下 50 个运行产物（stdout 日志与迭代截图）从未入库。`_tools/_tmp/` 维持为空（既定暂存位）。
