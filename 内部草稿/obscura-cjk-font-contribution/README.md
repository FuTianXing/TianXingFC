# Upstream Contribution Package — Obscura CJK Font Support

本目录包含向上游 h4ckf0r0day/obscura 提交 issue 评论 / PR 所需的全部材料。

## 文件清单

| 文件 | 用途 |
|---|---|
| `issue-comment.md` | 发布到 issue #606 的英文评论（报告根因 + 临时方案 + 补丁） |
| `pr-description.md` | 开 PR 时的描述文本（若直接提 PR） |
| `obscura-system-fonts.patch` | 最终补丁（`git apply` 即可） |

## 提交流程建议

1. Fork 上游仓库 → clone fork → 建分支：
   ```
   git checkout -b fix/606-opt-in-system-fonts
   git apply obscura-system-fonts.patch   # 或手动套用 inline.rs 改动
   git commit -s
   git push origin fix/606-opt-in-system-fonts
   ```
2. 开 PR 时标题用 `render: opt-in system font loading via OBSCURA_LOAD_SYSTEM_FONTS for CJK text (#606)`，
   正文粘贴 `pr-description.md`。
3. 若只想先评论 issue：在 #606 下贴 `issue-comment.md` 内容并附上 patch 文件。

## 注意事项

- 补丁基于上游 main 分支（2026-08-25 快照，cosmic-text 为 vendor 路径版本）。
- PR 里建议说明：默认行为零变化（不设置环境变量时不触碰系统字体扫描），
  以回应上游对"跨机器布局确定性"的顾虑。
- 截图证据：修复前豆腐块截图 + 修复后百度首页正常渲染截图（可从本地 tmp 目录取）。
