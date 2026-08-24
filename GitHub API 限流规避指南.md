# GitHub API 限流规避指南

## 问题

用 `Invoke-RestMethod` 匿名直连 `api.github.com` 查询数据时频繁遇到：

```
403 (rate limit exceeded): API rate limit exceeded for <IP>.
(But here's the good news: Authenticated requests get a higher rate limit.)
```

## 根因

| 访问方式 | 配额 | 说明 |
|---|---|---|
| 匿名 REST API | **60 次/小时/IP** | 按出口 IP 全局共享；search 接口每次扣 10 次权重 |
| gh CLI 已登录 | **5000 次/小时** | token 存于 keyring（本机已登录 FuTianXing） |

匿名配额是整个出口 IP 共享的，多设备/多会话很快烧完，重置需等整点。

## 正确做法：一律用 gh CLI

```powershell
# 单个对象
gh api repos/<owner>/<repo>/issues/<n>

# issue 详情
gh issue view <n> --repo <owner>/<repo>

# PR 详情（files[].patch 含每个文件的完整 diff）
gh pr view <n> --repo <owner>/<repo> --json files,body,additions,deletions

# 实时查看剩余配额
gh api rate_limit
```

输出为 JSON 时用 `ConvertFrom-Json` 解析。

## 匿名兜底方案

必须匿名看网页内容（非 API）时，用 Obscura 无头浏览器导航 + snapshot 绕开 API 限流
（网页端不受 API 配额约束）。