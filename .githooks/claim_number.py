#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""占号串行化工具：把「读最大号、加一、写入 README、提交」整段放进互斥锁里。

背景：`验证脚本/README.md` 的编号由多条并行会话共享。手工取号时，
两个会话可能同时读到同一个最大号，各自加一得到同一个号，于是撞号。
本工具用互斥锁把这一整段串行化，后到者排队，读到的最大号必然已含
前者刚写入的那一行。

锁的实现：`O_CREAT | O_EXCL` 建锁文件，这是文件系统级的原子操作，
Windows 与 POSIX 都支持（Windows 无 fcntl，故不用 flock）。锁文件里写
进程号与时间戳，超过 TTL 视为陈旧锁并回收（防进程被强杀后留下死锁）。

用法：
    python .githooks/claim_number.py <本线名>
    python .githooks/claim_number.py <本线名> --dry-run

退出码：0 成功；1 参数或前置条件不满足；2 锁等待超时。
"""
import io
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO, "验证脚本", "README.md")
LOCK = os.path.join(REPO, ".githooks", ".claim.lock")
ENTRY = re.compile(r"^- \*\*(\d+)([a-z]*)\.")
TIMEOUT = 120.0
STALE = 300.0


def max_number(lines):
    nums = []
    for ln in lines:
        m = ENTRY.match(ln)
        if m and not m.group(2):
            nums.append(int(m.group(1)))
    return max(nums) if nums else 0


def dirty_readme():
    r = subprocess.run(["git", "status", "--short", "--", "验证脚本/README.md"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip()


def acquire(deadline):
    """原子地创建锁文件。返回 True 表示拿到锁。"""
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, ("%d %f" % (os.getpid(), time.time())).encode("ascii"))
            os.close(fd)
            return True
        except FileExistsError:
            # 陈旧锁回收
            try:
                age = time.time() - os.path.getmtime(LOCK)
            except OSError:
                age = 0.0
            if age > STALE:
                try:
                    os.unlink(LOCK)
                    continue
                except OSError:
                    pass
            if time.time() > deadline:
                return False
            time.sleep(0.3)
        except OSError:
            return False


def release():
    try:
        os.unlink(LOCK)
    except OSError:
        pass


def main(argv):
    dry = "--dry-run" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print("用法：python .githooks/claim_number.py <本线名> [--dry-run]")
        return 1
    line_name = args[0]

    if not os.path.exists(README):
        print("找不到 " + README)
        return 1

    if not dry and dirty_readme():
        print("拒绝作业：验证脚本/README.md 有未提交改动。")
        print("先提交或还原它再占号，否则锁保护不到你读到的那一版。")
        return 1

    if not acquire(time.time() + TIMEOUT):
        print("锁等待超时（%.0f 秒）：另一会话正在占号，稍后重试。" % TIMEOUT)
        return 2

    try:
        lines = io.open(README, encoding="utf-8").read().splitlines()
        cur = max_number(lines)
        target = cur + 1

        last = None
        for i, ln in enumerate(lines):
            if ENTRY.match(ln):
                last = i
        if last is None:
            print("README 里找不到任何编号条目，结构异常，未改动。")
            return 1

        row = "- **%d.** 占位：%s 施工中" % (target, line_name)

        if dry:
            print("当前最大号 %d，将占 %d" % (cur, target))
            print("待插入：" + row)
            return 0

        lines = lines[:last + 1] + [row] + lines[last + 1:]
        io.open(README, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

        r = subprocess.run(["git", "add", "--", "验证脚本/README.md"],
                           cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print("git add 失败：" + (r.stderr or "")[:300])
            return 1
        r = subprocess.run(["git", "commit", "-q", "-m", "占位：%s（第 %d 号）" % (line_name, target)],
                           cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print("提交失败（可能被 pre-commit 钩子拦下）：" + (r.stderr or r.stdout or "")[:400])
            return 1

        print("已占号：第 %d 号（%s）" % (target, line_name))
        print("下一步：把 %d 写进脚本名与注记名，交付时用正式行替换占位行。" % target)
        return 0
    finally:
        release()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
