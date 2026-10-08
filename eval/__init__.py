"""评估包。

这里只做一件事：把标准输出强制设成 UTF-8。

Windows 下 `python -m eval.run > out.log` 会让 stdout 退回系统 ANSI 代码页
（简中环境是 gbk），于是脚本里的「¥」「—」这类字符一打印就抛
`UnicodeEncodeError`，进程当场死掉。而死法很有欺骗性：

- 日志里只剩半行输出，看起来像「还在跑」
- 若命令以 `; echo "退出码 $?"` 收尾，退出码会被 echo 洗成 0，
  后台任务通知于是显示「completed (exit code 0)」

实测两次都是靠 grep 失败标志才发现的（`lessons.md` 7.29）。
在包初始化里设一次，所有 `python -m eval.*` 入口都覆盖到，
不必再逐条命令记得加 PYTHONIOENCODING。
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        # 被重定向成非文本流或已关闭时忽略——不值得为日志编码让脚本起不来
        pass
