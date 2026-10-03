"""从巨潮资讯网下载 A 股年报 PDF。

    python -m scripts.fetch_reports                 下载默认样本
    python -m scripts.fetch_reports 600519 000858   指定股票代码

巨潮是证监会指定的信息披露平台，年报 PDF 为官方原件。
下载结果存入 data/raw/，该目录已在 .gitignore 中。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_BASE = "http://static.cninfo.com.cn/"

# 默认样本：覆盖不同行业，因为报表科目与表格结构差异显著，
# 单一行业会让分块策略过拟合（架构 §11.3）
DEFAULT_TARGETS = [
    ("600519", "贵州茅台", "sse"),   # 消费
    ("300750", "宁德时代", "szse"),  # 制造
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
}


async def query_annual_reports(
    client: httpx.AsyncClient, code: str, market: str, page_size: int = 30
) -> list[dict]:
    """查询某只股票的年报公告列表。"""
    payload = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "column": market,
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},",
        "searchkey": "",
        "secid": "",
        "category": "category_ndbg_szsh",  # 年度报告
        "trade": "",
        "seDate": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    resp = await client.post(QUERY_URL, data=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("announcements") or []


def pick_main_reports(announcements: list[dict], limit: int = 1) -> list[dict]:
    """筛掉摘要、英文版、更正公告，只保留年报正文。"""
    skip_words = ("摘要", "英文", "已取消", "更正", "补充", "问询")
    picked = []
    for ann in announcements:
        title = ann.get("announcementTitle", "")
        if any(w in title for w in skip_words):
            continue
        if "年度报告" not in title:
            continue
        picked.append(ann)
        if len(picked) >= limit:
            break
    return picked


async def download(client: httpx.AsyncClient, ann: dict, code: str, name: str) -> Path | None:
    path = ann.get("adjunctUrl")
    if not path:
        return None
    url = PDF_BASE + path
    title = ann.get("announcementTitle", "report").replace("/", "_").replace(":", "_")
    target = RAW_DIR / f"{code}_{name}_{title}.pdf"
    if target.exists():
        print(f"  [skip] 已存在 {target.name}")
        return target

    print(f"  下载 {title} ...")
    async with client.stream("GET", url, headers=HEADERS, timeout=120) as resp:
        resp.raise_for_status()
        with target.open("wb") as fh:
            async for chunk in resp.aiter_bytes(65536):
                fh.write(chunk)
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"  [ OK ] {target.name}  ({size_mb:.1f} MB)")
    return target


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="*", help="股票代码，留空则下载默认样本")
    parser.add_argument("--per-stock", type=int, default=1, help="每只股票下载几份年报")
    args = parser.parse_args()

    if args.codes:
        targets = [(c, c, "szse" if c.startswith(("0", "3")) else "sse") for c in args.codes]
    else:
        targets = DEFAULT_TARGETS

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载目录：{RAW_DIR}\n")

    downloaded: list[Path] = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for code, name, market in targets:
            print(f"── {code} {name} ──")
            try:
                anns = await query_annual_reports(client, code, market)
            except Exception as exc:
                print(f"  [FAIL] 查询失败：{type(exc).__name__}: {exc}")
                continue
            if not anns:
                print("  [FAIL] 未查到年报公告")
                continue
            for ann in pick_main_reports(anns, args.per_stock):
                try:
                    p = await download(client, ann, code, name)
                    if p:
                        downloaded.append(p)
                except Exception as exc:
                    print(f"  [FAIL] 下载失败：{type(exc).__name__}: {exc}")

    print(f"\n完成，共 {len(downloaded)} 份文件")
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
