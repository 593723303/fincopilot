"""从巨潮资讯网下载 A 股年报 PDF。

    python -m scripts.fetch_reports                 下载默认样本
    python -m scripts.fetch_reports 600519 000858   指定股票代码
    python -m scripts.fetch_reports --years 2       每只股票取最近 N 份年报

巨潮是证监会指定的信息披露平台，年报 PDF 为官方原件。
下载结果存入 data/raw/，该目录已在 .gitignore 中。

两个必须注意的点：
  1. 查询接口的 stock 参数必须是 "{code},{orgId}" 格式。只传 code 不会报错，
     但会被忽略，接口返回全市场最新公告——下载"成功"而内容完全无关。
  2. 因此下载后必须校验 PDF 首页是否包含目标公司代码。
     这个 bug 曾导致两份不同公司的年报实际是同一个无关文件。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_BASE = "http://static.cninfo.com.cn/"

# 默认样本：覆盖不同行业——报表科目与表格结构差异显著，
# 单一行业会让分块策略过拟合（架构 §11.3）
DEFAULT_CODES = ["600519", "300750"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
}

OK, NG = "  [ OK ]", "  [FAIL]"


async def resolve_stock(client: httpx.AsyncClient, code: str) -> dict | None:
    """查询股票的 orgId 与简称。orgId 是查询公告的必需参数。"""
    resp = await client.post(
        SEARCH_URL, data={"keyWord": code, "maxNum": "10"}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    for item in resp.json() or []:
        if item.get("code") == code and item.get("category") == "A股":
            return item
    return None


def market_column(org_id: str) -> str:
    """gssh* 为沪市，gssz* 为深市。"""
    return "sse" if org_id.startswith("gssh") else "szse"


async def query_annual_reports(client: httpx.AsyncClient, code: str, org_id: str) -> list[dict]:
    payload = {
        "pageNum": "1",
        "pageSize": "30",
        "column": market_column(org_id),
        "tabName": "fulltext",
        "plate": "",
        # 关键：必须是 "code,orgId"。只传 code 会被静默忽略
        "stock": f"{code},{org_id}",
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
    return resp.json().get("announcements") or []


def pick_main_reports(announcements: list[dict], limit: int) -> list[dict]:
    """筛掉摘要、英文版、更正公告，只保留年报正文。"""
    skip_words = ("摘要", "英文", "已取消", "更正", "补充", "问询", "说明")
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


def verify_pdf(path: Path, code: str, name: str) -> tuple[bool, str]:
    """校验 PDF 首页确实属于目标公司。

    下载成功不等于下载正确——接口参数失效时会返回无关文件，
    文件名却是我们自己拼的，不校验内容根本看不出来。
    """
    try:
        doc = pymupdf.open(path)
        head = ""
        for i in range(min(3, doc.page_count)):
            head += doc[i].get_text("text") or ""
        pages = doc.page_count
        doc.close()
    except Exception as exc:
        return False, f"PDF 无法打开：{exc}"

    if code not in head and name not in head:
        snippet = head.strip().replace("\n", " ")[:50]
        return False, f"内容不属于 {code} {name}，首页为：{snippet}"
    return True, f"{pages} 页"


async def download(client: httpx.AsyncClient, ann: dict, code: str, name: str) -> Path | None:
    url_path = ann.get("adjunctUrl")
    if not url_path:
        return None
    title = ann.get("announcementTitle", "report").replace("/", "_").replace(":", "_")
    target = RAW_DIR / f"{code}_{name}_{title}.pdf"

    if target.exists():
        ok, msg = verify_pdf(target, code, name)
        if ok:
            print(f"{OK} 已存在且校验通过 {target.name}（{msg}）")
            return target
        print(f"  已存在但校验失败，重新下载：{msg}")

    async with client.stream("GET", PDF_BASE + url_path, headers=HEADERS, timeout=180) as resp:
        resp.raise_for_status()
        with target.open("wb") as fh:
            async for chunk in resp.aiter_bytes(65536):
                fh.write(chunk)

    ok, msg = verify_pdf(target, code, name)
    size_mb = target.stat().st_size / 1024 / 1024
    if not ok:
        target.unlink(missing_ok=True)
        print(f"{NG} {title}：{msg}")
        return None
    print(f"{OK} {target.name}  ({size_mb:.1f} MB, {msg})")
    return target


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="*", help="股票代码，留空则下载默认样本")
    parser.add_argument("--years", type=int, default=1, help="每只股票下载最近几份年报")
    args = parser.parse_args()

    codes = args.codes or DEFAULT_CODES
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载目录：{RAW_DIR}\n")

    downloaded: list[Path] = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for code in codes:
            try:
                stock = await resolve_stock(client, code)
            except Exception as exc:
                print(f"── {code} ──\n{NG} 查询股票信息失败：{exc}")
                continue
            if not stock:
                print(f"── {code} ──\n{NG} 未找到该 A 股代码")
                continue

            name, org_id = stock["zwjc"], stock["orgId"]
            print(f"── {code} {name}（orgId={org_id}）──")

            try:
                anns = await query_annual_reports(client, code, org_id)
            except Exception as exc:
                print(f"{NG} 查询年报失败：{exc}")
                continue

            reports = pick_main_reports(anns, args.years)
            if not reports:
                print(f"{NG} 未查到年报正文（共 {len(anns)} 条公告）")
                continue

            for ann in reports:
                try:
                    p = await download(client, ann, code, name)
                    if p:
                        downloaded.append(p)
                except Exception as exc:
                    print(f"{NG} 下载失败：{type(exc).__name__}: {exc}")

    print(f"\n完成，共 {len(downloaded)} 份文件通过校验")
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
