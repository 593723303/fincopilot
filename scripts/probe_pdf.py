"""年报 PDF 结构探查。

在写解析器之前先看清真实文档长什么样——中文年报的表格、标题、量纲
分布与想象往往不同，凭假设写解析器是这类任务最容易返工的地方。

    python -m scripts.probe_pdf                     探查 data/raw 下全部 PDF
    python -m scripts.probe_pdf --pages 30          只看前 30 页
    python -m scripts.probe_pdf --dump-page 45      打印指定页的结构细节
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 中文年报的量纲表述，决定数值能否被正确读取
UNIT_PAT = re.compile(r"单位[：: ]*\s*(人民币)?\s*(元|千元|万元|亿元|百万元)")
# 常见的一级标题形态
SECTION_PAT = re.compile(r"^第[一二三四五六七八九十]+节\s*\S+")
SUBSEC_PAT = re.compile(r"^[一二三四五六七八九十]+[、．.]\s*\S+")


def analyze(path: Path, max_pages: int | None) -> dict:
    doc = pymupdf.open(path)
    total = doc.page_count
    limit = min(total, max_pages) if max_pages else total

    stat = {
        "path": path,
        "total_pages": total,
        "scanned_pages": limit,
        "empty_text_pages": 0,
        "table_pages": 0,
        "table_count": 0,
        "tables_with_unit": 0,
        "unit_mentions": Counter(),
        "sections": [],
        "font_sizes": Counter(),
        "cross_page_suspects": 0,
        "wide_tables": 0,
        "max_table_rows": 0,
    }

    prev_table_at_bottom = False
    for i in range(limit):
        page = doc[i]
        text = page.get_text("text") or ""
        if not text.strip():
            stat["empty_text_pages"] += 1

        for m in UNIT_PAT.finditer(text):
            stat["unit_mentions"][m.group(2)] += 1

        for line in text.splitlines():
            line = line.strip()
            if SECTION_PAT.match(line) and len(stat["sections"]) < 40:
                stat["sections"].append((i + 1, line[:40]))

        # 字号分布用于判断能否靠字号识别标题层级
        for block in page.get_text("dict").get("blocks", []):
            for ln in block.get("lines", []):
                for span in ln.get("spans", []):
                    if span.get("text", "").strip():
                        stat["font_sizes"][round(span.get("size", 0), 1)] += 1

        try:
            tables = page.find_tables()
            tlist = list(tables)
        except Exception:
            tlist = []

        if tlist:
            stat["table_pages"] += 1
            stat["table_count"] += len(tlist)
            page_h = page.rect.height
            for t in tlist:
                rows = len(t.extract())
                cols = len(t.extract()[0]) if rows else 0
                stat["max_table_rows"] = max(stat["max_table_rows"], rows)
                if cols > 6:
                    stat["wide_tables"] += 1
                # 表格上方 120pt 内是否出现"单位：..."
                clip = pymupdf.Rect(0, max(0, t.bbox[1] - 120), page.rect.width, t.bbox[1])
                above = page.get_text("text", clip=clip)
                if UNIT_PAT.search(above or ""):
                    stat["tables_with_unit"] += 1
                # 表格贴近页面底部 → 可能跨页延续
                if t.bbox[3] > page_h - 80:
                    prev_table_at_bottom = True
                    continue
            # 上一页底部有表、本页顶部也有表 → 疑似跨页
            if prev_table_at_bottom and tlist and tlist[0].bbox[1] < 150:
                stat["cross_page_suspects"] += 1
            if not any(t.bbox[3] > page_h - 80 for t in tlist):
                prev_table_at_bottom = False
        else:
            prev_table_at_bottom = False

    doc.close()
    return stat


def report(stat: dict) -> None:
    p = stat["path"]
    print(f"\n{'=' * 70}\n  {p.name}\n{'=' * 70}")
    print(f"  页数            {stat['total_pages']}（本次分析 {stat['scanned_pages']} 页）")
    empty = stat["empty_text_pages"]
    verdict = "文本层完好" if empty == 0 else f"有 {empty} 页无文本层，可能是扫描件或图片页"
    print(f"  文本层          {verdict}")
    print(f"  含表格页        {stat['table_pages']} 页，共检出 {stat['table_count']} 个表格")
    print(f"  最大表格行数    {stat['max_table_rows']}")
    print(f"  宽表（>6 列）   {stat['wide_tables']} 个")
    print(f"  疑似跨页表格    {stat['cross_page_suspects']} 处")

    tw = stat["tables_with_unit"]
    tc = stat["table_count"]
    ratio = f"{tw / tc * 100:.0f}%" if tc else "—"
    print(f"  表格上方有「单位：」 {tw}/{tc}（{ratio}）")

    if stat["unit_mentions"]:
        units = "，".join(f"{u}×{n}" for u, n in stat["unit_mentions"].most_common())
        print(f"  量纲分布        {units}")
    else:
        print("  量纲分布        未检出「单位：」声明")

    sizes = stat["font_sizes"].most_common(6)
    print(f"  主要字号        {'，'.join(f'{s}pt×{n}' for s, n in sizes)}")

    print(f"  识别到的章节    {len(stat['sections'])} 个")
    for pg, title in stat["sections"][:8]:
        print(f"      p{pg:<4} {title}")


def dump_page(path: Path, pageno: int) -> None:
    doc = pymupdf.open(path)
    page = doc[pageno - 1]
    print(f"\n{'=' * 70}\n  {path.name}  第 {pageno} 页结构\n{'=' * 70}")
    tables = list(page.find_tables())
    print(f"  检出 {len(tables)} 个表格")
    for i, t in enumerate(tables):
        data = t.extract()
        print(f"\n  —— 表格 {i + 1}：{len(data)} 行 × {len(data[0]) if data else 0} 列 ——")
        for row in data[:6]:
            cells = [(c or "").replace("\n", " ")[:16] for c in row]
            print("    | " + " | ".join(f"{c:<16}" for c in cells))
        if len(data) > 6:
            print(f"    ...（共 {len(data)} 行）")
    text = page.get_text("text")
    print("\n  —— 纯文本前 400 字 ——")
    print("    " + text[:400].replace("\n", "\n    "))
    doc.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=None, help="只分析前 N 页")
    parser.add_argument("--dump-page", type=int, default=None, help="打印指定页的结构")
    parser.add_argument("--file", type=str, default=None, help="指定文件名片段")
    args = parser.parse_args()

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if args.file:
        pdfs = [p for p in pdfs if args.file in p.name]
    if not pdfs:
        print(f"{RAW_DIR} 下没有 PDF，先执行 python -m scripts.fetch_reports")
        return 1

    if args.dump_page:
        dump_page(pdfs[0], args.dump_page)
        return 0

    for p in pdfs:
        report(analyze(p, args.pages))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
