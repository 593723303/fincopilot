"""文档管理接口。

上传后立即返回，解析入库交给 Worker 异步处理（架构 §4.2）：
一份年报数百页，解析是分钟级工作，同步处理会把 HTTP 连接挂死。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

from arq.jobs import Job
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.config.experiment import load_experiment
from app.config.settings import PROJECT_ROOT
from app.rag.indexer import infer_meta
from app.schemas.document import DocumentOut, IngestAccepted, JobStatus, ReindexRequest
from app.store.milvus_schema import collection_names
from app.store.models import Document
from app.store.pg import session_factory
from app.store.queue import queue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAX_UPLOAD_MB = 80


def sanitize_filename(raw: str | None) -> str:
    """把客户端提供的文件名收敛为安全的基础名。

    上传的 filename 完全由客户端控制，直接拼进路径就是路径穿越漏洞：
    传 ../../x.pdf 即可写到项目目录之外。
    PurePosixPath 与 PureWindowsPath 各切一遍，防止用另一种分隔符绕过。
    """
    name = PureWindowsPath(PurePosixPath(raw or "").name).name
    if not name or name in (".", "..") or name.startswith("."):
        raise ValueError("文件名无效")
    return name


@router.post("", response_model=IngestAccepted, status_code=202)
async def upload(
    file: UploadFile = File(..., description="年报 PDF"),
    exp_id: str | None = Query(None, description="实验配置 ID"),
    max_pages: int | None = Query(None, ge=1, description="只处理前 N 页"),
    force: bool = Query(False, description="内容未变也强制重建"),
) -> IngestAccepted:
    try:
        safe_name = sanitize_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 文件")

    # 文件名承载公司代码与年度，推断失败就不该入库——
    # 错误的 doc_key 会污染检索时的标量过滤
    try:
        infer_meta(Path(safe_name))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if exp_id:
        try:
            load_experiment(exp_id)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / safe_name
    try:
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, length=1 << 20)
    finally:
        await file.close()

    size_mb = target.stat().st_size / 1024 / 1024
    if size_mb > MAX_UPLOAD_MB:
        target.unlink(missing_ok=True)
        raise HTTPException(413, f"文件 {size_mb:.1f}MB 超过上限 {MAX_UPLOAD_MB}MB")

    job = await queue().enqueue_job(
        "ingest_document",
        str(target),
        exp_id=exp_id,
        max_pages=max_pages,
        force=force,
    )
    if job is None:
        raise HTTPException(500, "任务投递失败")
    logger.info("已接收 %s（%.1fMB），job=%s", safe_name, size_mb, job.job_id)
    return IngestAccepted(job_id=job.job_id, file_name=safe_name)


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    status: str | None = Query(None, description="按入库状态过滤"),
    company: str | None = Query(None, description="按股票代码过滤"),
    limit: int = Query(50, ge=1, le=200),
) -> list[DocumentOut]:
    stmt = select(Document).order_by(Document.updated_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Document.status == status)
    if company:
        stmt = stmt.where(Document.company_code == company)
    async with session_factory()() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [DocumentOut.model_validate(r) for r in rows]


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: int) -> DocumentOut:
    async with session_factory()() as session:
        doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, f"文档 {doc_id} 不存在")
    return DocumentOut.model_validate(doc)


@router.post("/{doc_id}/reindex", response_model=IngestAccepted, status_code=202)
async def reindex(doc_id: int, req: ReindexRequest) -> IngestAccepted:
    """按指定实验配置重建索引。

    不同策略的块按 strategy 字段隔离并存，这是 M4 消融实验的前提（ADR-010）。
    """
    async with session_factory()() as session:
        doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, f"文档 {doc_id} 不存在")

    path = Path(doc.file_path)
    if not path.exists():
        raise HTTPException(409, f"源文件已丢失：{doc.file_path}")

    if req.exp_id:
        try:
            load_experiment(req.exp_id)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc

    job = await queue().enqueue_job(
        "ingest_document",
        str(path),
        exp_id=req.exp_id,
        max_pages=req.max_pages,
        force=True,
    )
    if job is None:
        raise HTTPException(500, "任务投递失败")
    return IngestAccepted(job_id=job.job_id, file_name=path.name, message="重建任务已提交")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    job = Job(job_id, redis=queue(), _queue_name=queue().default_queue_name)
    status = await job.status()
    result = None
    try:
        result = await job.result(timeout=0.1)
    except Exception:
        # 任务未完成或已过期，状态字段本身已说明情况
        pass
    return JobStatus(job_id=job_id, status=str(status), result=result)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: int) -> None:
    """删除文档及其全部块。

    PG 侧靠外键级联，Milvus 没有外键，必须显式按 doc_key 清理，
    否则会留下检索得到却无法回溯原文的孤立向量。
    """
    from app.store.milvus import milvus

    async with session_factory()() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, f"文档 {doc_id} 不存在")
        doc_key = doc.doc_key
        await session.delete(doc)
        await session.commit()

    name, _ = collection_names()
    try:
        client = milvus()
        client.delete(name, filter=f'doc_key == "{doc_key}"')
        client.flush(name)
    except Exception as exc:
        # PG 已删除，此处失败会留下孤立向量，必须记录以便人工清理
        logger.error("文档 %s 的向量清理失败，存在孤立向量：%s", doc_key, exc)
        raise HTTPException(500, f"元数据已删除，但向量清理失败：{exc}") from exc
