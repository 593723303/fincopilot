from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    doc_key: str
    company_code: str
    company_name: str
    report_year: int
    report_type: str
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    index_cost: Decimal | None = None
    error_code: str | None = None
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IngestAccepted(BaseModel):
    """上传后立即返回，解析在后台进行。"""

    job_id: str
    file_name: str
    message: str = "已接收，解析在后台进行，可通过 /api/v1/documents 查看状态"


class ReindexRequest(BaseModel):
    exp_id: str | None = Field(None, description="实验配置 ID，决定分块策略；留空用当前默认")
    max_pages: int | None = Field(None, ge=1, description="只处理前 N 页，用于控制成本")


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
