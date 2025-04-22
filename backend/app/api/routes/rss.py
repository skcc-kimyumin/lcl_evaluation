from pathlib import Path

import feedparser
import urllib3
from api.deps import DatabaseDep, superadmin_roles
from core.config import get_setting
from database.database import get_db
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from log.logging import get_logging
from starlette.responses import HTMLResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"  # app/templates 폴더 지정
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))  # HTML 템플릿 파일 위치


@router.get("/get_rss_feed", response_class=HTMLResponse, summary="RSS feed 읽기")
async def get_rss_feed(request: Request, rss_url: str):

    # SSL 검증 비활성화
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    response = http.request("GET", rss_url)

    # `response.data`를 `feedparser.parse()`에 전달
    feed = feedparser.parse(response.data)

    return templates.TemplateResponse(
        "rss_template.html", {"request": request, "feed": feed}
    )
