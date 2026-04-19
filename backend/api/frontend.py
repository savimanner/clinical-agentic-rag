from __future__ import annotations

from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse, Response


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


def frontend_dist_dir() -> Path:
    return FRONTEND_DIST_DIR


def frontend_index_path() -> Path:
    return frontend_dist_dir() / "index.html"


def serve_frontend_index() -> Response:
    index_path = frontend_index_path()
    if index_path.exists():
        return FileResponse(index_path)

    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Local Agentic RAG</title>
          </head>
          <body style="margin:0;padding:40px;background:#f5efe4;color:#221f1a;font-family:'Avenir Next','Segoe UI',sans-serif;">
            <h1 style="margin:0 0 12px;font-size:2.5rem;font-family:'Iowan Old Style','Palatino Linotype',serif;">Frontend build not found</h1>
            <p style="max-width:52rem;line-height:1.5;">Run <code>npm install</code> and <code>npm run build</code> inside <code>frontend/</code>, or use <code>npm run dev</code> during local development.</p>
          </body>
        </html>
        """.strip()
    )
