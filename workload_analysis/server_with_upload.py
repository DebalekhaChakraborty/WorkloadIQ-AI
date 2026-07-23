# workload_analysis/server_with_upload.py
import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from pydantic import BaseModel, Field

# ✅ Load .env from THIS folder (workload_analysis/.env)
try:
    from dotenv import load_dotenv  # type: ignore

    HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(HERE, ".env"))
    print("[server] Loaded .env from:", os.path.join(HERE, ".env"))
except Exception as e:
    print("[server] dotenv not loaded (ok if env already set):", e)

from .tools.batch_evaluate_csv import batch_evaluate_csv
from .tools.heavy_hitter_analysis import heavy_hitter_analyze

# --------------------------------------------------------------------------------------
# 1) Build the ADK FastAPI app
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.getenv("ADK_AGENTS_DIR") or os.path.abspath(os.path.join(HERE, ".."))
ROOT_DIR = Path(HERE).parent
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
SAMPLE_DATASET = ROOT_DIR / "sample_data" / "service_desk_tickets.csv"

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:8000,http://127.0.0.1:8000,"
            r"regex:^http://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+):(?:3000|5173)$"
        ),
    ).split(",")
    if origin.strip()
]

app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    allow_origins=cors_origins,
    web=False,
)

# --------------------------------------------------------------------------------------
# 2) Output directories
# --------------------------------------------------------------------------------------
TICKET_QA_OUTPUT_DIR = os.getenv("TICKET_QA_OUTPUT_DIR")

out = (TICKET_QA_OUTPUT_DIR or "").strip()
if not out:
    out = "./outputs"
    TICKET_QA_OUTPUT_DIR = out

UPLOAD_DIR = os.path.join(TICKET_QA_OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

JOBS_DIR = os.path.join(TICKET_QA_OUTPUT_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

print("[server] FILE =", __file__)
print("[server] CWD =", os.getcwd())
print("[server] AGENTS_DIR =", AGENTS_DIR)
print("[server] TICKET_QA_OUTPUT_DIR =", TICKET_QA_OUTPUT_DIR)
print("[server] UPLOAD_DIR =", UPLOAD_DIR)
print("[server] JOBS_DIR =", JOBS_DIR)


def _read_json_file(path: str) -> dict:
    try:
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _safe_job_id(job_id: str) -> str:
    job_id = (job_id or "").strip()
    job_id = os.path.basename(job_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", job_id):
        return "invalid_job_id"
    return job_id


def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, _safe_job_id(job_id))


def _job_progress_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "progress.json")


def _job_results_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "results.json")


def _job_metadata_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "request.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"wla_{stamp}_{uuid.uuid4().hex[:8]}"


def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _extract_result_payload(payload: dict) -> dict:
    """
    Normalize shapes into 'result' dict.
    Supports:
    - {ts, message, data: {result: {...}}}
    - {result: {...}}
    - direct {...}
    """
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        return data.get("result") or {}
    if isinstance(payload.get("result"), dict):
        return payload.get("result") or {}
    return payload


def _progress_percent(progress: Dict[str, Any], has_result: bool, failed: bool) -> int:
    if has_result:
        return 100
    if failed:
        return 100

    message = str(progress.get("message") or "").lower()
    data = progress.get("data") if isinstance(progress.get("data"), dict) else {}
    stage = str(data.get("stage") or "").lower()
    rows = data.get("rows") or data.get("done")
    total = data.get("total")
    if isinstance(rows, (int, float)) and isinstance(total, (int, float)) and total:
        base = 30 if "normalize" in stage else 42
        span = 22 if "normalize" in stage else 43
        return min(92, int(base + span * (float(rows) / float(total))))

    stage_markers = [
        (("queued",), 4),
        (("start", "reading"), 12),
        (("loaded", "normalize"), 30),
        (("aggregation", "deterministic"), 58),
        (("llm",), 72),
        (("chart", "writing"), 88),
        (("done", "complete"), 100),
        (("error", "failed"), 100),
    ]
    haystack = f"{stage} {message}"
    for markers, value in stage_markers:
        if any(marker in haystack for marker in markers):
            return value
    return 8


def _assessment_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    jid = _safe_job_id(job_id)
    job_dir = os.path.join(JOBS_DIR, jid)
    if not os.path.isdir(job_dir):
        return None
    metadata = _read_json_file(_job_metadata_path(jid))
    progress = _read_json_file(_job_progress_path(jid))
    raw_results = _read_json_file(_job_results_path(jid))

    if not metadata and not progress and not raw_results:
        return None

    result = _extract_result_payload(raw_results) if raw_results else {}
    error = result.get("error") if isinstance(result, dict) else None
    requested_status = str(metadata.get("status") or "queued")
    if error or requested_status == "failed":
        status = "failed"
    elif raw_results:
        status = "completed"
    elif progress:
        status = "running"
    else:
        status = requested_status

    return {
        "job_id": jid,
        "mode": metadata.get("mode") or result.get("kind") or "workload",
        "original_name": metadata.get("original_name") or metadata.get("filename") or "Assessment",
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at") or progress.get("ts"),
        "status": status,
        "error": error,
        "progress": {
            "message": progress.get("message") or ("Assessment complete" if status == "completed" else "Queued"),
            "stage": ((progress.get("data") or {}).get("stage") if isinstance(progress.get("data"), dict) else None),
            "percent": _progress_percent(progress, bool(raw_results), status == "failed"),
            "rows": ((progress.get("data") or {}).get("rows") if isinstance(progress.get("data"), dict) else None),
            "total": ((progress.get("data") or {}).get("total") if isinstance(progress.get("data"), dict) else None),
        },
        "result": result if raw_results else None,
    }


def _find_file_recursive(root_dir: str, filename: str, max_hits: int = 5) -> List[str]:
    """
    Recursively search for an exact filename under root_dir.
    Returns up to max_hits matches.
    """
    hits: List[str] = []
    if not root_dir or not os.path.isdir(root_dir):
        return hits

    for r, _, files in os.walk(root_dir):
        if filename in files:
            hits.append(os.path.join(r, filename))
            if len(hits) >= max_hits:
                break
    return hits


def _try_resolve_csv_via_results_json(filename: str) -> Optional[str]:
    """
    Walk jobs/**/results.json and try to find an output_csv_path that matches the filename.
    This helps when the frontend only knows the filename but not job_id.
    """
    if not os.path.isdir(JOBS_DIR):
        return None

    for r, _, files in os.walk(JOBS_DIR):
        if "results.json" not in files:
            continue
        results_path = os.path.join(r, "results.json")
        payload = _read_json_file(results_path)
        if not payload:
            continue
        result = _extract_result_payload(payload)
        out_csv = result.get("output_csv_path")
        if isinstance(out_csv, str) and os.path.basename(out_csv) == filename and os.path.isfile(out_csv):
            return out_csv

    return None


# --------------------------------------------------------------------------------------
# 3) Upload endpoint
# --------------------------------------------------------------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".csv"
        safe = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_DIR, safe)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "ok": True,
            "originalName": file.filename,
            "filename": safe,
            "filePath": file_path,
            "size": len(content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------------
# 4) Download endpoint (job-safe + improved legacy)
# --------------------------------------------------------------------------------------
@app.get("/api/download")
async def download(job_id: str = "", filename: str = ""):
    """
    Preferred: /api/download?job_id=<job_id>
      - Reads jobs/<job_id>/results.json
      - Serves output_csv_path from the job result payload

    Legacy: /api/download?filename=<name>
      - Also searches jobs/**/<filename> recursively
      - Also tries to resolve by reading jobs/**/results.json -> output_csv_path
    """
    try:
        # --- Job-safe path (BEST) ---
        if job_id:
            jid = _safe_job_id(job_id)
            results_path = _job_results_path(jid)
            payload = _read_json_file(results_path)
            if not payload:
                return JSONResponse(
                    status_code=404,
                    content={"ok": False, "error": "Job results not found", "job_id": jid},
                )

            result = _extract_result_payload(payload)
            out_csv = result.get("output_csv_path")
            if not isinstance(out_csv, str) or not out_csv.strip():
                return JSONResponse(
                    status_code=404,
                    content={
                        "ok": False,
                        "error": "Job has no output_csv_path",
                        "job_id": jid,
                        "results_path": results_path,
                    },
                )

            out_csv = out_csv.strip()
            if not os.path.isfile(out_csv):
                # As a fallback: maybe CSV exists in the job folder but path stored is stale
                wanted_name = os.path.basename(out_csv)
                hits = _find_file_recursive(_job_dir(jid), wanted_name, max_hits=3)
                if hits:
                    out_csv = hits[0]
                else:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "ok": False,
                            "error": "Output CSV missing on disk",
                            "job_id": jid,
                            "output_csv_path": out_csv,
                        },
                    )

            dl_name = os.path.basename(out_csv)
            return FileResponse(out_csv, media_type="text/csv", filename=dl_name)

        # --- Legacy filename search (IMPROVED) ---
        if not filename:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Missing job_id or filename"})

        filename = os.path.basename(filename)  # security
        cwd = os.getcwd()
        ticket_dir = TICKET_QA_OUTPUT_DIR

        # 1) Fast path: common legacy locations
        candidates = [
            os.path.join(ticket_dir, filename),
            os.path.join(ticket_dir, "reports", filename),
            os.path.join(ticket_dir, "tests", filename),
            os.path.join(ticket_dir, "outputs", filename),
            os.path.join(cwd, filename),
            os.path.join(cwd, "outputs", filename),
            os.path.join(cwd, "ticket_qa_agent", "tests", filename),
        ]
        file_path = next((p for p in candidates if os.path.isfile(p)), None)

        # 2) NEW: search inside jobs/** recursively
        if not file_path:
            hits = _find_file_recursive(JOBS_DIR, filename, max_hits=3)
            if hits:
                file_path = hits[0]

        # 3) NEW: resolve via jobs/**/results.json -> output_csv_path
        if not file_path:
            resolved = _try_resolve_csv_via_results_json(filename)
            if resolved:
                file_path = resolved

        if not file_path:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "File not found",
                    "filename": filename,
                    "debug": {
                        "cwd": cwd,
                        "TICKET_QA_OUTPUT_DIR": ticket_dir,
                        "JOBS_DIR": JOBS_DIR,
                        "searched": candidates + [f"{JOBS_DIR}/**/{filename}", f"{JOBS_DIR}/**/results.json -> output_csv_path"],
                    },
                },
            )

        return FileResponse(file_path, media_type="text/csv", filename=filename)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --------------------------------------------------------------------------------------
# 5) Job-scoped recovery endpoints (NO /latest)
# --------------------------------------------------------------------------------------
@app.get("/api/progress/{job_id}")
async def progress_by_job(job_id: str):
    payload = _read_json_file(_job_progress_path(job_id))
    if not payload:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "No progress for job yet.", "job_id": job_id},
        )
    return JSONResponse(content={"ok": True, "job_id": job_id, "progress": payload})


@app.get("/api/results/{job_id}")
async def results_by_job(job_id: str):
    payload = _read_json_file(_job_results_path(job_id))
    if not payload:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "No results for job yet.", "job_id": job_id},
        )

    result = _extract_result_payload(payload)

    return JSONResponse(
        content={
            "ok": True,
            "job_id": job_id,
            "meta": {"ts": payload.get("ts"), "message": payload.get("message")},
            "results": result,
        }
    )


# --------------------------------------------------------------------------------------
# 6) Native SaaS assessment API
# --------------------------------------------------------------------------------------
def _run_assessment(job_id: str, file_path: str, filename: str, mode: str) -> None:
    metadata_path = _job_metadata_path(job_id)
    metadata = _read_json_file(metadata_path)
    metadata.update({"status": "running", "updated_at": _utc_now()})
    _write_json_file(metadata_path, metadata)

    context = SimpleNamespace(
        state={
            "wla_job_id": job_id,
            "uploaded_file_path": file_path,
            "uploads": {"filePath": file_path},
        }
    )

    try:
        if mode == "quality":
            batch_evaluate_csv(
                file_path=file_path,
                filename=filename,
                tool_context=context,
            )
        else:
            heavy_hitter_analyze(tool_context=context, filename=filename)

        result = _extract_result_payload(_read_json_file(_job_results_path(job_id)))
        metadata["status"] = "failed" if result.get("error") else "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        if not os.path.isfile(_job_results_path(job_id)):
            _write_json_file(
                _job_results_path(job_id),
                {"ok": False, "job_id": job_id, "kind": mode, "error": str(exc)},
            )
    finally:
        metadata["updated_at"] = _utc_now()
        _write_json_file(metadata_path, metadata)


def _queue_assessment(
    background_tasks: BackgroundTasks,
    *,
    content: bytes,
    filename: str,
    mode: str,
) -> Dict[str, Any]:
    if mode not in {"workload", "quality"}:
        raise HTTPException(status_code=422, detail="Mode must be 'workload' or 'quality'.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are supported.")
    if not content:
        raise HTTPException(status_code=422, detail="The CSV file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The CSV file exceeds the 25 MB limit.")

    job_id = _new_job_id()
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename)) or "tickets.csv"
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_filename}")
    with open(upload_path, "wb") as handle:
        handle.write(content)

    metadata = {
        "job_id": job_id,
        "mode": mode,
        "original_name": filename,
        "filename": safe_filename,
        "upload_path": upload_path,
        "size": len(content),
        "status": "queued",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_json_file(_job_metadata_path(job_id), metadata)
    _write_json_file(
        _job_progress_path(job_id),
        {
            "ts": _utc_now(),
            "job_id": job_id,
            "message": "Assessment queued",
            "data": {"stage": "queued"},
        },
    )
    background_tasks.add_task(_run_assessment, job_id, upload_path, safe_filename, mode)
    return _assessment_snapshot(job_id) or metadata


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "service": "workload-intelligence",
        "frontend_built": FRONTEND_DIST.is_dir(),
    }


@app.get("/api/assessments")
async def list_assessments(limit: int = 20):
    items: List[Dict[str, Any]] = []
    if os.path.isdir(JOBS_DIR):
        job_ids = sorted(
            (
                entry
                for entry in os.listdir(JOBS_DIR)
                if os.path.isdir(os.path.join(JOBS_DIR, entry))
            ),
            reverse=True,
        )
        for job_id in job_ids[: max(1, min(limit, 100))]:
            snapshot = _assessment_snapshot(job_id)
            if snapshot and (
                snapshot.get("created_at")
                or os.path.isfile(_job_metadata_path(job_id))
            ):
                summary = dict(snapshot)
                result = summary.pop("result", None)
                if isinstance(result, dict):
                    summary["result_summary"] = {
                        "total_tickets": result.get("total_tickets")
                        or (result.get("stats") or {}).get("total_tickets"),
                        "avg_score": (result.get("stats") or {}).get("avg_score"),
                    }
                items.append(summary)
    return {"items": items}


@app.post("/api/assessments", status_code=202)
async def create_assessment(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("workload"),
):
    content = await file.read()
    return _queue_assessment(
        background_tasks,
        content=content,
        filename=file.filename or "tickets.csv",
        mode=mode,
    )


@app.post("/api/assessments/sample", status_code=202)
async def create_sample_assessment(
    background_tasks: BackgroundTasks,
    mode: str = Form("workload"),
):
    if not SAMPLE_DATASET.is_file():
        raise HTTPException(status_code=404, detail="Sample dataset is not available.")
    return _queue_assessment(
        background_tasks,
        content=SAMPLE_DATASET.read_bytes(),
        filename=SAMPLE_DATASET.name,
        mode=mode,
    )


@app.get("/api/assessments/{job_id}")
async def get_assessment(job_id: str):
    snapshot = _assessment_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return snapshot


class CopilotRequest(BaseModel):
    job_id: str
    query: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    user_id: Optional[str] = Field(default=None, max_length=128)


def _compact_result_for_chat(result: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(result)
    compact.pop("charts", None)
    compact.pop("output_csv_path", None)
    llm_insights = compact.get("llm_insights")
    if isinstance(llm_insights, dict):
        llm_insights = dict(llm_insights)
        llm_insights.pop("raw_llm_output", None)
        compact["llm_insights"] = llm_insights
    return compact


def _fallback_chat_answer(result: Dict[str, Any], query: str) -> str:
    query_lower = query.lower()
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    categories = result.get("top_categories") if isinstance(result.get("top_categories"), list) else []
    groups = result.get("top_assignment_groups") if isinstance(result.get("top_assignment_groups"), list) else []
    parsed = (((result.get("llm_insights") or {}).get("parsed_json")) or {})
    hitters = parsed.get("top_heavy_hitters") if isinstance(parsed, dict) else []

    if stats:
        total = int(stats.get("total_tickets") or 0)
        evaluated = int(stats.get("evaluated_tickets") or 0)
        pass_count = int(stats.get("pass_count") or 0)
        fail_count = int(stats.get("fail_count") or 0)
        average = stats.get("avg_score") or 0
        if "improve" in query_lower or "action" in query_lower:
            opportunities = ((result.get("quality_insights") or {}).get("top_opportunities")) or []
            if opportunities:
                top = opportunities[:3]
                lines = [
                    f"{item.get('label')}: {item.get('issue_count', 0)} flagged tickets."
                    for item in top
                ]
                return "Prioritize these quality gaps:\n" + "\n".join(lines)
        return (
            f"{evaluated} of {total} tickets were scored, with an average of {average}/100. "
            f"{pass_count} passed and {fail_count} failed. "
            "Open the Quality view to compare section scores and the largest control gaps."
        )

    if ("action" in query_lower or "recommend" in query_lower) and hitters:
        actions: List[str] = []
        for hitter in hitters[:3]:
            recommended = hitter.get("recommended_actions") or []
            if isinstance(recommended, str):
                recommended = [recommended]
            if recommended:
                actions.append(f"{hitter.get('theme')}: {recommended[0]}")
        if actions:
            return "Recommended priorities:\n" + "\n".join(actions)

    total = result.get("total_tickets") or 0
    top_category = categories[0] if categories else {}
    top_group = groups[0] if groups else {}
    category_name = top_category.get("bucket") or "No category detected"
    category_share = round(float(top_category.get("pct") or 0) * 100, 1)
    group_name = top_group.get("bucket") or "No assignment group detected"
    return (
        f"The assessment covers {total} tickets. {category_name} is the largest category "
        f"at {category_share}% of volume, and {group_name} carries the highest assignment-group load. "
        "Use the Drivers view to inspect concentration and the recommended response."
    )


def _answer_copilot(result: Dict[str, Any], query: str) -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
    if not project or not location:
        return _fallback_chat_answer(result, query)

    try:
        import vertexai
        from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

        vertexai.init(project=project, location=location)
        model_name = os.getenv("HEAVY_HITTER_LLM_MODEL", "gemini-2.5-flash")
        model = GenerativeModel(model_name)
        context_json = json.dumps(_compact_result_for_chat(result), ensure_ascii=False)[:30000]
        response = model.generate_content(
            [
                (
                    "You are an IT service-management assessment analyst. Answer only from the "
                    "provided assessment. Be concise, quantify claims, and state when the data "
                    "does not support an answer. Do not claim to rerun or change the assessment."
                ),
                f"\nASSESSMENT:\n{context_json}\n\nQUESTION:\n{query}",
            ],
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )
        answer = (response.text or "").strip()
        return answer or _fallback_chat_answer(result, query)
    except Exception:
        return _fallback_chat_answer(result, query)


def _safe_chat_identity(value: Optional[str], prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.@-]+", "-", (value or "").strip())
    cleaned = cleaned.strip(".-")[:96]
    return cleaned or f"{prefix}-{uuid.uuid4().hex[:16]}"


async def _ask_adk_copilot(
    *,
    result: Dict[str, Any],
    query: str,
    job_id: str,
    session_id: str,
    user_id: str,
) -> Optional[str]:
    if os.getenv("COPILOT_USE_ADK", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if not os.getenv("GOOGLE_CLOUD_PROJECT", "").strip():
        return None

    try:
        import httpx

        port = int(os.getenv("PORT", "8000"))
        base_url = os.getenv("ADK_INTERNAL_BASE_URL", f"http://127.0.0.1:{port}").rstrip("/")
        app_name = "workload_analysis"
        session_url = (
            f"{base_url}/apps/{app_name}/users/{user_id}/sessions/{session_id}"
        )
        context_json = json.dumps(
            _compact_result_for_chat(result),
            ensure_ascii=False,
            separators=(",", ":"),
        )[:30000]
        grounded_message = (
            "Answer a follow-up question about the completed assessment below. "
            "Do not run an analysis tool and do not invent facts. Use only the supplied "
            "assessment result, quantify claims where possible, and say when the result "
            "does not support an answer.\n\n"
            f"[assessment_job_id]: {job_id}\n"
            f"[assessment_result]: {context_json}\n\n"
            f"[question]: {query}"
        )
        state = {
            "assessment_job_id": job_id,
            "channel": "workloadiq-web",
            "metadata": {
                "assessment_job_id": job_id,
                "channel": "workloadiq-web",
            },
        }
        payload = {
            "appName": app_name,
            "userId": user_id,
            "sessionId": session_id,
            "newMessage": {
                "role": "user",
                "parts": [{"text": grounded_message}],
            },
            "streaming": True,
            "state_delta": {"session": {"state": state}, **state},
            "stateDelta": {"session": {"state": state}, **state},
        }

        timeout = httpx.Timeout(
            connect=float(os.getenv("ADK_CONNECT_TIMEOUT_SECONDS", "5")),
            read=float(os.getenv("ADK_CHAT_TIMEOUT_SECONDS", "120")),
            write=20.0,
            pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            session_response = await client.post(session_url, json={})
            if session_response.status_code >= 400 and "already exists" not in session_response.text.lower():
                return None

            texts: List[str] = []
            async with client.stream("POST", f"{base_url}/run_sse", json=payload) as response:
                if response.status_code >= 400:
                    return None
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = event.get("content") if isinstance(event, dict) else None
                    if not isinstance(content, dict) or content.get("role") == "user":
                        continue
                    for part in content.get("parts") or []:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                texts.append(text.strip())

        return max(texts, key=len) if texts else None
    except Exception as exc:
        print(f"[copilot] ADK session fallback: {exc}")
        return None


@app.post("/api/copilot/chat")
async def copilot_chat(request: CopilotRequest):
    snapshot = _assessment_snapshot(request.job_id)
    if not snapshot or not isinstance(snapshot.get("result"), dict):
        raise HTTPException(status_code=404, detail="Completed assessment not found.")

    session_id = _safe_chat_identity(request.session_id, "session")
    user_id = _safe_chat_identity(request.user_id, "workloadiq-user")
    answer = await _ask_adk_copilot(
        result=snapshot["result"],
        query=request.query,
        job_id=request.job_id,
        session_id=session_id,
        user_id=user_id,
    )
    source = "adk"
    if not answer:
        answer = await asyncio.to_thread(_answer_copilot, snapshot["result"], request.query)
        source = "grounded-fallback"
    return {
        "answer": answer,
        "job_id": request.job_id,
        "session_id": session_id,
        "grounded": True,
        "source": source,
    }


# Serve the compiled client from the same process in production.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


# --------------------------------------------------------------------------------------
# 7) Run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    keep_alive = int(os.getenv("SERVER_TIMEOUT_KEEP_ALIVE", "600"))
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=keep_alive)
