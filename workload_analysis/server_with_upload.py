# workload_analysis/server_with_upload.py
import os
import uuid
import uvicorn
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
import json
from typing import Any, Dict, Optional, List

# ✅ Load .env from THIS folder (workload_analysis/.env)
try:
    from dotenv import load_dotenv  # type: ignore

    HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(HERE, ".env"))
    print("[server] Loaded .env from:", os.path.join(HERE, ".env"))
except Exception as e:
    print("[server] dotenv not loaded (ok if env already set):", e)

# --------------------------------------------------------------------------------------
# 1) Build the ADK FastAPI app
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.getenv("ADK_AGENTS_DIR") or os.path.abspath(os.path.join(HERE, ".."))

app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
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
    job_id = os.path.basename(job_id)  # security: disallow traversal
    return job_id


def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, _safe_job_id(job_id))


def _job_progress_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "progress.json")


def _job_results_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "results.json")


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
# 6) Run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    keep_alive = int(os.getenv("SERVER_TIMEOUT_KEEP_ALIVE", "600"))
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=keep_alive)
