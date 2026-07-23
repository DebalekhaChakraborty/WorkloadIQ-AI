# workload_analysis/tools/batch_evaluate_csv.py
import base64
import csv
import io
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from google.adk.tools import FunctionTool, ToolContext  # type: ignore

from .evaluate_ticket import (  # type: ignore
    evaluate_ticket,
    PARAM_SPECS,
    TICKET_QA_PASS_THRESHOLD,
)
from .normalize_columns import normalize_dataframe_with_report, normalize_ticket_dict  # type: ignore

# -----------------------------------------------------------------------------
# Output directory (same contract as Heavy Hitter)
# -----------------------------------------------------------------------------
TICKET_QA_OUTPUT_DIR = os.getenv("TICKET_QA_OUTPUT_DIR")


def _ensure_output_dir() -> str:
    out = (TICKET_QA_OUTPUT_DIR or "").strip()
    if not out:
        out = "./outputs"
    os.makedirs(out, exist_ok=True)
    return out


def _utc_compact_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _shortuuid() -> str:
    return uuid.uuid4().hex[:8]


def _new_job_id() -> str:
    # readable, sortable job id (match Heavy Hitter style)
    return f"wla_{_utc_compact_ts()}_{_shortuuid()}"


def _jobs_dir() -> str:
    d = os.path.join(_ensure_output_dir(), "jobs")
    os.makedirs(d, exist_ok=True)
    return d


def _job_dir(job_id: str) -> str:
    d = os.path.join(_jobs_dir(), job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _job_progress_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "progress.json")


def _job_results_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "results.json")


def _emit_progress(
    tool_context: Optional[ToolContext],
    job_id: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "message": message,
        "data": data or {},
    }

    # 1) ADK state (in-session)
    if tool_context is not None:
        try:
            state = tool_context.state or {}
            state["qa_progress"] = payload
            state["wla_job_id"] = job_id  # keep same key pattern used elsewhere
            tool_context.state = state
        except Exception:
            pass

    # 2) Persist job-scoped progress.json
    try:
        with open(_job_progress_path(job_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 3) Console
    try:
        print(f"[qa:{job_id}] {message}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers to build header and flatten per-ticket results
# ---------------------------------------------------------------------------
def _build_output_header() -> List[str]:
    header: List[str] = [
        "ticket_id",
        "overall_score",
        "verdict",
        "fatal_found",
        "fatal_reasons",
        "summary_feedback",
        "human_review_required",
    ]

    # Section scores
    sections = sorted({spec["section"] for spec in PARAM_SPECS})
    for section in sections:
        header.append(f"section_{section}_score")
        header.append(f"section_{section}_max_points")

    # Per-parameter scores & reasons
    for spec in PARAM_SPECS:
        pid = spec["id"]
        header.append(f"{pid}_score")
        header.append(f"{pid}_max_points")
        header.append(f"{pid}_reason")

    return header


def _params_index(params_obj: Any) -> Dict[str, Dict[str, Any]]:
    """
    normalize evaluate_ticket's `parameters` into a dict keyed by param id.
    supports:
      - list[{id, score, max_points, reason, ...}]
      - dict[id -> {...}]
    """
    if isinstance(params_obj, dict):
        return params_obj
    if isinstance(params_obj, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in params_obj:
            if isinstance(item, dict):
                pid = str(item.get("id") or "").strip()
                if pid:
                    out[pid] = item
        return out
    return {}


def _ticket_row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["ticket_id"] = result.get("ticket_id")
    row["overall_score"] = result.get("overall_score")
    row["verdict"] = result.get("verdict")
    row["fatal_found"] = result.get("fatal_found")
    row["fatal_reasons"] = "; ".join(result.get("fatal_reasons", []) or [])
    row["summary_feedback"] = result.get("summary_feedback", "")
    row["human_review_required"] = result.get("human_review_required", "NO")

    # Sections (dict)
    sections = result.get("sections", {}) or {}
    for section_name, sec_data in sections.items():
        sec_data = sec_data or {}
        row[f"section_{section_name}_score"] = sec_data.get("score")
        row[f"section_{section_name}_max_points"] = sec_data.get("max_points")

    # Parameters (evaluate_ticket returns LIST in your repo -> index it)
    params = _params_index(result.get("parameters", []))
    for spec in PARAM_SPECS:
        pid = spec["id"]
        pdata = params.get(pid, {}) or {}
        row[f"{pid}_score"] = pdata.get("score")
        row[f"{pid}_max_points"] = pdata.get("max_points")
        row[f"{pid}_reason"] = pdata.get("reason", "")

    return row


def _compute_batch_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    evaluated = 0
    excluded = 0
    scores: List[float] = []
    pass_count = 0
    fail_count = 0
    fatal_count = 0

    for r in results:
        # manual review tickets are excluded from avg
        if str(r.get("human_review_required", "NO")).upper() == "YES":
            excluded += 1
            continue

        score = r.get("overall_score")
        if isinstance(score, (int, float)):
            evaluated += 1
            scores.append(float(score))

        # evaluate_ticket returns "Pass"/"Fail" (Title case) in your repo
        verdict = str(r.get("verdict") or "").strip().lower()
        if verdict == "pass":
            pass_count += 1
        elif verdict == "fail":
            fail_count += 1

        if r.get("fatal_found") is True:
            fatal_count += 1

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "total_tickets": total,
        "evaluated_tickets": evaluated,
        "excluded_from_average": excluded,
        "avg_score": avg_score,
        "pass_threshold": TICKET_QA_PASS_THRESHOLD,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "fatal_count": fatal_count,
    }


def _compute_quality_insights(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    section_totals: Dict[str, Dict[str, float]] = {}
    parameter_totals: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if str(result.get("human_review_required", "NO")).upper() == "YES":
            continue

        for section, values in (result.get("sections") or {}).items():
            bucket = section_totals.setdefault(section, {"score": 0.0, "max_points": 0.0})
            bucket["score"] += float((values or {}).get("score") or 0)
            bucket["max_points"] += float((values or {}).get("max_points") or 0)

        for parameter in result.get("parameters") or []:
            if not isinstance(parameter, dict):
                continue
            pid = str(parameter.get("id") or "")
            if not pid:
                continue
            max_points = float(parameter.get("max_points") or 0)
            score = float(parameter.get("score") or 0)
            bucket = parameter_totals.setdefault(
                pid,
                {
                    "id": pid,
                    "label": parameter.get("label") or pid,
                    "section": parameter.get("section") or "other",
                    "lost_points": 0.0,
                    "issue_count": 0,
                },
            )
            bucket["lost_points"] += max(0.0, max_points - score)
            if parameter.get("issue_flag"):
                bucket["issue_count"] += 1

    section_scores = []
    for section, totals in section_totals.items():
        maximum = totals["max_points"]
        section_scores.append(
            {
                "section": section,
                "score_pct": round((totals["score"] / maximum) * 100, 1) if maximum else 0.0,
            }
        )

    opportunities = sorted(
        parameter_totals.values(),
        key=lambda item: (float(item["lost_points"]), int(item["issue_count"])),
        reverse=True,
    )[:8]
    for item in opportunities:
        item["lost_points"] = round(float(item["lost_points"]), 1)

    ticket_preview = [
        {
            "ticket_id": result.get("ticket_id"),
            "overall_score": result.get("overall_score"),
            "verdict": result.get("verdict"),
            "fatal_found": result.get("fatal_found"),
            "summary_feedback": result.get("summary_feedback"),
            "human_review_required": result.get("human_review_required", "NO"),
        }
        for result in results[:50]
    ]

    return {
        "section_scores": sorted(section_scores, key=lambda item: item["score_pct"]),
        "top_opportunities": opportunities,
        "ticket_preview": ticket_preview,
    }


# ---------------------------------------------------------------------------
# Main tool: Batch evaluate CSV (JOB-SCOPED like Heavy Hitter)
# ---------------------------------------------------------------------------
def batch_evaluate_csv(
    csv_file_base64: Optional[str] = None,
    csv_text: Optional[str] = None,
    file_path: Optional[str] = None,
    filename: Optional[str] = None,
    tool_context: ToolContext | None = None,
) -> Dict[str, Any]:
    """
    Batch-evaluate tickets from CSV.

    This version matches Heavy Hitter job flow:
    - creates jobs/<job_id>/
    - writes progress.json during normalize & evaluation
    - writes results.json at end (and also on error)
    - writes QA detailed CSV inside job folder

    IMPORTANT: Uses seeded `wla_job_id` from tool_context.state when present
    to avoid frontend/backend job id mismatches.
    """

    if not csv_file_base64 and not csv_text and not file_path:
        return {
            "summary_text": (
                "No CSV data provided. Please either upload a CSV file "
                "(csv_file_base64), pass raw CSV text via csv_text, "
                "or provide a server-side file_path."
            ),
            "stats": {
                "total_tickets": 0,
                "evaluated_tickets": 0,
                "excluded_from_average": 0,
                "avg_score": 0.0,
                "pass_threshold": TICKET_QA_PASS_THRESHOLD,
                "pass_count": 0,
                "fail_count": 0,
                "fatal_count": 0,
            },
            "job_id": None,
            "output_csv_path": None,
        }

    # Prefer seeded job id from route.ts so /api/results/{jobId} polling matches
    seed_job_id: Optional[str] = None
    try:
        seed_job_id = (tool_context.state or {}).get("wla_job_id") if tool_context is not None else None
    except Exception:
        seed_job_id = None

    job_id = str(seed_job_id).strip() if seed_job_id else _new_job_id()
    _job_dir(job_id)  # ensure folder exists

    _emit_progress(tool_context, job_id, "Job started", {"stage": "start"})

    try:
        # 1) Decode CSV into text
        decoded_text: str = ""
        if csv_text:
            decoded_text = csv_text
        elif csv_file_base64:
            decoded = base64.b64decode(csv_file_base64)
            decoded_text = decoded.decode("utf-8-sig", errors="replace")
        else:
            # file_path mode
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f_in:
                decoded_text = f_in.read()

        # 2) Parse CSV via pandas (auto-detect delimiter)
        results: List[Dict[str, Any]] = []

        # Hook normalize progress (same style you wanted)
        def _norm_progress(payload: Dict[str, Any]) -> None:
            try:
                stage = str(payload.get("stage") or payload.get("phase") or "normalize")
                rows = payload.get("rows")
                total = payload.get("total")
                if rows is not None and total is not None:
                    msg = f"Normalize: stage={stage}, rows={rows}/{total}"
                else:
                    msg = f"Normalize: stage={stage}"
                _emit_progress(tool_context, job_id, msg, {"stage": f"normalize_{stage}", **payload})
            except Exception:
                pass

        _emit_progress(tool_context, job_id, "Reading CSV into DataFrame", {"stage": "read_csv"})
        df = pd.read_csv(io.StringIO(decoded_text), sep=None, engine="python")

        _emit_progress(
            tool_context,
            job_id,
            "Normalizing columns",
            {"stage": "normalize_start", "rows": 0, "total": len(df)},
        )
        df, _cols, normalization_report = normalize_dataframe_with_report(
            df,
            create_combined_work_notes=True,
            progress_hook=_norm_progress,
        )

        total_rows = len(df)
        _emit_progress(
            tool_context,
            job_id,
            "Evaluation started",
            {"stage": "evaluate_start", "rows": 0, "total": total_rows},
        )

        # 3) Evaluate each ticket
        # Read progress cadence from env (default 10 in your current setup)
        try:
            N = int(os.getenv("QA_PROGRESS_EVERY_N", "10").strip())
        except Exception:
            N = 10
        N = max(1, N)

        for i, ticket in enumerate(df.fillna("").to_dict("records"), start=1):
            ticket = normalize_ticket_dict(ticket)
            try:
                r = evaluate_ticket(ticket, tool_context=tool_context)
            except Exception as e:
                # hard-fail safety: still return something row-shaped
                r = {
                    "ticket_id": ticket.get("ticket_id") or ticket.get("id") or ticket.get("number") or f"row_{i}",
                    "overall_score": 0,
                    "verdict": "Fail",
                    "fatal_found": True,
                    "fatal_reasons": [f"Exception during evaluation: {e}"],
                    "summary_feedback": "Human review required due to evaluation exception.",
                    "human_review_required": "YES",
                    "sections": {},
                    "parameters": [],
                }
            results.append(r)

            if i % N == 0 or i == total_rows:
                _emit_progress(
                    tool_context,
                    job_id,
                    f"Evaluate: rows={i}/{total_rows}",
                    {"stage": "evaluate", "rows": i, "total": total_rows},
                )

        # 4) Write output CSV inside job folder
        safe_base = os.path.splitext(os.path.basename(filename or f"{job_id}.csv"))[0]
        output_name = f"{safe_base}_qa_analysis.csv"
        output_csv_path = os.path.join(_job_dir(job_id), output_name)

        header = _build_output_header()
        _emit_progress(tool_context, job_id, "Writing QA analysis CSV", {"stage": "write_csv"})
        with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            for r in results:
                writer.writerow(_ticket_row_from_result(r))

        # 5) Stats + summary
        stats = _compute_batch_stats(results)
        quality_insights = _compute_quality_insights(results)

        summary_text = (
            f"I have analyzed the CSV ticket data.\n\n"
            f"Total Tickets in CSV: {stats['total_tickets']}\n"
            f"Tickets Evaluated in score averages: {stats['evaluated_tickets']}\n"
            f"Tickets Marked for Manual review (not evaluated): {stats['excluded_from_average']}\n"
            f"Average Score (evaluated tickets): {stats['avg_score']}/100\n"
            f"Pass: {stats['pass_count']} | Fail: {stats['fail_count']} | Fatal: {stats['fatal_count']}\n\n"
            f"Detailed QA output saved to: {output_csv_path}\n"
            f"Job folder: {_job_dir(job_id)}"
        )

        # 6) Write results.json (what your /api/results/{job_id} expects)
        results_payload: Dict[str, Any] = {
            "ok": True,
            "job_id": job_id,
            "kind": "qa_batch",
            "summary_text": summary_text,
            "stats": stats,
            "quality_insights": quality_insights,
            "output_csv_path": output_csv_path,
            "normalization_report": normalization_report,
        }
        with open(_job_results_path(job_id), "w", encoding="utf-8") as f:
            json.dump(results_payload, f, ensure_ascii=False, indent=2)

        _emit_progress(tool_context, job_id, "Job complete", {"stage": "done", "output_csv_path": output_csv_path})

        return {
            "summary_text": summary_text,
            "stats": stats,
            "quality_insights": quality_insights,
            "job_id": job_id,
            "output_csv_path": output_csv_path,
            "normalization_report": normalization_report,
        }

    except Exception as e:
        # Always write results.json on failure too (prevents endless 404 polling)
        try:
            err_payload: Dict[str, Any] = {
                "ok": False,
                "job_id": job_id,
                "kind": "qa_batch",
                "error": str(e),
            }
            with open(_job_results_path(job_id), "w", encoding="utf-8") as f:
                json.dump(err_payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        _emit_progress(tool_context, job_id, "Job failed", {"stage": "error", "error": str(e)})
        raise


batch_evaluate_csv_tool = FunctionTool(batch_evaluate_csv)
