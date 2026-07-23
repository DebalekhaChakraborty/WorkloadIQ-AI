# workload_analysis/tools/heavy_hitter_analysis.py
from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from google.adk.tools import FunctionTool, ToolContext  # type: ignore

# Vertex AI (repo-aligned)
import vertexai  # type: ignore
from vertexai.preview.generative_models import GenerativeModel  # type: ignore

# Optional charts (ADK Web only, in-memory)
try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Output directory — HARD ALIGNED with Ticket QA
# -----------------------------------------------------------------------------
TICKET_QA_OUTPUT_DIR = os.getenv("TICKET_QA_OUTPUT_DIR")


def _ensure_output_dir() -> str:
    out = (TICKET_QA_OUTPUT_DIR or "").strip()
    if not out:
        out = "./outputs"
    os.makedirs(out, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Job-scoped output (STRICT: job-scoped only; NO latest)
# -----------------------------------------------------------------------------
def _utc_compact_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _shortuuid() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


def _new_job_id() -> str:
    # ✅ readable, sortable job id
    return f"wla_{_utc_compact_ts()}_{_shortuuid()}"


def _jobs_dir() -> str:
    out_dir = _ensure_output_dir()
    d = os.path.join(out_dir, "jobs")
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


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------
HEAVY_HITTER_LLM_MODEL = os.getenv("HEAVY_HITTER_LLM_MODEL", "gemini-2.5-flash")
HEAVY_HITTER_TOP_N = int(os.getenv("HEAVY_HITTER_TOP_N", "10"))
HEAVY_HITTER_SAMPLE_ROWS = int(os.getenv("HEAVY_HITTER_SAMPLE_ROWS", "30"))

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "")


def _vertex_ready() -> bool:
    return bool(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION)


def _init_vertex():
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)


# -----------------------------------------------------------------------------
# Input parsing helpers
# -----------------------------------------------------------------------------
_UPLOAD_PATH_RE = re.compile(r"\[uploaded_file_path\]\s*:\s*(.+)", re.IGNORECASE)
_WLA_JOB_ID_RE = re.compile(r"\[wla_job_id\]\s*:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)

# Strict allowlist for job ids to prevent traversal / weird chars
_JOB_ID_SAFE_RE = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")


def _sanitize_job_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not _JOB_ID_SAFE_RE.match(s):
        return None
    return s


def _read_seeded_job_id(tool_context: ToolContext) -> Optional[str]:
    """
    Option A: portal injects a deterministic job id.
    Read it from (most -> least preferred):

    1) state['wla_job_id']
    2) state['metadata']['wla_job_id']
    3) state['session']['state']['wla_job_id']
    4) state['session']['state']['metadata']['wla_job_id']
    5) last user message text: [wla_job_id]: <id>
    """
    state = tool_context.state or {}

    def _get_str(d: Any) -> Optional[str]:
        if isinstance(d, str):
            return _sanitize_job_id(d)
        return None

    # 1) direct
    s1 = _get_str(state.get("wla_job_id"))
    if s1:
        return s1

    # 2) metadata
    meta = state.get("metadata")
    if isinstance(meta, dict):
        s2 = _get_str(meta.get("wla_job_id"))
        if s2:
            return s2

    # 3/4) nested session.state
    sess = state.get("session")
    if isinstance(sess, dict):
        sess_state = sess.get("state")
        if isinstance(sess_state, dict):
            s3 = _get_str(sess_state.get("wla_job_id"))
            if s3:
                return s3
            meta2 = sess_state.get("metadata")
            if isinstance(meta2, dict):
                s4 = _get_str(meta2.get("wla_job_id"))
                if s4:
                    return s4

    # 5) parse from last user message (if ADK kept it in state)
    for key in ("messages", "history", "chat_history"):
        msgs = state.get(key)
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                if not isinstance(m, dict):
                    continue
                if m.get("role") != "user":
                    continue
                parts = m.get("parts")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        txt = p["text"]
                        m2 = _WLA_JOB_ID_RE.search(txt)
                        if m2:
                            s5 = _sanitize_job_id(m2.group(1))
                            if s5:
                                return s5

    return None


def _read_upload_path(tool_context: ToolContext) -> Optional[str]:
    """
    Resolve uploaded file path from:
    1) state['uploads']['filePath'] or state['uploads']['file_path']
    2) state['uploaded_file_path']
    3) nested state: state['session']['state']['uploads']['filePath']
    4) parse from user message: [uploaded_file_path]: /abs/path.csv
    """
    state = tool_context.state or {}

    uploads = state.get("uploads")
    if isinstance(uploads, dict):
        fp = uploads.get("filePath") or uploads.get("file_path")
        if isinstance(fp, str) and fp.strip():
            return fp.strip()

    fp2 = state.get("uploaded_file_path")
    if isinstance(fp2, str) and fp2.strip():
        return fp2.strip()

    sess = state.get("session")
    if isinstance(sess, dict):
        sess_state = sess.get("state")
        if isinstance(sess_state, dict):
            uploads2 = sess_state.get("uploads")
            if isinstance(uploads2, dict):
                fp3 = uploads2.get("filePath") or uploads2.get("file_path")
                if isinstance(fp3, str) and fp3.strip():
                    return fp3.strip()

    for key in ("messages", "history", "chat_history"):
        msgs = state.get(key)
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                if not isinstance(m, dict):
                    continue
                if m.get("role") != "user":
                    continue
                parts = m.get("parts")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        txt = p["text"]
                        m2 = _UPLOAD_PATH_RE.search(txt)
                        if m2:
                            candidate = m2.group(1).strip()
                            if candidate:
                                return candidate
    return None


def _read_csv_text(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _read_csv(decoded_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(decoded_text), sep=None, engine="python")


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------
def _safe_bucket(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    return s if s else "(blank)"


def _series_table_with_denominator(s: pd.Series, *, denominator: Optional[int]) -> pd.DataFrame:
    if s.empty:
        return pd.DataFrame(columns=["bucket", "count", "pct", "cum_pct"])
    d = pd.DataFrame({"bucket": s.index.astype(str), "count": s.values})
    denom = float(denominator) if denominator and denominator > 0 else float(d["count"].sum())
    d["pct"] = (d["count"] / denom).round(4)
    d["cum_pct"] = d["pct"].cumsum().round(4)
    return d


def _series_table(s: pd.Series) -> pd.DataFrame:
    return _series_table_with_denominator(s, denominator=None)


def _top_series_with_other(
    df: pd.DataFrame,
    col: Optional[str],
    top_n: int,
    *,
    other_label: str = "(All other categories)",
) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype=int)
    vc = df[col].map(_safe_bucket).value_counts()
    head = vc.head(top_n)
    rest = int(vc.iloc[top_n:].sum()) if len(vc) > top_n else 0
    if rest > 0:
        head = pd.concat([head, pd.Series({other_label: rest})])
    return head


# -----------------------------------------------------------------------------
# In-memory JPG charts (no disk writes)
# -----------------------------------------------------------------------------
def _fig_to_jpg_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpeg", dpi=160)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _pareto_chart_b64(series: pd.Series, title: str) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE or series.empty:
        return None

    s = series.copy()
    counts = s.values.astype(float)
    total = counts.sum()
    if total <= 0:
        return None
    pct = counts / total
    cum = pct.cumsum()

    fig = plt.figure(figsize=(10, 4.8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.bar(range(len(s)), counts)
    ax2.plot(range(len(s)), cum, marker="o")

    ax1.set_title(title)
    ax1.set_xticks(range(len(s)))
    ax1.set_xticklabels([str(x) for x in s.index], rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.8, linestyle="--")

    return _fig_to_jpg_base64(fig)


def _topn_bar_b64(series: pd.Series, title: str) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE or series.empty:
        return None
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(range(len(series)), series.values.astype(float))
    ax.set_title(title)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([str(x) for x in series.index], rotation=45, ha="right")
    return _fig_to_jpg_base64(fig)


# -----------------------------------------------------------------------------
# Canonical column mapping
# -----------------------------------------------------------------------------
class CanonicalCols:
    def __init__(
        self,
        ticket_id: Optional[str] = None,
        created: Optional[str] = None,
        closed: Optional[str] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        assignment_group: Optional[str] = None,
        priority: Optional[str] = None,
        state: Optional[str] = None,
        short_description: Optional[str] = None,
        description: Optional[str] = None,
        work_notes: Optional[str] = None,
        ticket_type: Optional[str] = None,
        resolution: Optional[str] = None,
        impact: Optional[str] = None,
        urgency: Optional[str] = None,
        user_confirmation: Optional[str] = None,
    ):
        self.ticket_id = ticket_id
        self.created = created
        self.closed = closed
        self.category = category
        self.subcategory = subcategory
        self.assignment_group = assignment_group
        self.priority = priority
        self.state = state
        self.short_description = short_description
        self.description = description
        self.work_notes = work_notes
        self.ticket_type = ticket_type
        self.resolution = resolution
        self.impact = impact
        self.urgency = urgency
        self.user_confirmation = user_confirmation


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
from .normalize_columns import normalize_dataframe_with_report  # type: ignore


def _emit_progress(
    tool_context: ToolContext,
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
    try:
        state = tool_context.state or {}
        state["heavy_hitter_progress"] = payload
        state["wla_job_id"] = job_id
        tool_context.state = state
    except Exception:
        pass

    # 2) Persist job-scoped progress
    try:
        with open(_job_progress_path(job_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 3) Console
    try:
        print(f"[heavy_hitter:{job_id}] {message}")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Deterministic stats
# -----------------------------------------------------------------------------
def _compute_time_stats(df: pd.DataFrame, cols: CanonicalCols) -> Dict[str, Any]:
    if not cols.created or cols.created not in df.columns:
        return {"has_created": False}
    out: Dict[str, Any] = {"has_created": True}

    created = pd.to_datetime(df[cols.created], errors="coerce", utc=True)
    out["created_parseable_rows"] = int(created.notna().sum())

    if cols.closed and cols.closed in df.columns:
        closed = pd.to_datetime(df[cols.closed], errors="coerce", utc=True)
        out["has_closed"] = True
        out["closed_parseable_rows"] = int(closed.notna().sum())
        resolved_mask = created.notna() & closed.notna()
        out["resolved_rows"] = int(resolved_mask.sum())
        if resolved_mask.any():
            delta_hours = (closed[resolved_mask] - created[resolved_mask]).dt.total_seconds() / 3600.0
            out["resolution_time_hours_median"] = float(delta_hours.median())
            out["resolution_time_hours_p90"] = float(delta_hours.quantile(0.9))
    else:
        out["has_closed"] = False

    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - created).dt.total_seconds() / 86400.0
    age_days = age_days.dropna()
    if len(age_days) > 0:
        out["ticket_age_days_median"] = float(age_days.median())
        out["ticket_age_days_p90"] = float(age_days.quantile(0.9))
    return out


def _compute_fix_notes_stats(df: pd.DataFrame, cols: CanonicalCols) -> Dict[str, Any]:
    if not cols.work_notes or cols.work_notes not in df.columns:
        return {"has_fix_notes": False}
    s = df[cols.work_notes].fillna("").astype(str)
    non_empty = s.str.strip().ne("")
    lens = s[non_empty].str.len()
    return {
        "fix_notes_column": cols.work_notes,
        "fix_notes_non_empty_ratio": float(non_empty.mean()),
        "fix_notes_median_length": float(lens.median()) if len(lens) > 0 else 0.0,
        "fix_notes_non_empty_rows": int(non_empty.sum()),
    }


def _top_statuses(df: pd.DataFrame, cols: CanonicalCols, top_n: int = 10) -> Dict[str, Any]:
    if not cols.state or cols.state not in df.columns:
        return {"has_status": False}
    vc = df[cols.state].fillna("").astype(str).str.strip()
    vc = vc[vc.ne("")]
    counts = vc.value_counts().head(top_n)
    return {
        "status_column": cols.state,
        "top_status_counts": [{"status": k, "count": int(v)} for k, v in counts.items()],
        "distinct_status_count": int(vc.nunique()),
        "non_empty_ratio": float(df[cols.state].fillna("").astype(str).str.strip().ne("").mean()),
    }


# -----------------------------------------------------------------------------
# LLM helpers (robust to multi-part responses)
# -----------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_text_from_vertex_response(resp: Any) -> str:
    """
    Vertex SDK sometimes returns multiple content parts; resp.text can raise:
      ValueError: Multiple content parts are not supported.
    This function safely concatenates text across all parts.
    """
    # 1) try the convenience property
    try:
        t = getattr(resp, "text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # 2) candidates -> content -> parts
    texts: List[str] = []
    try:
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list) and cands:
            cand0 = cands[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None)
            if isinstance(parts, list):
                for p in parts:
                    # parts may be objects with .text or dicts
                    if isinstance(p, dict):
                        txt = p.get("text")
                        if isinstance(txt, str) and txt:
                            texts.append(txt)
                    else:
                        txt = getattr(p, "text", None)
                        if isinstance(txt, str) and txt:
                            texts.append(txt)
    except Exception:
        pass

    return "".join(texts).strip()


def _best_effort_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Prefer JSON object. Accepts:
      - pure JSON
      - fenced ```json ... ```
      - JSON + extra prose (we attempt to pull the first {...} block)
    """
    if not text:
        return None

    # fenced JSON
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # pure JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # first object heuristic
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None

    return None


# -----------------------------------------------------------------------------
# LLM prompt + run
# -----------------------------------------------------------------------------
LLM_PROMPT_TEMPLATE = """You are an IT Service Management analyst.

You are given deterministic statistics computed from a ticket dump (CSV) AND a small sample of tickets.
Your job: identify the top heavy hitters (recurring themes causing the most ticket volume or delay) and provide actions.

Rules:
- Ground your analysis in the deterministic tables.
- Do NOT hallucinate tools/systems.
- Be specific, practical, and concise.
- Return ONLY valid JSON (no markdown, no prose).

Return valid JSON with keys:
- top_heavy_hitters: list of objects with fields: rank, theme, primary_bucket, why_it_happens, recommended_actions, automation_candidates
- cross_cutting_observations: list of strings
- data_gaps: list of strings

Deterministic Stats:
TOTAL_TICKETS: {total}

COLUMN_MAPPING:
{cols_json}

TOP CATEGORY (by count):
{top_cat_csv}

TOP CATEGORY::SUBCATEGORY:
{top_cat_sub_csv}

TOP ASSIGNMENT GROUPS:
{top_group_csv}

TIME STATS:
{time_stats_json}

FIX NOTES STATS:
{fix_notes_stats_json}

STATUS STATS:
{status_stats_json}

SAMPLE TICKETS (up to {sample_n} rows):
{sample_csv}
"""


def _build_llm_prompt(
    total: int,
    cols: CanonicalCols,
    top_cat: pd.DataFrame,
    top_cat_sub: pd.DataFrame,
    top_group: pd.DataFrame,
    sample_df: pd.DataFrame,
    time_stats: Dict[str, Any],
    fix_notes_stats: Dict[str, Any],
    status_stats: Dict[str, Any],
) -> str:
    return LLM_PROMPT_TEMPLATE.format(
        total=total,
        cols_json=json.dumps(cols.__dict__, ensure_ascii=False, indent=2),
        top_cat_csv=top_cat.to_csv(index=False),
        top_cat_sub_csv=top_cat_sub.to_csv(index=False) if not top_cat_sub.empty else "",
        top_group_csv=top_group.to_csv(index=False) if not top_group.empty else "",
        time_stats_json=json.dumps(time_stats, ensure_ascii=False, indent=2),
        fix_notes_stats_json=json.dumps(fix_notes_stats, ensure_ascii=False, indent=2),
        status_stats_json=json.dumps(status_stats, ensure_ascii=False, indent=2),
        sample_n=len(sample_df),
        sample_csv=sample_df.to_csv(index=False),
    )


def _run_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """
    ✅ Robust to multi-part responses.
    ✅ Attempts to parse JSON.
    ✅ Still returns raw text for debugging.
    """
    if not _vertex_ready():
        return None

    _init_vertex()
    model = GenerativeModel(HEAVY_HITTER_LLM_MODEL)

    try:
        # If SDK supports it, this nudges model towards pure JSON
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 65536,
                # Some SDK versions accept this; if ignored, no harm.
                "response_mime_type": "application/json",
            },
        )
    except TypeError:
        # Older SDK: no generation_config accepted
        resp = model.generate_content(prompt)

    text = _extract_text_from_vertex_response(resp)
    if not text:
        return None

    parsed = _best_effort_parse_json(text)

    out: Dict[str, Any] = {"raw_llm_output": text}
    if parsed is not None:
        out["parsed_json"] = parsed
    else:
        out["parse_error"] = "Model did not return valid JSON."
    return out


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def _sample_rows(df: pd.DataFrame, cols: CanonicalCols) -> pd.DataFrame:
    keep = []
    for c in [
        cols.ticket_id,
        cols.state,
        cols.ticket_type,
        cols.category,
        cols.subcategory,
        cols.assignment_group,
        cols.priority,
        cols.short_description,
        cols.description,
        cols.work_notes,
        cols.resolution,
        cols.created,
        cols.closed,
    ]:
        if c and c in df.columns and c not in keep:
            keep.append(c)

    sample = df[keep].head(HEAVY_HITTER_SAMPLE_ROWS) if keep else df.head(HEAVY_HITTER_SAMPLE_ROWS)
    return sample.copy()


# -----------------------------------------------------------------------------
# Main tool
# -----------------------------------------------------------------------------
def heavy_hitter_analyze(tool_context: ToolContext, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a ticket dump to find heavy hitters.

    Option A (portal-seeded job_id) + STRICT job-scoped files:
    - Uses seeded job_id if present; else generates one.
    - Writes:
        TICKET_QA_OUTPUT_DIR/jobs/<job_id>/progress.json
        TICKET_QA_OUTPUT_DIR/jobs/<job_id>/results.json
        TICKET_QA_OUTPUT_DIR/jobs/<job_id>/<base>_heavy_hitter_summary.csv
    - Returns job_id + output_csv_path for the portal.
    """
    seeded = _read_seeded_job_id(tool_context)
    job_id = seeded or _new_job_id()

    # Always create the job dir up-front so progress/results can be written even on failure
    _job_dir(job_id)

    file_path = _read_upload_path(tool_context)
    if not file_path:
        err = {
            "error": (
                "No uploaded file path found. Expected state['uploads']['filePath'] "
                "OR a user message containing '[uploaded_file_path]: /abs/path'."
            ),
            "job_id": job_id,
        }
        _emit_progress(tool_context, job_id, "ERROR: No uploaded file path found.", data=err)
        try:
            with open(_job_results_path(job_id), "w", encoding="utf-8") as f:
                json.dump(
                    {"ts": datetime.utcnow().isoformat() + "Z", "job_id": job_id, "message": "Failed", "data": err},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass
        return err

    if filename is None:
        filename = os.path.basename(file_path)

    try:
        _emit_progress(tool_context, job_id, f"Reading CSV: {file_path}", data={"job_id": job_id})
        text = _read_csv_text(file_path)
        df = _read_csv(text)

        _emit_progress(tool_context, job_id, f"Loaded CSV. Rows={len(df)}, Cols={len(df.columns)}. Normalizing columns…")

        top_n = HEAVY_HITTER_TOP_N

        # ---- Progress hook adapter (supports BOTH dict payload and positional args) ----
        def _norm_progress(*args, **kwargs) -> None:
            if len(args) == 1 and isinstance(args[0], dict):
                payload = dict(args[0])
                stage = str(payload.get("stage") or payload.get("phase") or payload.get("step") or "normalize")
                batch_idx = int(payload.get("batch") or payload.get("batch_idx") or 0)
                batch_total = int(payload.get("batch_total") or payload.get("batches") or 0)
                done = int(payload.get("done") or payload.get("rows_done") or payload.get("rows") or 0)
                total_rows = int(payload.get("total") or payload.get("rows_total") or 0)
            else:
                stage = str(args[0]) if len(args) > 0 else "normalize"
                batch_idx = int(args[1]) if len(args) > 1 else 0
                batch_total = int(args[2]) if len(args) > 2 else 0
                done = int(args[3]) if len(args) > 3 else 0
                total_rows = int(args[4]) if len(args) > 4 else 0
                payload = {
                    "stage": stage,
                    "batch": batch_idx,
                    "batch_total": batch_total,
                    "done": done,
                    "total": total_rows,
                }

            parts = [f"stage={stage}"]
            if batch_total:
                parts.append(f"batch={batch_idx}/{batch_total}")
            if total_rows:
                parts.append(f"rows={done}/{total_rows}")
            msg = "Normalize: " + ", ".join(parts)
            _emit_progress(tool_context, job_id, msg, data=payload)

        df, cols, normalization_report = normalize_dataframe_with_report(
            df,
            create_combined_work_notes=True,
            progress_hook=_norm_progress,
        )
        total = len(df)

        _emit_progress(tool_context, job_id, f"Normalization complete. Rows={total}. Computing aggregations…")

        # Deterministic grounding
        cat_series = _top_series_with_other(df, cols.category, top_n)
        top_cat = _series_table_with_denominator(cat_series, denominator=total)

        top_cat_sub = (
            _series_table(
                (df[cols.category].map(_safe_bucket) + " :: " + df[cols.subcategory].map(_safe_bucket))
                .value_counts()
                .head(top_n)
            )
            if cols.category and cols.subcategory
            else pd.DataFrame()
        )

        group_series = _top_series_with_other(df, cols.assignment_group, top_n, other_label="(All other groups)")
        top_group = _series_table_with_denominator(group_series, denominator=total)

        # Deterministic metrics
        time_stats = _compute_time_stats(df, cols)
        fix_notes_stats = _compute_fix_notes_stats(df, cols)
        status_stats = _top_statuses(df, cols)

        _emit_progress(tool_context, job_id, "Deterministic stats computed. Running heavy-hitter LLM summary…")

        # LLM intelligence
        llm_prompt = _build_llm_prompt(
            total,
            cols,
            top_cat,
            top_cat_sub,
            top_group,
            _sample_rows(df, cols),
            time_stats,
            fix_notes_stats,
            status_stats,
        )
        llm_insights = _run_llm(llm_prompt)

        _emit_progress(tool_context, job_id, "LLM summary complete. Generating charts/output…")

        # Charts (ADK Web only)
        charts: Dict[str, str] = {}
        pareto = _pareto_chart_b64(cat_series, "Pareto: Categories")
        group_bar = _topn_bar_b64(group_series, "Top Assignment Groups")
        if pareto:
            charts["pareto_category_jpg_base64"] = pareto
        if group_bar:
            charts["top_assignment_group_jpg_base64"] = group_bar

        # ✅ CSV output strictly inside job dir
        job_dir = _job_dir(job_id)
        safe_name = os.path.basename(filename or os.path.basename(file_path) or "ticket_dump.csv")
        base = safe_name.rsplit(".", 1)[0]
        out_csv = os.path.join(job_dir, f"{base}_heavy_hitter_summary.csv")
        out_csv_filename = os.path.basename(out_csv)

        rows: List[Dict[str, Any]] = []
        rows.append({"section": "overview", "key": "job_id", "value": job_id})
        rows.append({"section": "overview", "key": "total_tickets", "value": total})
        rows.append({"section": "overview", "key": "columns_detected", "value": cols.__dict__})
        rows.append({"section": "overview", "key": "llm_enabled", "value": _vertex_ready()})

        rows.append({"section": "overview", "key": "normalization_report", "value": normalization_report})
        rows.append({"section": "overview", "key": "time_stats", "value": time_stats})
        rows.append({"section": "overview", "key": "fix_notes_stats", "value": fix_notes_stats})
        rows.append({"section": "overview", "key": "status_stats", "value": status_stats})

        for _, r in top_cat.iterrows():
            rows.append({"section": "top_category", **r.to_dict()})
        for _, r in top_cat_sub.iterrows():
            rows.append({"section": "top_category_subcategory", **r.to_dict()})
        for _, r in top_group.iterrows():
            rows.append({"section": "top_assignment_group", **r.to_dict()})

        if llm_insights:
            rows.append({"section": "llm", "key": "insights_json", "value": llm_insights})

        pd.DataFrame(rows).to_csv(out_csv, index=False)

        _emit_progress(
            tool_context,
            job_id,
            "Done.",
            data={"job_id": job_id, "output_csv_path": out_csv, "output_csv_filename": out_csv_filename},
        )

        result_payload: Dict[str, Any] = {
            "job_id": job_id,
            "summary_text": "Heavy hitter analysis complete (LLM-infused).",
            "total_tickets": total,
            "detected_columns": cols.__dict__,
            "top_categories": top_cat.to_dict("records"),
            "top_category_subcategories": top_cat_sub.to_dict("records"),
            "top_assignment_groups": top_group.to_dict("records"),
            "charts": charts,
            "output_csv_path": out_csv,
            "output_csv_filename": out_csv_filename,
            "llm_insights": llm_insights,
            "normalization_report": normalization_report,
            "time_stats": time_stats,
            "fix_notes_stats": fix_notes_stats,
            "status_stats": status_stats,
        }

        # Persist final results job-scoped
        final = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "job_id": job_id,
            "message": "Done.",
            "data": {"result": result_payload},
        }

        try:
            with open(_job_results_path(job_id), "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return result_payload

    except Exception as e:
        # ✅ Ensure progress + results.json exist even on failure
        err_payload = {
            "job_id": job_id,
            "error": str(e),
            "message": "Heavy hitter analysis failed.",
        }
        _emit_progress(tool_context, job_id, f"ERROR: {e}", data=err_payload)
        try:
            with open(_job_results_path(job_id), "w", encoding="utf-8") as f:
                json.dump(
                    {"ts": datetime.utcnow().isoformat() + "Z", "job_id": job_id, "message": "Failed", "data": err_payload},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass
        return err_payload


heavy_hitter_analyze_tool = FunctionTool(heavy_hitter_analyze)
