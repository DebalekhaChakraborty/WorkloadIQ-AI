from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore


def _normalize_header(name: str) -> str:
    """Normalize a header into a stable snake-ish key for matching."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


@dataclass
class CanonicalColumns:
    """
    Canonical field mapping for ticket dumps.

    The values are the *actual* column names present in the incoming dataframe
    that best match each canonical concept.
    """
    ticket_id: Optional[str] = None
    created: Optional[str] = None
    closed: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    assignment_group: Optional[str] = None
    priority: Optional[str] = None
    state: Optional[str] = None
    short_description: Optional[str] = None
    description: Optional[str] = None
    work_notes: Optional[str] = None

    # QA-centric common fields (may be the same as above depending on dataset)
    ticket_type: Optional[str] = None
    resolution: Optional[str] = None
    impact: Optional[str] = None
    urgency: Optional[str] = None
    user_confirmation: Optional[str] = None


# -----------------------------------------------------------------------------
# Synonyms across ServiceNow/ITSM + Jira exports.
# Keep this list conservative and additive.
# -----------------------------------------------------------------------------
_COL_SYNONYMS: Dict[str, List[str]] = {
    "ticket_id": [
        "ticket_id", "id", "number", "incident", "request", "case", "task",
        "issue key", "issue_key", "issue id", "issue_id", "key",
    ],
    "created": ["created", "opened_at", "opened", "created_at", "open_time"],
    "closed": [
        "closed", "closed_at", "resolved_at", "resolved",
        "resolution_date", "resolved_date",
        "custom field (resolved date)", "custom_field_resolved_date",
    ],
    "category": [
        "category", "cat", "issue_category",
        "issue type", "issue_type",
        "custom field (issue type name)", "custom_field_issue_type_name",
    ],
    "subcategory": [
        "subcategory", "sub_category", "sub category", "subcat",
        "component", "components", "component/s", "components/s",
    ],
    "assignment_group": [
        "assignment_group", "resolver_group", "support_group", "assignment group",
        "support group", "team",
        "custom field (resolved by)", "custom_field_resolved_by",
        "assignee", "assignee name", "assignee display name",
    ],
    "priority": ["priority", "prio", "severity"],
    "state": ["state", "status"],
    "short_description": ["short_description", "summary", "title", "short description"],
    "description": ["description", "long_description", "custom field (description)", "customfield_description"],
    "work_notes": [
        "work_notes", "work notes", "notes", "resolution_notes", "resolution notes",
        "custom field (fix notes)", "custom field (fix notes1)",
        "custom field (mas notes)", "custom field (release notes)",
        "service request resolution", "service request resolution simplified",
        "comment", "comments",
    ],
    "ticket_type": [
        "type", "ticket_type", "ticket type",
        "issue type", "issue_type",
        "custom field (issue type name)", "custom_field_issue_type_name",
    ],
    "resolution": [
        "resolution",
        "resolution_notes", "resolution notes",
        "custom field (resolution code)", "resolution code",
    ],
    "impact": ["impact"],
    "urgency": ["urgency", "priority", "prio", "severity"],
    "user_confirmation": ["user_confirmation", "user confirmation", "confirmation", "customer_confirmation", "user confirm"],
}

_WORKNOTES_MERGE_HINTS = [
    "fix notes", "fix notes1", "mas notes", "release notes",
    "service request resolution", "service request resolution simplified",
    "resolution", "work notes", "notes", "comment"
]

_SUBCATEGORY_EXCLUDE_HINTS = [
    "fix version", "fix versions", "fix_version", "fix_version_s",
    "affects version", "affects versions", "affects_version",
    "release", "build", "sprint",
]


# -----------------------------------------------------------------------------
# Env parsing helpers (avoid import-time crashes)
# -----------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v)


# -----------------------------------------------------------------------------
# LLM semantic categorization (free-form themes + auto-bucketing)
# -----------------------------------------------------------------------------
NORMALIZE_USE_LLM = _env_bool("NORMALIZE_USE_LLM", True)
NORMALIZE_LLM_MODEL = _env_str("NORMALIZE_LLM_MODEL", "gemini-2.5-flash").strip()
NORMALIZE_LLM_BATCH_SIZE = _env_int("NORMALIZE_LLM_BATCH_SIZE", 20)

# IMPORTANT (Strategy A):
# This is now used as "taxonomy seed sample size", NOT a hard cap on analysis rows.
NORMALIZE_LLM_MAX_ROWS = _env_int("NORMALIZE_LLM_MAX_ROWS", 5000)

NORMALIZE_LLM_MAX_BUCKETS = _env_int("NORMALIZE_LLM_MAX_BUCKETS", 12)
NORMALIZE_LLM_CATEGORY_COL = (_env_str("NORMALIZE_LLM_CATEGORY_COL", "category__semantic").strip() or "category__semantic")

# large file mode threshold (skip stage-1 themes for the *seed* if large)
NORMALIZE_LLM_LARGE_FILE_THRESHOLD = _env_int("NORMALIZE_LLM_LARGE_FILE_THRESHOLD", 300)

# force semantic category derivation even if category doesn't look generic
NORMALIZE_FORCE_SEMANTIC_CATEGORY = _env_bool("NORMALIZE_FORCE_SEMANTIC_CATEGORY", False)

GCP_PROJECT = _env_str("GOOGLE_CLOUD_PROJECT", "").strip()
GCP_LOCATION = _env_str("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
USE_VERTEX = _env_bool("GOOGLE_GENAI_USE_VERTEXAI", True)

_OTHER_BUCKET = "Other / Unclear"

_GENERIC_CATEGORY_VALUES = {
    "incident",
    "service request",
    "service request with approvals",
    "service request with approval",
    "request",
    "sr",
}


def _vertex_ready() -> bool:
    return bool(NORMALIZE_USE_LLM and USE_VERTEX and GCP_PROJECT and GCP_LOCATION and NORMALIZE_LLM_MODEL)


def _try_import_vertex() -> Tuple[bool, Optional[str]]:
    try:
        import vertexai  # type: ignore
        from vertexai.preview.generative_models import GenerativeModel  # type: ignore  # noqa
        return True, None
    except Exception as e:
        return False, str(e)


def _pick_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    norm_map = {_normalize_header(c): c for c in df.columns}
    for syn in _COL_SYNONYMS.get(canonical, []):
        key = _normalize_header(syn)
        if key in norm_map:
            return norm_map[key]
    return None


def infer_columns(df: pd.DataFrame) -> CanonicalColumns:
    return CanonicalColumns(
        ticket_id=_pick_column(df, "ticket_id"),
        created=_pick_column(df, "created"),
        closed=_pick_column(df, "closed"),
        category=_pick_column(df, "category"),
        subcategory=_pick_column(df, "subcategory"),
        assignment_group=_pick_column(df, "assignment_group"),
        priority=_pick_column(df, "priority"),
        state=_pick_column(df, "state"),
        short_description=_pick_column(df, "short_description"),
        description=_pick_column(df, "description"),
        work_notes=_pick_column(df, "work_notes"),
        ticket_type=_pick_column(df, "ticket_type"),
        resolution=_pick_column(df, "resolution"),
        impact=_pick_column(df, "impact"),
        urgency=_pick_column(df, "urgency"),
        user_confirmation=_pick_column(df, "user_confirmation"),
    )


def _non_empty_ratio(series: pd.Series) -> float:
    s = series.fillna("").astype(str).str.strip()
    s = s.where(~s.str.lower().isin(["nan", "none"]), "")
    return float(s.ne("").mean())


def _is_viable_column(df: pd.DataFrame, col: Optional[str], *, min_non_empty_ratio: float = 0.05) -> bool:
    if not col or col not in df.columns:
        return False
    try:
        return _non_empty_ratio(df[col]) >= float(min_non_empty_ratio)
    except Exception:
        return False


def _maybe_discard_semantically_wrong_cols(cols: CanonicalColumns) -> None:
    if cols.subcategory:
        n = _normalize_header(cols.subcategory)
        if any(_normalize_header(h) in n for h in _SUBCATEGORY_EXCLUDE_HINTS):
            cols.subcategory = None


def _distinct_non_empty_values(series: pd.Series, limit: int = 50) -> List[str]:
    s = series.fillna("").astype(str).str.strip()
    s = s.where(~s.str.lower().isin(["nan", "none"]), "")
    vals = [v for v in s.unique().tolist() if v]
    return vals[:limit]


def _looks_generic_category(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not col or col not in df.columns:
        return False
    vals = _distinct_non_empty_values(df[col], limit=50)
    if not vals:
        return False
    normed = {_normalize_header(v).replace("_", " ") for v in vals}
    generic_hits = sum(1 for v in normed if v in _GENERIC_CATEGORY_VALUES)
    return generic_hits >= max(1, int(0.6 * len(normed)))


def _detect_notes_merge_candidates(df: pd.DataFrame) -> List[str]:
    norm_cols = {_normalize_header(c): c for c in df.columns}
    hint_norms = [_normalize_header(x) for x in _WORKNOTES_MERGE_HINTS]

    candidates: List[str] = []
    for n, original in norm_cols.items():
        if any(h in n for h in hint_norms):
            candidates.append(original)

    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    create_combined_work_notes: bool = True,
    combined_work_notes_col: str = "work_notes__combined",
) -> Tuple[pd.DataFrame, CanonicalColumns]:
    cols = infer_columns(df)

    _maybe_discard_semantically_wrong_cols(cols)

    if not _is_viable_column(df, cols.assignment_group):
        cols.assignment_group = None
    if not _is_viable_column(df, cols.subcategory):
        cols.subcategory = None
    if not _is_viable_column(df, cols.priority):
        cols.priority = None

    if not cols.ticket_type and cols.category:
        cols.ticket_type = cols.category

    if not create_combined_work_notes:
        return df, cols

    candidates = _detect_notes_merge_candidates(df)
    if not candidates:
        return df, cols

    df2 = df.copy()

    def _combine_row(row: pd.Series) -> str:
        parts: List[str] = []
        for c in candidates:
            v = row.get(c)
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() in ("nan", "none"):
                continue
            parts.append(f"[{c}] {s}")
        return "\n".join(parts)

    df2[combined_work_notes_col] = df2.apply(_combine_row, axis=1)

    if (not cols.work_notes) or (cols.work_notes and not _is_viable_column(df2, cols.work_notes)):
        cols.work_notes = combined_work_notes_col

    return df2, cols


# -----------------------------------------------------------------------------
# LLM JSON parsing robustness
# -----------------------------------------------------------------------------
def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return t[start:end + 1]


def _call_vertex_llm(prompt: str) -> Dict[str, Any]:
    import vertexai  # type: ignore
    from vertexai.preview.generative_models import GenerativeModel  # type: ignore

    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = GenerativeModel(NORMALIZE_LLM_MODEL)
    text = (model.generate_content(prompt).text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    extracted = _extract_json_object(text)
    if extracted:
        try:
            return json.loads(extracted)
        except Exception:
            pass

    return {"raw_llm_output": text}


# -----------------------------------------------------------------------------
# LLM prompts
# -----------------------------------------------------------------------------
def _build_theme_prompt(items: List[Dict[str, str]]) -> str:
    payload = json.dumps(items, ensure_ascii=False)
    return f"""
You are labeling IT support tickets with a short, human-friendly THEME.

Rules:
- Return a short theme label (2-6 words) derived only from the supplied ticket.
- Do not copy or infer labels from these instructions.
- Do NOT use generic ITSM type words like "Incident" or "Service Request" as themes.
- Use "{_OTHER_BUCKET}" only if the content is truly unclear.
- Return STRICT JSON only, no markdown.

Input tickets (array):
{payload}

Return JSON in this exact shape:
{{"labels":[{{"id":"<id>","theme":"<short theme>"}}]}}
""".strip()


def _build_bucket_per_ticket_prompt(items: List[Dict[str, str]], max_buckets: int) -> str:
    """
    items: [{"id":"..","theme":"..","summary":"..","description":"..","work_notes":".."}, ...]
    """
    payload = json.dumps(items, ensure_ascii=False)
    return f"""
You are consolidating IT support tickets into reporting BUCKETS.

You will receive tickets with a short THEME and brief context.
Your job:
- Assign exactly ONE bucket per ticket.
- Bucket names must be short and clear (2-6 words).
- Do NOT use generic ITSM type words like "Incident" or "Service Request" as bucket names.
- Reuse bucket names across similar tickets (merge synonyms / near-duplicates).
- Use "{_OTHER_BUCKET}" only when truly unclear.
- Use at most {max_buckets} distinct bucket names total (excluding "{_OTHER_BUCKET}" which may appear).
- Return STRICT JSON only, no markdown.

Input tickets (array):
{payload}

Return JSON in this exact shape:
{{"labels":[{{"id":"<id>","bucket":"<bucket name>"}}]}}
""".strip()


def _build_bucket_only_prompt(items: List[Dict[str, str]], max_buckets: int) -> str:
    """
    Large-file seed mode: no theme field; bucket directly from context.
    items: [{"id":"..","summary":"..","description":"..","work_notes":".."}, ...]
    """
    payload = json.dumps(items, ensure_ascii=False)
    return f"""
You are consolidating IT support tickets into reporting BUCKETS.

You will receive tickets with brief context.
Your job:
- Assign exactly ONE bucket per ticket.
- Bucket names must be short and clear (2-6 words).
- Do NOT use generic ITSM type words like "Incident" or "Service Request" as bucket names.
- Reuse bucket names across similar tickets (merge synonyms / near-duplicates).
- Use "{_OTHER_BUCKET}" only when truly unclear.
- Use at most {max_buckets} distinct bucket names total (excluding "{_OTHER_BUCKET}" which may appear).
- Return STRICT JSON only, no markdown.

Input tickets (array):
{payload}

Return JSON in this exact shape:
{{"labels":[{{"id":"<id>","bucket":"<bucket name>"}}]}}
""".strip()


def _build_classify_to_known_buckets_prompt(items: List[Dict[str, str]], allowed_buckets: List[str]) -> str:
    """
    Strategy A: classify each ticket to exactly one of the already learned buckets.
    """
    payload = json.dumps(items, ensure_ascii=False)
    allowed_payload = json.dumps(allowed_buckets, ensure_ascii=False)
    return f"""
You are categorizing IT support tickets into one of the APPROVED BUCKETS.

Approved buckets (choose exactly ONE of these for each ticket):
{allowed_payload}

Rules:
- You MUST choose one of the approved bucket names exactly as written.
- Only use "{_OTHER_BUCKET}" if truly unclear even after reading context.
- Return STRICT JSON only, no markdown.

Input tickets (array):
{payload}

Return JSON in this exact shape:
{{"labels":[{{"id":"<id>","bucket":"<one approved bucket>"}}]}}
""".strip()


def _clean_theme(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return _OTHER_BUCKET
    if _normalize_header(t) in _GENERIC_CATEGORY_VALUES:
        return _OTHER_BUCKET
    if len(t) > 80:
        t = t[:80].rstrip()
    return t


def _clean_bucket(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return _OTHER_BUCKET
    if _normalize_header(t) in _GENERIC_CATEGORY_VALUES:
        return _OTHER_BUCKET
    if len(t) > 80:
        t = t[:80].rstrip()
    return t


def _derive_semantic_category_with_llm(
    df: pd.DataFrame,
    cols: CanonicalColumns,
    *,
    combined_work_notes_col: str,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Strategy A (dynamic, no hardcoding):
    1) Build a bucket taxonomy from a *seed sample* (up to NORMALIZE_LLM_MAX_ROWS).
    2) Classify ALL remaining rows into ONLY those learned buckets.
    This avoids the "only first N rows got categories" issue and keeps output consistent.
    """
    llm_meta: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "model": NORMALIZE_LLM_MODEL,
        "batch_size": NORMALIZE_LLM_BATCH_SIZE,
        "batches": 0,
        "row_limit_applied": False,  # now means "seed sample was used"
        "parse_failures": 0,
        "stage1_theme_labels": 0,
        "stage2_bucket_labels": 0,
        "max_buckets": NORMALIZE_LLM_MAX_BUCKETS,
        "raw_sample": None,
        "large_file_mode": False,
        "large_file_threshold": NORMALIZE_LLM_LARGE_FILE_THRESHOLD,
        "strategy": "A_seed_then_classify_all",
        "seed_rows": 0,
        "total_rows": int(len(df)),
    }

    if not _vertex_ready():
        llm_meta["enabled"] = False
        llm_meta["reason"] = "Vertex/LLM not configured or NORMALIZE_USE_LLM disabled."
        return pd.Series([_OTHER_BUCKET] * len(df), index=df.index), llm_meta

    ok, err = _try_import_vertex()
    if not ok:
        llm_meta["enabled"] = False
        llm_meta["reason"] = f"vertexai import failed: {err}"
        return pd.Series([_OTHER_BUCKET] * len(df), index=df.index), llm_meta

    def _get_col(c: Optional[str]) -> pd.Series:
        if c and c in df.columns:
            return df[c].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    summary_s = _get_col(cols.short_description)
    desc_s = _get_col(cols.description)
    notes_s = _get_col(cols.work_notes)
    combined_notes_s = _get_col(combined_work_notes_col)

    if cols.ticket_id and cols.ticket_id in df.columns:
        ids = df[cols.ticket_id].fillna("").astype(str)
        ids = ids.where(ids.str.strip().ne(""), df.index.astype(str))
    else:
        ids = df.index.astype(str)

    llm_meta["enabled"] = True
    llm_meta["reason"] = None

    bs = max(1, int(NORMALIZE_LLM_BATCH_SIZE))
    max_buckets = max(3, int(NORMALIZE_LLM_MAX_BUCKETS))

    # -------------------------
    # Seed sample (taxonomy builder)
    # -------------------------

    seed_n = max(1, int(NORMALIZE_LLM_MAX_ROWS))
    seed_df = df.head(seed_n) if len(df) > seed_n else df
    llm_meta["seed_rows"] = int(len(seed_df))
    llm_meta["row_limit_applied"] = bool(len(seed_df) < len(df))

    seed_large_mode = len(seed_df) > int(NORMALIZE_LLM_LARGE_FILE_THRESHOLD)
    llm_meta["large_file_mode"] = bool(seed_large_mode)

    # 1) Build buckets for seed
    out_seed_bucket: Dict[str, str] = {}

    if seed_large_mode:
        # Seed in bucket-only mode
        seed_items: List[Dict[str, str]] = []
        for i in range(len(seed_df)):
            seed_items.append({
                "id": str(ids.iloc[i]),
                "summary": str(summary_s.iloc[i])[:300],
                "description": str(desc_s.iloc[i])[:500],
                "work_notes": (str(notes_s.iloc[i]) + "\n" + str(combined_notes_s.iloc[i]))[:800],
            })

        total_batches = (len(seed_items) + bs - 1) // bs
        for bi, start in enumerate(range(0, len(seed_items), bs), start=1):
            batch = seed_items[start:start + bs]
            if progress_hook:
                progress_hook({
                    "stage": "seed_bucket_only",
                    "batch": bi,
                    "total_batches": total_batches,
                    "done": min(start, len(seed_items)),
                    "total": len(seed_items),
                })

            prompt = _build_bucket_only_prompt(batch, max_buckets=max_buckets)
            resp = _call_vertex_llm(prompt)
            llm_meta["batches"] += 1

            labels = resp.get("labels") if isinstance(resp, dict) else None
            if not isinstance(labels, list):
                llm_meta["parse_failures"] += 1
                continue

            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                tid = str(obj.get("id", "")).strip()
                bucket = _clean_bucket(str(obj.get("bucket", "")).strip())
                if tid:
                    out_seed_bucket[tid] = bucket

        llm_meta["stage1_theme_labels"] = 0
        llm_meta["stage2_bucket_labels"] = len(out_seed_bucket)

    else:
        # Seed in theme -> bucket mode
        seed_items: List[Dict[str, str]] = []
        for i in range(len(seed_df)):
            seed_items.append({
                "id": str(ids.iloc[i]),
                "summary": str(summary_s.iloc[i])[:500],
                "description": str(desc_s.iloc[i])[:800],
                "work_notes": (str(notes_s.iloc[i]) + "\n" + str(combined_notes_s.iloc[i]))[:1200],
            })

        out_theme: Dict[str, str] = {}
        total_batches_stage1 = (len(seed_items) + bs - 1) // bs
        for bi, start in enumerate(range(0, len(seed_items), bs), start=1):
            batch = seed_items[start:start + bs]
            if progress_hook:
                progress_hook({
                    "stage": "seed_theme",
                    "batch": bi,
                    "total_batches": total_batches_stage1,
                    "done": min(start, len(seed_items)),
                    "total": len(seed_items),
                })

            prompt = _build_theme_prompt(batch)
            resp = _call_vertex_llm(prompt)
            llm_meta["batches"] += 1

            if llm_meta["raw_sample"] is None and isinstance(resp, dict) and "raw_llm_output" in resp:
                llm_meta["raw_sample"] = str(resp.get("raw_llm_output"))[:800]

            labels = resp.get("labels") if isinstance(resp, dict) else None
            if not isinstance(labels, list):
                llm_meta["parse_failures"] += 1
                continue

            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                tid = str(obj.get("id", "")).strip()
                theme = _clean_theme(str(obj.get("theme", "")).strip())
                if tid:
                    out_theme[tid] = theme

        llm_meta["stage1_theme_labels"] = len(out_theme)

        if not out_theme:
            return pd.Series([_OTHER_BUCKET] * len(df), index=df.index), llm_meta

        seed_bucket_items: List[Dict[str, str]] = []
        for i in range(len(seed_df)):
            tid = str(ids.iloc[i])
            seed_bucket_items.append({
                "id": tid,
                "theme": out_theme.get(tid, _OTHER_BUCKET),
                "summary": str(summary_s.iloc[i])[:300],
                "description": str(desc_s.iloc[i])[:500],
                "work_notes": (str(notes_s.iloc[i]) + "\n" + str(combined_notes_s.iloc[i]))[:800],
            })

        total_batches_stage2 = (len(seed_bucket_items) + bs - 1) // bs
        for bi, start in enumerate(range(0, len(seed_bucket_items), bs), start=1):
            batch = seed_bucket_items[start:start + bs]
            if progress_hook:
                progress_hook({
                    "stage": "seed_bucket",
                    "batch": bi,
                    "total_batches": total_batches_stage2,
                    "done": min(start, len(seed_bucket_items)),
                    "total": len(seed_bucket_items),
                })

            prompt = _build_bucket_per_ticket_prompt(batch, max_buckets=max_buckets)
            resp = _call_vertex_llm(prompt)
            llm_meta["batches"] += 1

            labels = resp.get("labels") if isinstance(resp, dict) else None
            if not isinstance(labels, list):
                llm_meta["parse_failures"] += 1
                continue

            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                tid = str(obj.get("id", "")).strip()
                bucket = _clean_bucket(str(obj.get("bucket", "")).strip())
                if tid:
                    out_seed_bucket[tid] = bucket

        llm_meta["stage2_bucket_labels"] = len(out_seed_bucket)

    # Build allowed bucket list from seed (dynamic taxonomy)
    allowed = []
    seen = set()
    for b in out_seed_bucket.values():
        bb = _clean_bucket(b)
        if not bb or bb == _OTHER_BUCKET:
            continue
        if bb not in seen:
            seen.add(bb)
            allowed.append(bb)

    # If for any reason we learned nothing, fall back (still safe)
    if not allowed:
        final_fallback = []
        for i in range(len(df)):
            tid = str(ids.iloc[i])
            final_fallback.append(out_seed_bucket.get(tid, _OTHER_BUCKET))
        return pd.Series(final_fallback, index=df.index), llm_meta


    # -------------------------
    # 2) Classify ALL remaining rows into allowed buckets
    # -------------------------
    out_all: Dict[str, str] = dict(out_seed_bucket) # keep seed results

    start_idx = int(len(seed_df))
    remaining = len(df) - start_idx
    if remaining > 0:
        classify_items: List[Dict[str, str]] = []
        for i in range(start_idx, len(df)):
            classify_items.append({
                "id": str(ids.iloc[i]),
                "summary": str(summary_s.iloc[i])[:220],
                "description": str(desc_s.iloc[i])[:350],
                "work_notes": (str(notes_s.iloc[i]) + "\n" + str(combined_notes_s.iloc[i]))[:500],
            })

        total_batches = (len(classify_items) + bs - 1) // bs
        for bi, start in enumerate(range(0, len(classify_items), bs), start=1):
            batch = classify_items[start:start + bs]
            if progress_hook:
                progress_hook({
                    "stage": "classify_all",
                    "batch": bi,
                    "total_batches": total_batches,
                    "done": min(start, len(classify_items)),
                    "total": len(classify_items),
                })

            prompt = _build_classify_to_known_buckets_prompt(batch, allowed_buckets=allowed + [_OTHER_BUCKET])
            resp = _call_vertex_llm(prompt)
            llm_meta["batches"] += 1

            labels = resp.get("labels") if isinstance(resp, dict) else None
            if not isinstance(labels, list):
                llm_meta["parse_failures"] += 1
                continue

            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                tid = str(obj.get("id", "")).strip()
                bucket = _clean_bucket(str(obj.get("bucket", "")).strip())
                if tid:
                    # enforce allowed buckets, else fallback
                    if bucket != _OTHER_BUCKET and bucket not in allowed:
                        bucket = _OTHER_BUCKET
                    out_all[tid] = bucket
                    
    # Final series for entire df
    final: List[str] = []
    for i in range(len(df)):
        tid = str(ids.iloc[i])
        final.append(out_all.get(tid, _OTHER_BUCKET))

    return pd.Series(final, index=df.index), llm_meta


def normalize_ticket_dict(ticket: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ticket or {})
    key_map = {_normalize_header(k): k for k in out.keys()}

    def _get_by_syn(canonical: str) -> Optional[Any]:
        for syn in _COL_SYNONYMS.get(canonical, []):
            k_norm = _normalize_header(syn)
            if k_norm in key_map:
                return out.get(key_map[k_norm])
        return None

    if not out.get("ticket_id"):
        v = _get_by_syn("ticket_id")
        if v is not None:
            out["ticket_id"] = v

    if not out.get("status"):
        v = _get_by_syn("state")
        if v is not None:
            out["status"] = v

    if not out.get("type"):
        v = _get_by_syn("ticket_type")
        if v is not None:
            out["type"] = v

    if not out.get("urgency"):
        v = _get_by_syn("urgency")
        if v is not None:
            out["urgency"] = v

    if not out.get("impact"):
        v = _get_by_syn("impact")
        if v is not None:
            out["impact"] = v

    if not out.get("resolution"):
        v = _get_by_syn("resolution")
        if v is not None:
            out["resolution"] = v

    if not out.get("user_confirmation"):
        v = _get_by_syn("user_confirmation")
        if v is not None:
            out["user_confirmation"] = v

    if not out.get("summary") and not out.get("short_description"):
        v = _get_by_syn("short_description")
        if v is not None:
            out["summary"] = v
            out["short_description"] = v

    if not out.get("description"):
        v = _get_by_syn("description")
        if v is not None:
            out["description"] = v

    if not out.get("work_notes"):
        v = _get_by_syn("work_notes")
        if v is not None:
            out["work_notes"] = v

    return out


def build_normalization_report(
    df: pd.DataFrame,
    cols: CanonicalColumns,
    *,
    combined_work_notes_col: str = "work_notes__combined",
    notes_merge_candidates: Optional[List[str]] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mapping = dict(cols.__dict__)
    notes_merge_candidates = notes_merge_candidates or []
    llm_meta = llm_meta or {}

    synthesized: Dict[str, Any] = {}

    if combined_work_notes_col in df.columns and notes_merge_candidates:
        non_empty = int(df[combined_work_notes_col].fillna("").astype(str).str.strip().ne("").sum())
        synthesized[combined_work_notes_col] = {
            "from_columns": notes_merge_candidates,
            "non_empty_rows": non_empty,
        }

    if NORMALIZE_LLM_CATEGORY_COL in df.columns:
        non_empty = int(df[NORMALIZE_LLM_CATEGORY_COL].fillna("").astype(str).str.strip().ne("").sum())
        synthesized[NORMALIZE_LLM_CATEGORY_COL] = {
            "non_empty_rows": non_empty,
            "method": "llm_option_b_direct_bucket_per_ticket",
            "model": llm_meta.get("model"),
        }

    missing = [k for k, v in mapping.items() if v is None]

    viability: Dict[str, Any] = {}
    for k, col in mapping.items():
        if not col or col not in df.columns:
            continue
        try:
            viability[k] = {"column": col, "non_empty_ratio": round(_non_empty_ratio(df[col]), 4)}
        except Exception:
            continue

    return {
        "input": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
        },
        "detected_mapping": mapping,
        "viability": viability,
        "synthesized_columns": synthesized,
        "notes_merge_candidates": notes_merge_candidates,
        "llm_category_derivation": llm_meta,
        "missing_canonical_fields": missing,
    }


def normalize_dataframe_with_report(
    df: pd.DataFrame,
    *,
    create_combined_work_notes: bool = True,
    combined_work_notes_col: str = "work_notes__combined",
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, CanonicalColumns, Dict[str, Any]]:
    notes_candidates = _detect_notes_merge_candidates(df) if create_combined_work_notes else []

    df2, cols = normalize_dataframe(
        df,
        create_combined_work_notes=create_combined_work_notes,
        combined_work_notes_col=combined_work_notes_col,
    )

    llm_meta: Dict[str, Any] = {"enabled": False, "used": False, "reason": None, "model": NORMALIZE_LLM_MODEL}

    # - If forced, always derive semantic category
    # - Else if category missing or non-viable, derive semantic category
    # - Else only derive when category "looks generic"
    category_missing_or_bad = (not cols.category) or (cols.category and not _is_viable_column(df2, cols.category))

    should_derive = bool(
        NORMALIZE_FORCE_SEMANTIC_CATEGORY
        or category_missing_or_bad
        or (cols.category and _looks_generic_category(df2, cols.category))
    )

    if should_derive:
        series, meta = _derive_semantic_category_with_llm(
            df2,
            cols,
            combined_work_notes_col=combined_work_notes_col,
            progress_hook=progress_hook,
        )
        llm_meta = dict(meta)
        semantic_values = series.fillna("").astype(str).str.strip()
        semantic_has_meaningful_values = bool(
            semantic_values
            .loc[
                ~semantic_values.str.lower().isin(
                    {"", "other", "unclear", "other / unclear", "(blank)"}
                )
            ]
            .any()
        )
        semantic_is_usable = bool(
            meta.get("enabled") and semantic_has_meaningful_values
        )
        if semantic_is_usable:
            df2 = df2.copy()
            df2[NORMALIZE_LLM_CATEGORY_COL] = series
            cols.category = NORMALIZE_LLM_CATEGORY_COL
            llm_meta["used"] = True
            if NORMALIZE_FORCE_SEMANTIC_CATEGORY and meta.get("reason") is None:
                llm_meta["reason"] = "Forced semantic category derivation via NORMALIZE_FORCE_SEMANTIC_CATEGORY."
        else:
            llm_meta["used"] = False
            llm_meta["reason"] = (
                meta.get("reason")
                or "Semantic category derivation returned no usable values; preserved the detected source category."
            )
    else:
        llm_meta["enabled"] = _vertex_ready()
        llm_meta["used"] = False
        llm_meta["reason"] = "Category not generic; semantic derivation not needed."

    report = build_normalization_report(
        df2,
        cols,
        combined_work_notes_col=combined_work_notes_col,
        notes_merge_candidates=notes_candidates,
        llm_meta=llm_meta,
    )
    return df2, cols, report
