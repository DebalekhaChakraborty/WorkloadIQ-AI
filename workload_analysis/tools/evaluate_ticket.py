import os
from typing import Dict, Any, List, Tuple
import json
import re

from vertexai.preview.generative_models import (  # type: ignore
    GenerativeModel,
    GenerationConfig,
)
from google.adk.tools import FunctionTool, ToolContext  # type: ignore
from .normalize_columns import normalize_ticket_dict  # type: ignore


TICKET_QA_MODEL = os.getenv("TICKET_QA_MODEL", "gemini-2.5-flash")
TICKET_QA_PASS_THRESHOLD = int(os.getenv("TICKET_QA_PASS_THRESHOLD", "70"))
TICKET_QA_FATAL_THRESHOLD = int(os.getenv("TICKET_QA_FATAL_THRESHOLD", "0"))  # kept for future use, not directly used right now



# ---------------------------------------------------------------------------
# Canonical parameter specs – single source of truth
# ---------------------------------------------------------------------------

PARAM_SPECS: List[Dict[str, Any]] = [
    # Ticket documentation (25)
    {
        "id": "DOC_ISSUE_TYPE",
        "section": "ticket_documentation",
        "label": "Correct Issue Type (INC or SR)",
        "max_points": 5,
        "include_in_total": True,
    },
    {
        "id": "DOC_SUMMARY_DESCRIPTION",
        "section": "ticket_documentation",
        "label": "Std. Summary & Description updated",
        "max_points": 8,
        "include_in_total": True,
    },
    {
        "id": "DOC_WORK_NOTES",
        "section": "ticket_documentation",
        "label": "Work notes updated",
        "max_points": 6,
        "include_in_total": True,
    },
    {
        "id": "DOC_STATUS_USAGE",
        "section": "ticket_documentation",
        "label": "Use of Proper Status with relevant comment",
        "max_points": 4,
        "include_in_total": True,
    },
    {
        "id": "DOC_REASSIGNMENT",
        "section": "ticket_documentation",
        "label": "Correct Reassignment or Not",
        "max_points": 2,
        "include_in_total": True,
    },

    # Ticket priority (10)
    {
        "id": "PRI_IMPACT",
        "section": "ticket_priority",
        "label": "Impact tagged",
        "max_points": 5,
        "include_in_total": True,
    },
    {
        "id": "PRI_URGENCY",
        "section": "ticket_priority",
        "label": "Urgency tagged",
        "max_points": 5,
        "include_in_total": True,
    },

    # Ticket categorization (20)
    {
        "id": "CAT_CUSTOMER_REQUEST_TYPE",
        "section": "ticket_categorization",
        "label": "Correct Customer Request Type",
        "max_points": 5,
        "include_in_total": True,
    },
    {
        "id": "CAT_ON_HOLD_WIP",
        "section": "ticket_categorization",
        "label": "On hold/Work in Progress",
        "max_points": 3,
        "include_in_total": True,
    },
    {
        "id": "CAT_CATEGORY",
        "section": "ticket_categorization",
        "label": "Correct Category",
        "max_points": 6,
        "include_in_total": True,
    },
    {
        "id": "CAT_SUBCATEGORY",
        "section": "ticket_categorization",
        "label": "Correct Sub-category",
        "max_points": 4,
        "include_in_total": True,
    },
    {
        "id": "CAT_NEXTHINK_USAGE",
        "section": "ticket_categorization",
        "label": "Nexthink tool usage",
        "max_points": 2,
        "include_in_total": True,
    },

    # Resolution confirmation (30) – fatal is a gate only
    {
        "id": "RES_FIX_STEPS",
        "section": "resolution_confirmation",
        "label": "Resolution fix with updated steps",
        "max_points": 12,
        "include_in_total": True,
    },
    {
        "id": "RES_USER_CONFIRMATION",
        "section": "resolution_confirmation",
        "label": "Confirmation from User taken",
        "max_points": 8,
        "include_in_total": True,
    },
    {
        "id": "RES_KB_ATTACHED",
        "section": "resolution_confirmation",
        "label": "KB Attached",
        "max_points": 5,
        "include_in_total": True,
    },
    {
        "id": "RES_KB_CORRECT",
        "section": "resolution_confirmation",
        "label": "Correct KB attached or not",
        "max_points": 5,
        "include_in_total": True,
    },
    # Non-scoring gate: fatal error
    {
        "id": "RES_FATAL_ERROR",
        "section": "resolution_confirmation",
        "label": "Fatal Error gate (non-scoring)",
        "max_points": 0,
        "include_in_total": False,
    },

    # Soft skills (15)
    {
        "id": "SOFT_GRAMMAR",
        "section": "soft_skills",
        "label": "Grammatical mistakes",
        "max_points": 5,
        "include_in_total": True,
    },
    {
        "id": "SOFT_EMPATHY",
        "section": "soft_skills",
        "label": "Empathy and acknowledged throughout the call/chat",
        "max_points": 4,
        "include_in_total": True,
    },
    {
        "id": "SOFT_DEAD_AIR",
        "section": "soft_skills",
        "label": "Dead air/time appropriately during Call/Chat",
        "max_points": 3,
        "include_in_total": True,
    },
    {
        "id": "SOFT_HOLD_PROCEDURE",
        "section": "soft_skills",
        "label": "Hold procedure appropriately for Call/Chat",
        "max_points": 3,
        "include_in_total": True,
    },
]

PARAM_SPECS_BY_ID = {spec["id"]: spec for spec in PARAM_SPECS}
TOTAL_MAX_POINTS = sum(spec["max_points"] for spec in PARAM_SPECS if spec["include_in_total"])

# Which parameters are RULE-BASED vs LLM-BASED
RULE_PARAM_IDS = {
    "DOC_ISSUE_TYPE",
    "PRI_IMPACT",
    "PRI_URGENCY",
    "RES_KB_ATTACHED",
}
LLM_PARAM_IDS = {spec["id"] for spec in PARAM_SPECS} - RULE_PARAM_IDS


# ---------------------------------------------------------------------------
# Rule-based scoring helpers
# ---------------------------------------------------------------------------

def _rule_score_issue_type(ticket: Dict[str, Any]) -> Tuple[int, bool, str]:
    spec = PARAM_SPECS_BY_ID["DOC_ISSUE_TYPE"]
    max_points = spec["max_points"]
    ttype = (ticket.get("type") or "").strip().upper()

    if not ttype:
        return 0, True, "Issue type is missing; expected 'INC' or 'SR'."

    if ttype in ("INC", "SR"):
        return max_points, False, f"Issue type '{ttype}' is valid."
    else:
        return 0, True, f"Issue type '{ttype}' is not one of the expected values (INC or SR)."


def _rule_score_impact(ticket: Dict[str, Any]) -> Tuple[int, bool, str]:
    spec = PARAM_SPECS_BY_ID["PRI_IMPACT"]
    max_points = spec["max_points"]
    impact = (ticket.get("impact") or "").strip().lower()

    if not impact:
        return 0, True, "Impact is not tagged."
    return max_points, False, f"Impact is tagged as '{impact}'."


def _rule_score_urgency(ticket: Dict[str, Any]) -> Tuple[int, bool, str]:
    spec = PARAM_SPECS_BY_ID["PRI_URGENCY"]
    max_points = spec["max_points"]
    urgency = (ticket.get("urgency") or "").strip().lower()

    if not urgency:
        return 0, True, "Urgency is not tagged."
    return max_points, False, f"Urgency is tagged as '{urgency}'."


def _rule_score_kb_attached(ticket: Dict[str, Any]) -> Tuple[int, bool, str]:
    spec = PARAM_SPECS_BY_ID["RES_KB_ATTACHED"]
    max_points = spec["max_points"]
    kb_ids = ticket.get("kb_ids_attached") or ticket.get("kb_ids") or []

    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]

    if kb_ids and len(kb_ids) > 0:
        return max_points, False, f"KB attached: {kb_ids}."
    else:
        return 0, True, "No KB is attached to the ticket."


def _rule_detect_fatal(ticket: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Simple hard fatal rules:
      - Final status is Resolved/Closed but resolution is empty/very short.
      - (You can add more later if needed.)
    """
    fatal_found = False
    reasons: List[str] = []

    status = (ticket.get("status") or "").strip().lower()
    resolution = (ticket.get("resolution") or "").strip()
    impact = (ticket.get("impact") or "").strip().lower()

    if status in ("resolved", "closed"):
        if not resolution or len(resolution) < 20:
            fatal_found = True
            reasons.append(
                "Ticket is marked as Resolved/Closed but the resolution field is empty or too short."
            )

    # Optional: high impact but no confirmation field at all
    user_conf = (ticket.get("user_confirmation") or "").strip().lower()
    if status in ("resolved", "closed") and impact in ("high", "critical", "p1", "p2"):
        if not user_conf:
            fatal_found = True
            reasons.append(
                "High-impact ticket appears Closed/Resolved without explicit user confirmation."
            )

    return fatal_found, reasons


def _rule_based_scoring(ticket: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], bool, List[str]]:
    """
    Compute scores for all RULE_PARAM_IDS and apply rule-based fatal triggers.

    Returns:
      - rule_param_scores: { param_id: {score, issue_flag, reason} }
      - rule_fatal_found: bool
      - rule_fatal_reasons: list[str]
    """
    scores: Dict[str, Dict[str, Any]] = {}

    # DOC_ISSUE_TYPE
    s, flag, reason = _rule_score_issue_type(ticket)
    scores["DOC_ISSUE_TYPE"] = {"score": s, "issue_flag": flag, "reason": reason}

    # PRI_IMPACT
    s, flag, reason = _rule_score_impact(ticket)
    scores["PRI_IMPACT"] = {"score": s, "issue_flag": flag, "reason": reason}

    # PRI_URGENCY
    s, flag, reason = _rule_score_urgency(ticket)
    scores["PRI_URGENCY"] = {"score": s, "issue_flag": flag, "reason": reason}

    # RES_KB_ATTACHED
    s, flag, reason = _rule_score_kb_attached(ticket)
    scores["RES_KB_ATTACHED"] = {"score": s, "issue_flag": flag, "reason": reason}

    # Fatal gates (rule-based)
    fatal_found, fatal_reasons = _rule_detect_fatal(ticket)

    return scores, fatal_found, fatal_reasons


# ---------------------------------------------------------------------------
# LLM prompt – only for LLM_PARAM_IDS
# ---------------------------------------------------------------------------

def _build_llm_param_instructions() -> str:
    lines: List[str] = []
    for spec in PARAM_SPECS:
        pid = spec["id"]
        if pid not in LLM_PARAM_IDS:
            continue
        lines.append(f"- {pid} ({spec['max_points']} points): {spec['label']}")
    return "\n".join(lines)


_LLM_PARAM_INSTRUCTIONS = _build_llm_param_instructions()

_QA_SYSTEM_PROMPT = f"""
You are an AI Ticket Quality Auditor for an IT Service Desk.

You receive a single IT support ticket as structured JSON.
Fields may include (examples, not strict):
- ticket_id
- type ("INC" or "SR")
- summary
- description
- work_notes
- status, status_history
- priority, impact, urgency
- customer_request_type
- category, sub_category
- tools_used (e.g., Nexthink)
- resolution
- user_confirmation
- kb_ids_attached
- transcript_or_call_notes
- agent_name

Some parameters are scored by RULES in code already (for example: DOC_ISSUE_TYPE, PRI_IMPACT, PRI_URGENCY, RES_KB_ATTACHED).
Your job is to score ONLY the remaining parameters listed below and add semantic fatal checks.

LLM-SCORED PARAMETERS (IDs and max points):

{_LLM_PARAM_INSTRUCTIONS}

SCORING RULES (per parameter):
- For each parameter, assign an integer "score" between 0 and its max_points.
- Use the full range:
  - Full points = fully meets expectation.
  - Medium points = partially meets.
  - Zero = does not meet or clearly poor.
- Set "issue_flag" = true whenever:
  - score < max_points, OR
  - there is something noteworthy for QA (unusual but acceptable behaviour).
- Provide a short textual "reason" describing WHY you gave that score (keep each reason under 20 words).

FATAL LOGIC (RES_FATAL_ERROR):
- This parameter has max_points = 0 and does NOT change the numeric score directly.
- Use RES_FATAL_ERROR to capture SEVERE issues, such as:
  - Wrong or incomplete resolution.
  - Ticket closed without real fix.
  - Serious process violation likely to cause a reopen.
  - Closing without any attempt to confirm resolution for a serious impact issue.
- If such a severe issue exists:
  - In "parameter_scores", for id = "RES_FATAL_ERROR":
      issue_flag = true, score must remain 0, and reason must clearly explain the severe issue.
  - Set top-level "fatal_found" = true.
  - Add a human-readable explanation to "fatal_reasons" array.
- Do NOT set fatal_found = true for minor documentation or soft-skill problems.

IMPORTANT: OUTPUT FORMAT

You MUST respond with ONLY a single JSON-like object, nothing else.
Do NOT use markdown fences like ```json.
Do NOT add any commentary outside the braces.

Expected structure:

{{
  "parameter_scores": [
    {{
      "id": "<one of the LLM parameter IDs above>",
      "score": <integer>,
      "issue_flag": <true or false>,
      "reason": "<short explanation>"
    }},
    ...
  ],
  "fatal_found": <true or false>,
  "fatal_reasons": ["<reason1>", "<reason2>", ...],
  "summary_feedback": "<1-3 sentence overall feedback for the ticket>"
}}
"""


# ---------------------------------------------------------------------------
# LLM output parser (robust + truncation detection)
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: str) -> Tuple[Dict[str, Dict[str, Any]], bool, List[str], str, bool]:
    """
    Parse the LLM's raw text output in a best-effort way.

    We do NOT rely on strict JSON. Instead:
      - Extract each parameter block with id/score/issue_flag/reason via regex.
      - Extract fatal_found, fatal_reasons, summary_feedback via regex.
      - Detect likely truncation (more '{' than '}' or obvious cut).
    """
    llm_param_scores: Dict[str, Dict[str, Any]] = {}

    # Heuristic truncation detection: unmatched braces or obvious cut
    opens = raw.count("{")
    closes = raw.count("}")
    truncation_suspected = opens > closes

    # Also, if it literally ends in an incomplete word or comma
    if re.search(r'"issue_flag\s*$', raw) or raw.strip().endswith(":") or raw.strip().endswith(","):
        truncation_suspected = True

    # Find each object that starts with "id": "..."
    param_pattern = re.compile(
        r'\{\s*"id"\s*:\s*"(?P<id>[^"]+)"(?P<body>.*?)\}',
        re.DOTALL | re.IGNORECASE,
    )

    for m in param_pattern.finditer(raw):
        pid = m.group("id")
        if pid not in LLM_PARAM_IDS:
            continue

        body = m.group("body")

        score_match = re.search(r'"score"\s*:\s*(\d+)', body)
        issue_match = re.search(r'"issue_flag"\s*:\s*(true|false)', body, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', body)

        score = int(score_match.group(1)) if score_match else 0
        issue_flag = bool(issue_match and issue_match.group(1).lower() == "true")
        reason = reason_match.group(1) if reason_match else ""

        llm_param_scores[pid] = {
            "score": score,
            "issue_flag": issue_flag,
            "reason": reason,
        }

    # fatal_found
    fatal_found = False
    fatal_m = re.search(r'"fatal_found"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if fatal_m:
        fatal_found = fatal_m.group(1).lower() == "true"

    # fatal_reasons
    fatal_reasons: List[str] = []
    fr_m = re.search(r'"fatal_reasons"\s*:\s*\[(.*?)\]', raw, re.DOTALL | re.IGNORECASE)
    if fr_m:
        inner = fr_m.group(1)
        for s in re.finditer(r'"([^"]*)"', inner):
            fatal_reasons.append(s.group(1))

    # summary_feedback
    summary_feedback = ""
    sum_m = re.search(r'"summary_feedback"\s*:\s*"([^"]*)"', raw)
    if not sum_m:
        sum_m = re.search(r'"summary"\s*:\s*"([^"]*)"', raw)
    if not sum_m:
        sum_m = re.search(r'"feedback"\s*:\s*"([^"]*)"', raw)
    if sum_m:
        summary_feedback = sum_m.group(1)

    return llm_param_scores, fatal_found, fatal_reasons, summary_feedback, truncation_suspected


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

def evaluate_ticket(ticket: Dict[str, Any], tool_context: ToolContext | None = None) -> Dict[str, Any]:
    """
    Hybrid evaluation: rule-based + LLM-based.

    Returns:
      - ticket_id
      - overall_score (0-100)
      - verdict ("Pass" or "Fail")
      - fatal_found (bool)
      - fatal_reasons (list[str])
      - summary_feedback (str)
      - llm_truncation_suspected (bool)
      - llm_missing_params (list[str])
      - llm_param_coverage_ok (bool)
      - sections: { section_name: {score, max_points} }
      - parameters: list of per-parameter dicts
    """
    # Normalize common field names across dump formats (Jira/ServiceNow/etc.)
    ticket = normalize_ticket_dict(ticket)


    # 1) Rule-based scoring
    rule_param_scores, rule_fatal_found, rule_fatal_reasons = _rule_based_scoring(ticket)

    # -----------------------------------------------------------------------
    # 2) LLM scoring for semantic/subjective parameters, with one retry
    # -----------------------------------------------------------------------

    def _single_llm_pass(ticket_payload: Dict[str, Any]) -> Tuple[
        Dict[str, Dict[str, Any]], bool, List[str], str, bool, List[str]
    ]:
        """
        Run a single LLM evaluation pass and return:
        - llm_param_scores
        - llm_fatal_found
        - llm_fatal_reasons
        - summary_feedback
        - truncation_suspected
        - missing_llm_params
        """
        model = GenerativeModel(TICKET_QA_MODEL)
        payload = {
            "ticket": ticket_payload,
        }

        resp = model.generate_content(
            [
                _QA_SYSTEM_PROMPT,
                "\nHere is the ticket payload to evaluate:\n",
                json.dumps(payload, ensure_ascii=False),
            ],
            generation_config=GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=8192,  # increase output size to reduce truncation
                response_mime_type="text/plain",  # text; we'll parse ourselves
            ),
        )

        raw = ""
        try:
            raw = (resp.text or "").strip()  # type: ignore[attr-defined]
        except Exception:
            try:
                if resp.candidates and resp.candidates[0].content.parts:
                    part = resp.candidates[0].content.parts[0]
                    if hasattr(part, "text"):
                        raw = (part.text or "").strip()  # type: ignore
                    else:
                        raw = json.dumps(part, ensure_ascii=False)
            except Exception:
                raw = ""

        # DEBUG: Keeping to check LLM output during tuning
        # print("\n\n===== DEBUG: RAW LLM OUTPUT =====")
        # print(raw)
        # print("===== END DEBUG =====\n\n")

        if not raw:
            llm_param_scores_local: Dict[str, Dict[str, Any]] = {}
            llm_fatal_found_local = False
            llm_fatal_reasons_local: List[str] = []
            summary_feedback_local = (
                "Evaluation partially succeeded: rule-based checks ran, "
                "but LLM did not return any content."
            )
            truncation_suspected_local = False
        else:
            (
                llm_param_scores_local,
                llm_fatal_found_local,
                llm_fatal_reasons_local,
                summary_feedback_local,
                truncation_suspected_local,
            ) = _parse_llm_output(raw)

            if not summary_feedback_local:
                summary_feedback_local = (
                    "Ticket evaluated, but no clear overall feedback was provided by the model."
                )

        missing_llm_params_local = [
            pid for pid in LLM_PARAM_IDS if pid not in llm_param_scores_local
        ]

        return (
            llm_param_scores_local,
            llm_fatal_found_local,
            llm_fatal_reasons_local,
            summary_feedback_local,
            truncation_suspected_local,
            missing_llm_params_local,
        )

    # First LLM pass
    (
        llm_param_scores,
        llm_fatal_found,
        llm_fatal_reasons,
        summary_feedback,
        truncation_suspected,
        missing_llm_params,
    ) = _single_llm_pass(ticket)

    # Optional retry if coverage looks bad
    if truncation_suspected or missing_llm_params:
        (
            llm_param_scores2,
            llm_fatal_found2,
            llm_fatal_reasons2,
            summary_feedback2,
            truncation_suspected2,
            missing_llm_params2,
        ) = _single_llm_pass(ticket)

        # Prefer the pass with better coverage (no truncation, fewer missing params)
        def _coverage_key(trunc: bool, missing_count: int) -> Tuple[bool, int]:
            # True > False for "not truncated", and fewer missing is better (use negative)
            return (not trunc, -missing_count)

        if _coverage_key(truncation_suspected2, len(missing_llm_params2)) > _coverage_key(
            truncation_suspected, len(missing_llm_params)
        ):
            llm_param_scores = llm_param_scores2
            llm_fatal_found = llm_fatal_found2
            llm_fatal_reasons = llm_fatal_reasons2
            summary_feedback = summary_feedback2
            truncation_suspected = truncation_suspected2
            missing_llm_params = missing_llm_params2

    # C) If truncation suspected or some LLM params are missing, add a soft warning
    if truncation_suspected or missing_llm_params:
        warning_bits = []
        if truncation_suspected:
            warning_bits.append("LLM output may have been truncated")
        if missing_llm_params:
            warning_bits.append(f"missing scores for {len(missing_llm_params)} parameters")
        warning_text = "Note: " + "; ".join(warning_bits) + ". Some fields may be defaulted to 0."
        if summary_feedback:
            summary_feedback = summary_feedback.rstrip(".") + ". " + warning_text
        else:
            summary_feedback = warning_text

    # -----------------------------------------------------------------------
    # Merge rule-based and LLM-based parameter scores into a canonical list
    # -----------------------------------------------------------------------
    enriched_params: List[Dict[str, Any]] = []
    section_agg: Dict[str, Dict[str, float]] = {}

    for spec in PARAM_SPECS:
        pid = spec["id"]
        section = spec["section"]
        max_points = int(spec["max_points"])
        include_in_total = bool(spec.get("include_in_total", True))

        if pid in RULE_PARAM_IDS:
            src = rule_param_scores.get(pid, {"score": 0, "issue_flag": True, "reason": "No rule score found."})
            score = int(src.get("score", 0) or 0)
            issue_flag = bool(src.get("issue_flag", False))
            reason = src.get("reason") or ""
        else:
            p_data = llm_param_scores.get(pid, {})
            score = int(p_data.get("score", 0) or 0)
            issue_flag = bool(p_data.get("issue_flag", False))
            reason = p_data.get("reason") or ""
            if pid not in llm_param_scores:
                issue_flag = True
                if not reason:
                    reason = "LLM did not return this parameter; treated as 0 for safety."

        # Clamp scores
        if score < 0:
            score = 0
        if score > max_points:
            score = max_points

        # Aggregate section scores
        if include_in_total:
            if section not in section_agg:
                section_agg[section] = {"score": 0.0, "max_points": 0.0}
            section_agg[section]["score"] += score
            section_agg[section]["max_points"] += max_points

        enriched_params.append(
            {
                "id": pid,
                "section": section,
                "label": spec["label"],
                "max_points": max_points,
                "score": score,
                "issue_flag": issue_flag,
                "reason": reason,
            }
        )

    # -----------------------------------------------------------------------
    # Compute section scores and overall score
    # -----------------------------------------------------------------------
    sections: Dict[str, Dict[str, Any]] = {}
    total_score = 0.0
    total_max = 0.0

    for section, agg in section_agg.items():
        sec_score = float(agg["score"])
        sec_max = float(agg["max_points"]) if agg["max_points"] else 0.0
        sections[section] = {
            "score": round(sec_score, 2),
            "max_points": sec_max,
        }
        total_score += sec_score
        total_max += sec_max

    if total_max <= 0:
        overall_score = 0
    else:
        overall_score = int(round((total_score / total_max) * 100))

    # -----------------------------------------------------------------------
    # Combine fatal flags and reasons
    # -----------------------------------------------------------------------
    fatal_found = bool(rule_fatal_found or llm_fatal_found)
    fatal_reasons: List[str] = []
    fatal_reasons.extend(rule_fatal_reasons)
    fatal_reasons.extend(llm_fatal_reasons)

    # -----------------------------------------------------------------------
    # Final deterministic verdict
    # -----------------------------------------------------------------------
    if fatal_found:
        verdict = "Fail"
    elif overall_score >= TICKET_QA_PASS_THRESHOLD:
        verdict = "Pass"
    else:
        verdict = "Fail"

    llm_param_coverage_ok = not (truncation_suspected or missing_llm_params)

    result = {
        "ticket_id": ticket.get("ticket_id"),
        "overall_score": overall_score,
        "verdict": verdict,
        "fatal_found": fatal_found,
        "fatal_reasons": fatal_reasons,
        "summary_feedback": summary_feedback,
        "llm_truncation_suspected": truncation_suspected,
        "llm_missing_params": missing_llm_params,
        "llm_param_coverage_ok": llm_param_coverage_ok,
        "sections": sections,
        "parameters": enriched_params,
    }

    # Optional: keep this while tuning
    # print("[DEBUG evaluate_ticket]", json.dumps(result, indent=2))

    return result


evaluate_ticket_tool = FunctionTool(evaluate_ticket)
