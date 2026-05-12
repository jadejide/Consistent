from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
REVIEW_CSV = DATA_DIR / "review_samples.csv"
APP_VERSION = "single_history_score_aligned_2026_05_12"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
QUESTION_TASKS = {"KP_Weak", "KP_Stage", "Planning_Progress", "Planning_Target"}
ENTITY_TASKS = {"Personality_Individual", "Personality_Group"}
ALL_BENCHMARKS = QUESTION_TASKS | ENTITY_TASKS

TASK_LABELS = {
    "KP_Weak": "知识匹配：薄弱知识匹配",
    "KP_Stage": "知识匹配：教学进度知识匹配",
    "Planning_Progress": "规划适配：学习进程适配",
    "Planning_Target": "规划适配：学习目标适配",
    "Personality_Individual": "个性区分：个体区分",
    "Personality_Group": "个性区分：群体区分",
}

TASK_GUIDANCE = {
    "KP_Weak": {
        "goal": "根据学生历史作答表现，选择最适合用于针对性补弱的候选题。",
        "basis": ["优先关注历史中暴露出的薄弱知识点。", "候选题应能针对关键薄弱点练习。", "选择最值得优先推荐的一题。"],
    },
    "KP_Stage": {
        "goal": "根据学生历史题目与表现，选择最符合当前教学章节 / 知识阶段的候选题。",
        "basis": ["重点判断候选题是否符合当前教学章节。", "不要只看难度高低。", "明显过早或过晚的题不应优先推荐。"],
    },
    "Planning_Progress": {
        "goal": "根据学生最近学习轨迹，选择最适合作为下一步练习的候选题。",
        "basis": ["判断学生当前更适合推进、巩固，还是回退修复。", "避免过难题或无效重复题。", "从教学推进角度选择最合理的一题。"],
    },
    "Planning_Target": {
        "goal": "在给定学习目标下，选择最能推进该目标的候选题。",
        "basis": ["显式学习目标是核心条件。", "候选题应真正服务学习目标。", "不要只看与历史题的表面相似。"],
    },
    "Personality_Individual": {
        "goal": "给定目标题目，从候选学生中选择最适合推荐这道题的学生。",
        "basis": ["比较候选学生历史表现与目标题所需能力。", "不要只看总体成绩。", "判断谁最适合这道具体题目。"],
    },
    "Personality_Group": {
        "goal": "给定目标题目，从候选学生组中选择最适合推荐这道题的学生组。",
        "basis": ["比较小组整体能力结构与目标题的匹配程度。", "关注组内成员结构，不只看单个成员。", "判断哪个组最适合这道具体题目。"],
    },
}

SUP = str.maketrans("0123456789+-=()nix", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣ")
SUB = str.maketrans("0123456789+-=()nix", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙᵢₓ")

REQUIRED_COLUMNS = [
    "row_id",
    "benchmark",
    "task_type",
    "task_label",
    "case_type",
    "sample_rank_in_case",
    "best_model",
    "sample_id",
    "dataset",
    "source",
    "setting",
    "teacher_check_focus",
    "history_items_json",
    "candidate_items_json",
    "learning_goal",
    "progress_state",
    "weak_knowledge_points",
    "stage_knowledge_points",
    "target_question_json",
    "target_question_text",
    "candidate_entities_json",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_list(value: Any, field: str, row_id: str) -> list[dict[str, Any]]:
    text = clean(value)
    require(bool(text), f"row_id={row_id} 字段 {field} 不能为空")
    obj = json.loads(text)
    require(isinstance(obj, list), f"row_id={row_id} 字段 {field} 必须是 JSON list")
    return obj


def json_dict(value: Any, field: str, row_id: str) -> dict[str, Any]:
    text = clean(value)
    require(bool(text), f"row_id={row_id} 字段 {field} 不能为空")
    obj = json.loads(text)
    require(isinstance(obj, dict), f"row_id={row_id} 字段 {field} 必须是 JSON dict")
    return obj


def find_csv() -> Path:
    require(REVIEW_CSV.exists(), f"找不到 {REVIEW_CSV}。请把规范化后的 review_samples.csv 放到 data/review_samples.csv")
    return REVIEW_CSV


def load_samples() -> pd.DataFrame:
    df = pd.read_csv(find_csv(), dtype={"row_id": "string"})
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    require(not missing, "review_samples.csv 缺少必需字段：" + ", ".join(missing))
    df["row_id"] = df["row_id"].astype(str)
    validate_dataframe(df)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    invalid = sorted(set(df["benchmark"].astype(str)) - ALL_BENCHMARKS)
    require(not invalid, "存在未知 benchmark：" + ", ".join(invalid))

    for _, row in df.iterrows():
        row_id = clean(row["row_id"])
        benchmark = clean(row["benchmark"])
        require(bool(row_id), "row_id 不能为空")

        if benchmark in QUESTION_TASKS:
            require(clean(row["task_type"]) == "question_recommendation", f"row_id={row_id} task_type 应为 question_recommendation")
            history = json_list(row["history_items_json"], "history_items_json", row_id)
            candidates = json_list(row["candidate_items_json"], "candidate_items_json", row_id)
            require(len(history) > 0, f"row_id={row_id} history_items_json 不能为空列表")
            require(len(candidates) > 0, f"row_id={row_id} candidate_items_json 不能为空列表")
            for item in candidates:
                require("is_gt" not in item, f"row_id={row_id} candidate_items_json 不能包含 is_gt")
                require(clean(item.get("label")), f"row_id={row_id} 候选题缺少 label")
                require(clean(item.get("question_text")), f"row_id={row_id} 候选题缺少 question_text")
            if benchmark == "Planning_Target":
                require(clean(row["learning_goal"]), f"row_id={row_id} Planning_Target 必须有 learning_goal")
            if benchmark == "Planning_Progress":
                require(clean(row["progress_state"]), f"row_id={row_id} Planning_Progress 必须有 progress_state")

        if benchmark in ENTITY_TASKS:
            require(clean(row["task_type"]) == "entity_recommendation", f"row_id={row_id} task_type 应为 entity_recommendation")
            target = json_dict(row["target_question_json"], "target_question_json", row_id)
            entities = json_list(row["candidate_entities_json"], "candidate_entities_json", row_id)
            require(clean(target.get("question_text")), f"row_id={row_id} target_question_json 缺少 question_text")
            require(len(entities) >= 2, f"row_id={row_id} candidate_entities_json 候选数必须 >= 2")
            for entity in entities:
                require("is_gt" not in entity, f"row_id={row_id} candidate_entities_json 不能包含 is_gt")
                require(clean(entity.get("label")), f"row_id={row_id} 候选实体缺少 label")
                require(clean(entity.get("title")), f"row_id={row_id} 候选实体缺少 title")
                history = entity.get("history_items")
                require(isinstance(history, list) and len(history) > 0, f"row_id={row_id} 候选实体 {entity.get('label')} 缺少 history_items")

        for field in ["history_preview", "candidate_preview", "target_question_text", "learning_goal", "progress_state"]:
            if field in df.columns:
                text = clean(row.get(field, ""))
                require("GT:" not in text and "is_gt" not in text and "correct_answer" not in text, f"row_id={row_id} 面向老师字段 {field} 出现泄漏标记")


def _brace_end(s: str, start: int) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1



def _simple_arg_end(s: str, start: int) -> int:
    """Return end position for a simple unbraced LaTeX argument such as 2, x, AB, or (x+1).

    This is not a fallback-to-run mechanism; it is part of the renderer grammar because the
    dataset contains both \sqrt{2} and \sqrt2 / \sqrt 2 style text.
    """
    j = start
    while j < len(s) and s[j].isspace():
        j += 1
    if j >= len(s):
        return j
    if s[j] in "([|":
        pairs = {"(": ")", "[": "]", "|": "|"}
        close = pairs[s[j]]
        k = j + 1
        while k < len(s):
            if s[k] == close:
                return k + 1
            k += 1
        return j + 1
    if s[j].isalnum() or s[j] in "+-=.°παβγθ":
        k = j + 1
        while k < len(s) and (s[k].isalnum() or s[k] in "+-=.°παβγθ"):
            k += 1
        return k
    return j + 1


def _one_arg(s: str, cmd: str, renderer) -> str:
    """Replace LaTeX command with one argument.

    Supports both braced form (\sqrt{2}) and common compact form (\sqrt2).
    Unknown or incomplete command is preserved as text instead of raising during rendering.
    Structural validation belongs to the CSV generator; the app renderer should not crash
    because one historical question contains informal LaTeX.
    """
    needle = "\\" + cmd
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s.startswith(needle, i):
            out.append(s[i])
            i += 1
            continue
        j = i + len(needle)
        while j < len(s) and s[j].isspace():
            j += 1
        if j < len(s) and s[j] == "{":
            end = _brace_end(s, j)
            if end >= 0:
                out.append(renderer(s[j + 1:end]))
                i = end + 1
                continue
        # compact form, e.g. \sqrt2. If there is no usable argument, preserve command.
        end = _simple_arg_end(s, j)
        if end > j:
            out.append(renderer(s[j:end]))
            i = end
        else:
            out.append(needle)
            i = j
    return "".join(out)


def _two_args(s: str, cmd: str, renderer) -> str:
    """Replace LaTeX command with two braced arguments.

    Malformed two-argument commands are preserved so the page remains usable and the
    problematic text is visible to the annotator instead of crashing the whole app.
    """
    needle = "\\" + cmd
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s.startswith(needle, i):
            out.append(s[i])
            i += 1
            continue
        j = i + len(needle)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            out.append(needle)
            i = j
            continue
        end1 = _brace_end(s, j)
        if end1 < 0:
            out.append(s[i:j + 1])
            i = j + 1
            continue
        k = end1 + 1
        while k < len(s) and s[k].isspace():
            k += 1
        if k >= len(s) or s[k] != "{":
            out.append(s[i:end1 + 1])
            i = end1 + 1
            continue
        end2 = _brace_end(s, k)
        if end2 < 0:
            out.append(s[i:k + 1])
            i = k + 1
            continue
        out.append(renderer(s[j + 1:end1], s[k + 1:end2]))
        i = end2 + 1
    return "".join(out)


def normalize_latex(raw: Any) -> str:
    s = clean(raw)
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("$$", "").replace("$", "")
    s = re.sub(r"\\begin\{cases\}", "{ ", s)
    s = re.sub(r"\\end\{cases\}", " }", s)
    s = s.replace("&", " ").replace("\\\\", "； ")

    s = _two_args(s, "dfrac", lambda a, b: f"@@FRAC:{normalize_latex(a)}|{normalize_latex(b)}@@")
    s = _two_args(s, "frac", lambda a, b: f"@@FRAC:{normalize_latex(a)}|{normalize_latex(b)}@@")
    s = _one_arg(s, "sqrt", lambda a: f"√({normalize_latex(a)})")
    for cmd in ["text", "mathrm", "rm", "boldsymbol"]:
        s = _one_arg(s, cmd, normalize_latex)
    s = _one_arg(s, "overline", lambda a: f"{normalize_latex(a)}̅")
    s = _one_arg(s, "overparen", lambda a: f"⌒{normalize_latex(a)}")

    replacements = {
        r"\vartriangle": "△", r"\triangle": "△", r"\angle": "∠",
        r"\bot": "⊥", r"\perp": "⊥", r"\parallel": "∥", r"/\!/": "∥",
        r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\circ": "°",
        r"\leqslant": "≤", r"\leq": "≤", r"\geqslant": "≥", r"\geq": "≥", r"\neq": "≠",
        r"\pi": "π", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\theta": "θ",
        r"\left": "", r"\right": "", r"\quad": "　", r"\qquad": "　　",
        r"\sim": "∼", r"\infty": "∞", r"\cdots": "⋯", r"\dots": "…", r"\odot": "⊙",
        r"\#": "#",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\{\s*\^\s*°\s*\}", "°", s)
    s = re.sub(r"\^\s*\{\s*°\s*\}", "°", s)

    def sup(m: re.Match) -> str:
        body = normalize_latex(m.group(1) or m.group(2))
        return body.translate(SUP) if re.fullmatch(r"[0-9+\-=()nix]+", body) else f"@@SUP:{body}@@"

    def sub(m: re.Match) -> str:
        body = normalize_latex(m.group(1) or m.group(2))
        return body.translate(SUB) if re.fullmatch(r"[0-9+\-=()nix]+", body) else f"@@SUB:{body}@@"

    s = re.sub(r"\^\{([^{}]{1,80})\}|\^([A-Za-z0-9+\-=()])", sup, s)
    s = re.sub(r"_\{([^{}]{1,80})\}|_([A-Za-z0-9+\-=()])", sub, s)
    s = re.sub(r"\{([A-Za-z0-9+\-*/=<>≤≥.,，。:：_()（）\[\]αβγθπ°| ]{1,100})\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    return s.replace("{", "").replace("}", "")


def qhtml(raw: Any) -> str:
    s = normalize_latex(raw).replace("\r\n", "\n").replace("\r", "\n")
    protected: dict[str, str] = {}

    def protect(pattern: str, renderer) -> None:
        nonlocal s
        def repl(m: re.Match) -> str:
            token = f"@@HTML{len(protected)}@@"
            protected[token] = renderer(m)
            return token
        s = re.sub(pattern, repl, s, flags=re.S)

    protect(r"@@FRAC:(.*?)\|(.*?)@@", lambda m: f"<span class='frac'><span class='num'>{html.escape(m.group(1))}</span><span class='den'>{html.escape(m.group(2))}</span></span>")
    protect(r"@@SUP:(.*?)@@", lambda m: f"<sup>{html.escape(m.group(1))}</sup>")
    protect(r"@@SUB:(.*?)@@", lambda m: f"<sub>{html.escape(m.group(1))}</sub>")
    out = html.escape(s, quote=False).replace("\n", "<br>")
    for token, value in protected.items():
        out = out.replace(token, value)
    return re.sub(r"(?<![A-Za-z])([ABCD])\.", r"<span class='choice'>\1.</span>", out)


def short_hint(text: Any, n: int = 72) -> str:
    plain = re.sub(r"<[^>]+>", "", qhtml(text))
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain if len(plain) <= n else plain[:n] + "…"


def score_text(item: dict[str, Any]) -> str:
    score = clean(item.get("score"))
    total = clean(item.get("total"))
    if score and total:
        return f"{score}/{total}"
    if score:
        return score
    return "—"


def score_class(item: dict[str, Any]) -> str:
    score = clean(item.get("score"))
    total = clean(item.get("total"))
    if not score or not total:
        return " score-empty"
    s = float(score)
    t = float(total)
    if t > 0 and s >= t:
        return " score-full"
    if s == 0:
        return " score-zero"
    return " score-partial"


def history_key(item: dict[str, Any]) -> tuple[str, str]:
    return clean(item.get("qid")), clean(item.get("index"))


def collect_questions(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for ent in entities:
        for item in ent["history_items"]:
            key = history_key(item)
            if key not in seen:
                seen[key] = item
    def sort_key(x: dict[str, Any]) -> tuple[int, str]:
        idx = clean(x.get("index"))
        return (int(idx) if idx.isdigit() else 9999, clean(x.get("qid")))
    return sorted(seen.values(), key=sort_key)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    html_block("""
    <style>
    .block-container {max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stVerticalBlock"] {gap: .62rem;}
    .task-card,.goal-card,.target-card,.section-card,.candidate-card,.entity-card {border:1px solid #e5e7eb; border-radius:18px; padding:14px 16px; background:#fff; box-shadow:0 1px 2px rgba(15,23,42,.045); margin:.55rem 0;}
    .task-card {background:linear-gradient(180deg,#eff6ff 0%,#fff 90%); border-color:#bfdbfe;}
    .goal-card {background:#fffbeb; border-color:#f59e0b; border-left:7px solid #f59e0b;}
    .target-card {background:#f0fdf4; border-color:#22c55e; border-left:7px solid #22c55e;}
    .history-card {border-left:6px solid #94a3b8;}
    .candidate-card {border-left:7px solid #2563eb;}
    .entity-card {border-left:7px solid #7c3aed;}
    .section-title {font-weight:900; font-size:1.12rem; margin:1.05rem 0 .35rem; display:flex; align-items:center; gap:.45rem;}
    .section-title:before {content:""; display:inline-block; width:5px; height:18px; border-radius:999px; background:#2563eb;}
    .card-head {display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:8px;}
    .card-title {font-weight:900; color:#0f172a;}
    .meta {font-size:.82rem; color:#64748b; background:#f1f5f9; padding:3px 9px; border-radius:999px; white-space:nowrap;}
    .badge {display:inline-flex; align-items:center; justify-content:center; min-width:28px; height:28px; padding:0 8px; margin-right:8px; border-radius:999px; background:#2563eb; color:white; font-weight:900;}
    .entity-card .badge {background:#7c3aed;}
    .qtext {font-size:1.02rem; line-height:1.85; word-break:break-word; overflow-wrap:anywhere;}
    .smalltext {font-size:.95rem; line-height:1.72; color:#334155;}
    .frac {display:inline-grid; grid-template-rows:auto auto; align-items:center; justify-items:center; vertical-align:middle; margin:0 .14em; line-height:1.05;}
    .frac .num {border-bottom:1.4px solid currentColor; padding:0 .22em .08em;}
    .frac .den {padding:.08em .22em 0;}
    .choice {display:inline-block; margin-left:.5em; margin-right:.12em; padding:0 .28em; border-radius:6px; background:#eef2ff; color:#3730a3; font-weight:900;}
    .score-table {width:100%; border-collapse:separate; border-spacing:0; margin-top:10px; font-size:.92rem; overflow:hidden; border:1px solid #e5e7eb; border-radius:14px;}
    .score-table th {background:#f8fafc; color:#334155; font-weight:800; text-align:left; padding:8px 10px; border-bottom:1px solid #e5e7eb;}
    .score-table td {padding:8px 10px; border-bottom:1px solid #eef2f7; vertical-align:top;}
    .score-table tr:last-child td {border-bottom:0;}
    .qnum {font-weight:900; color:#1d4ed8; white-space:nowrap;}
    .qhint {color:#0f172a; min-width:280px;}
    .score-chip {display:inline-block; min-width:46px; text-align:center; border-radius:999px; padding:2px 8px; font-weight:850; background:#f1f5f9; color:#334155;}
    .score-full {background:#dcfce7; color:#166534;}
    .score-zero {background:#fee2e2; color:#991b1b;}
    .score-partial {background:#fef3c7; color:#92400e;}
    .score-empty {background:#f1f5f9; color:#64748b;}
    .member-total {display:inline-block; margin:4px 6px 4px 0; padding:3px 9px; border-radius:999px; background:#f5f3ff; color:#5b21b6; font-weight:800; font-size:.88rem;}
    </style>
    """)


def card(title: str, body: Any, meta: str = "", kind: str = "section-card", badge: str = "") -> None:
    badge_html = f"<span class='badge'>{html.escape(badge)}</span>" if badge else ""
    meta_html = f"<span class='meta'>{html.escape(meta)}</span>" if meta else ""
    html_block(
        f"<div class='{kind}'>"
        f"<div class='card-head'><div class='card-title'>{badge_html}{html.escape(title)}</div>{meta_html}</div>"
        f"<div class='qtext'>{qhtml(body)}</div>"
        "</div>"
    )


def render_task(row: pd.Series) -> None:
    benchmark = clean(row["benchmark"])
    info = TASK_GUIDANCE[benchmark]
    bullets = "".join(f"<li>{html.escape(x)}</li>" for x in info["basis"])
    focus = clean(row["teacher_check_focus"])
    focus_html = f"<div class='smalltext'><b>检查重点：</b>{html.escape(focus)}</div>" if focus else ""
    html_block(
        "<div class='task-card'>"
        f"<div class='card-title'>{html.escape(TASK_LABELS[benchmark])}</div>"
        f"<div class='smalltext'><b>任务：</b>{html.escape(info['goal'])}</div>"
        f"<ul class='smalltext'>{bullets}</ul>"
        f"{focus_html}"
        "</div>"
    )


def render_history_score_table(items: list[dict[str, Any]], title: str = "学生历史作答记录") -> None:
    """Compact score view for tasks where correctness/progress matters.

    Each row carries Q号 + qid + short question hint + score, matching the
    Personality task style so annotators do not need to map dry Q numbers
    back to a separate question list.
    """
    html_block(f"<div class='section-title'>{html.escape(title)}</div>")
    rows: list[str] = []
    for i, item in enumerate(items, start=1):
        idx = clean(item.get("index")) or str(i)
        qid = clean(item.get("qid"))
        rows.append(
            "<tr>"
            f"<td class='qnum'>Q{html.escape(idx)}</td>"
            f"<td><span class='meta'>{html.escape(qid or '—')}</span></td>"
            f"<td class='qhint'>{html.escape(short_hint(item.get('question_text', ''), 96))}</td>"
            f"<td><span class='score-chip{score_class(item)}'>{html.escape(score_text(item))}</span></td>"
            "</tr>"
        )
    html_block(
        "<table class='score-table'>"
        "<thead><tr><th>题号</th><th>qid</th><th>题目提示</th><th>得分</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_history_questions(items: list[dict[str, Any]], title: str, show_score: bool) -> None:
    html_block(f"<div class='section-title'>{html.escape(title)}</div>")
    for i, item in enumerate(items, start=1):
        idx = clean(item.get("index")) or str(i)
        qid = clean(item.get("qid"))
        meta_parts = [qid]
        if show_score:
            meta_parts.append(score_text(item))
        meta = " · ".join(x for x in meta_parts if x and x != "—")
        card(f"历史题 Q{idx}", item["question_text"], meta=meta, kind="section-card history-card")


def render_question_history(items: list[dict[str, Any]], benchmark: str) -> None:
    """Render history exactly once for non-Personality tasks.

    Personality tasks are special: teachers first read the full shared question set,
    then compare entity score matrices with short hints. For question-recommendation
    tasks, repeating the same history as both a table and full cards wastes visual
    attention, so each benchmark gets one history view only.
    """
    if benchmark == "KP_Stage":
        # 教学进度只判断章节 / 知识阶段：只看题目，不把得分放大。
        render_history_questions(items, "学生历史题目", show_score=False)
        return

    if benchmark == "Planning_Progress":
        # 学习进程必须看轨迹和得分：用个性区分同款表格，但不再重复展示完整题。
        render_history_score_table(items, "学生最近学习轨迹 / 得分记录")
        return

    # 薄弱匹配和目标适配也需要看到作答表现，但只展示一次。
    render_history_score_table(items, "学生历史作答记录")


def render_question_candidates(items: list[dict[str, Any]]) -> None:
    html_block("<div class='section-title'>候选题</div>")
    for i, item in enumerate(items):
        label = clean(item["label"])
        qid = clean(item.get("qid"))
        card(f"候选 {label}", item["question_text"], meta=qid, kind="candidate-card", badge=label)


def render_full_history_questions(entities: list[dict[str, Any]]) -> None:
    html_block("<div class='section-title'>完整历史题目</div>")
    html_block("<div class='section-card'><div class='smalltext'><b>阅读方式：</b>先完整看一遍历史题，形成题型印象；随后每个候选学生 / 学生组的作答记录表里会把 Q号、qid、题目提示和得分放在同一行。</div></div>")
    for item in collect_questions(entities):
        idx = clean(item.get("index"))
        qid = clean(item.get("qid"))
        card(f"历史题 Q{idx}", item["question_text"], meta=qid, kind="section-card history-card")


def group_items_by_member(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        member = clean(item.get("member_label")) or clean(item.get("member_id")) or "学生"
        grouped.setdefault(member, []).append(item)
    return grouped


def member_total(items: list[dict[str, Any]]) -> str:
    score = 0.0
    total = 0.0
    seen_score = False
    for item in items:
        s = clean(item.get("score"))
        t = clean(item.get("total"))
        if s:
            score += float(s)
            seen_score = True
        if t:
            total += float(t)
    if not seen_score:
        return "—"
    return f"{score:g}/{total:g}" if total else f"{score:g}"


def individual_table(entity: dict[str, Any]) -> str:
    rows = []
    for i, item in enumerate(entity["history_items"], start=1):
        idx = clean(item.get("index")) or str(i)
        qid = clean(item.get("qid"))
        rows.append(
            "<tr>"
            f"<td class='qnum'>Q{html.escape(idx)}</td>"
            f"<td><span class='meta'>{html.escape(qid or '—')}</span></td>"
            f"<td class='qhint'>{html.escape(short_hint(item['question_text'], 90))}</td>"
            f"<td><span class='score-chip{score_class(item)}'>{html.escape(score_text(item))}</span></td>"
            "</tr>"
        )
    return "<table class='score-table'><thead><tr><th>题号</th><th>qid</th><th>题目提示</th><th>得分</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def group_table(entity: dict[str, Any]) -> str:
    grouped = group_items_by_member(entity["history_items"])
    members = [clean(x) for x in entity.get("members", []) if clean(x)] or sorted(grouped)
    questions = collect_questions([entity])
    lookup = {(member, history_key(item)): item for member, items in grouped.items() for item in items}
    head = "<tr><th>题号</th><th>qid</th><th>题目提示</th>" + "".join(f"<th>成员 {html.escape(m)}</th>" for m in members) + "</tr>"
    rows = []
    for i, q in enumerate(questions, start=1):
        idx = clean(q.get("index")) or str(i)
        qid = clean(q.get("qid"))
        cells = [
            f"<td class='qnum'>Q{html.escape(idx)}</td>",
            f"<td><span class='meta'>{html.escape(qid or '—')}</span></td>",
            f"<td class='qhint'>{html.escape(short_hint(q['question_text'], 82))}</td>",
        ]
        for member in members:
            item = lookup.get((member, history_key(q)), {})
            cells.append(f"<td><span class='score-chip{score_class(item)}'>{html.escape(score_text(item))}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table class='score-table'><thead>" + head + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_entity_candidates(entities: list[dict[str, Any]], benchmark: str) -> None:
    html_block(f"<div class='section-title'>{'候选学生组' if benchmark == 'Personality_Group' else '候选学生'}</div>")
    for entity in entities:
        label = clean(entity["label"])
        entity_id = clean(entity.get("entity_id"))
        if benchmark == "Personality_Group":
            grouped = group_items_by_member(entity["history_items"])
            members = [clean(x) for x in entity.get("members", []) if clean(x)] or sorted(grouped)
            totals = "".join(f"<span class='member-total'>成员 {html.escape(m)}：{html.escape(member_total(grouped.get(m, [])))}</span>" for m in members)
            table = group_table(entity)
            meta = " · ".join(x for x in [entity_id, "成员：" + "、".join(members)] if x)
        else:
            totals = f"<span class='member-total'>总分：{html.escape(member_total(entity['history_items']))}</span>"
            table = individual_table(entity)
            meta = entity_id
        html_block(
            "<div class='entity-card'>"
            f"<div class='card-head'><div class='card-title'><span class='badge'>{html.escape(label)}</span>{html.escape(entity['title'])}</div><span class='meta'>{html.escape(meta)}</span></div>"
            f"<div>{totals}</div>"
            f"{table}"
            "</div>"
        )


def render_left(row: pd.Series) -> tuple[list[dict[str, Any]], str]:
    benchmark = clean(row["benchmark"])
    row_id = clean(row["row_id"])
    render_task(row)

    if benchmark == "Planning_Target":
        card("显式学习目标", row["learning_goal"], kind="goal-card")
    if benchmark == "Planning_Progress":
        card("学习进程状态", row["progress_state"], kind="goal-card")
    if benchmark == "KP_Weak" and clean(row["weak_knowledge_points"]):
        card("薄弱知识点", row["weak_knowledge_points"], kind="goal-card")
    if benchmark == "KP_Stage" and clean(row["stage_knowledge_points"]):
        card("当前知识阶段 / 章节", row["stage_knowledge_points"], kind="goal-card")

    if benchmark in QUESTION_TASKS:
        history = json_list(row["history_items_json"], "history_items_json", row_id)
        candidates = json_list(row["candidate_items_json"], "candidate_items_json", row_id)
        render_question_history(history, benchmark)
        render_question_candidates(candidates)
        return candidates, "question"

    target = json_dict(row["target_question_json"], "target_question_json", row_id)
    entities = json_list(row["candidate_entities_json"], "candidate_entities_json", row_id)
    card("目标题目", target["question_text"], meta=clean(target.get("qid")), kind="target-card")
    render_full_history_questions(entities)
    render_entity_candidates(entities, benchmark)
    return entities, "entity"


def init_state(df: pd.DataFrame) -> None:
    st.session_state.setdefault("annotations", {})
    st.session_state.setdefault("current_index", 0)
    valid = set(df["row_id"].astype(str))
    st.session_state.annotations = {str(k): v for k, v in st.session_state.annotations.items() if str(k) in valid}


def default_annotation() -> dict[str, Any]:
    return {"teacher_choice_label": "", "teacher_choice_text": "", "teacher_confidence": 3, "teacher_reason": "", "teacher_comment": ""}


def render_form(row: pd.Series, choices: list[dict[str, Any]], choice_type: str) -> None:
    row_id = clean(row["row_id"])
    current = {**default_annotation(), **st.session_state.annotations.get(row_id, {})}
    labels = [clean(x["label"]) for x in choices]
    options = [""] + labels + ["无法判断 / 暂不推荐"]
    label_to_text = {
        clean(x["label"]): clean(x["title"]) if choice_type == "entity" else clean(x["question_text"])
        for x in choices
    }

    st.subheader("教师标注")
    choice = st.radio(
        "请选择你会推荐的候选项",
        options=options,
        index=options.index(current["teacher_choice_label"]) if current["teacher_choice_label"] in options else 0,
        format_func=lambda x: "请选择" if x == "" else x,
        horizontal=True,
    )
    if choice in label_to_text:
        st.markdown(f"**已选：{choice}**")
        st.markdown(f"<div class='section-card'><div class='qtext'>{qhtml(label_to_text[choice])}</div></div>", unsafe_allow_html=True)
    confidence = st.slider("推荐把握度", 1, 5, int(current["teacher_confidence"]))
    reason = st.text_area("推荐理由", value=clean(current["teacher_reason"]), height=135, placeholder="请说明为什么推荐这个候选。")
    comment = st.text_area("补充备注", value=clean(current["teacher_comment"]), height=115, placeholder="可记录歧义、样本问题、其他备选意见。")

    st.session_state.annotations[row_id] = {
        "teacher_choice_label": choice,
        "teacher_choice_text": label_to_text.get(choice, choice),
        "teacher_confidence": confidence,
        "teacher_reason": reason,
        "teacher_comment": comment,
    }


def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
    base = df.set_index("row_id", drop=False).to_dict(orient="index")
    rows = []
    for row_id, annotation in st.session_state.annotations.items():
        rows.append({**base[row_id], **annotation, "saved_at": datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(rows)


def save_local(export_df: pd.DataFrame) -> Path:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATION_DIR / f"teacher_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def score_text(item: dict[str, Any]) -> str:
    score = clean(item.get("score"))
    total = clean(item.get("total"))
    response = clean(item.get("response"))
    if (not score and not total) and response:
        if response == "correct":
            score, total = "1", "1"
        elif response == "incorrect":
            score, total = "0", "1"
    if score and total:
        return f"{score}/{total}"
    if score:
        return score
    return "-"


def score_class(item: dict[str, Any]) -> str:
    score = clean(item.get("score"))
    total = clean(item.get("total"))
    response = clean(item.get("response"))
    if (not score and not total) and response:
        if response == "correct":
            score, total = "1", "1"
        elif response == "incorrect":
            score, total = "0", "1"
    if not score or not total:
        return " score-empty"
    s = float(score)
    t = float(total)
    if t > 0 and s >= t:
        return " score-full"
    if s == 0:
        return " score-zero"
    return " score-partial"


def member_total(items: list[dict[str, Any]]) -> str:
    score = 0.0
    total = 0.0
    seen_score = False
    for item in items:
        s = clean(item.get("score"))
        t = clean(item.get("total"))
        response = clean(item.get("response"))
        if (not s and not t) and response:
            if response == "correct":
                s, t = "1", "1"
            elif response == "incorrect":
                s, t = "0", "1"
        if s:
            score += float(s)
            seen_score = True
        if t:
            total += float(t)
    if not seen_score:
        return "-"
    return f"{score:g}/{total:g}" if total else f"{score:g}"


def main() -> None:
    st.set_page_config(page_title="教师推荐标注", layout="wide")
    inject_css()
    st.title("教师推荐标注")
    st.caption(f"版本：{APP_VERSION}")
    st.caption("严格版：只读取规范 review_samples.csv；缺字段或字段语义错误会直接报错，不做兼容兜底。")

    df = load_samples()
    init_state(df)

    st.sidebar.header("筛选")
    dims = ["全部"] + [TASK_LABELS[x] for x in sorted(df["benchmark"].unique())]
    selected = st.sidebar.selectbox("维度", dims)
    if selected == "全部":
        filtered = df.reset_index(drop=True)
    else:
        rev = {TASK_LABELS[k]: k for k in TASK_LABELS}
        filtered = df[df["benchmark"] == rev[selected]].reset_index(drop=True)

    require(len(filtered) > 0, "当前筛选下没有样本")
    if st.session_state.current_index >= len(filtered):
        st.session_state.current_index = 0

    annotated = sum(1 for rid in filtered["row_id"].astype(str) if st.session_state.annotations.get(rid, {}).get("teacher_choice_label"))
    st.sidebar.metric("当前筛选下已标注", f"{annotated}/{len(filtered)}")

    row_ids = filtered["row_id"].astype(str).tolist()
    selected_row_id = st.sidebar.selectbox(
        "跳转到样本",
        row_ids,
        index=st.session_state.current_index,
        format_func=lambda rid: f"{rid} | {TASK_LABELS[clean(filtered[filtered['row_id'].astype(str) == rid].iloc[0]['benchmark'])]}",
    )
    st.session_state.current_index = row_ids.index(selected_row_id)

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("上一条", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    if c2.button("下一条", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
    c3.progress((st.session_state.current_index + 1) / len(filtered))

    row = filtered.iloc[st.session_state.current_index]
    st.caption(f"样本 {st.session_state.current_index + 1} / {len(filtered)} · row_id={clean(row['row_id'])} · sample_id={clean(row['sample_id'])}")

    left, right = st.columns([1.62, 1], gap="large")
    with left:
        choices, choice_type = render_left(row)
    with right:
        render_form(row, choices, choice_type)

    st.divider()
    export_df = build_export_df(df)
    st.subheader("导出")
    st.write(f"当前会话已保存标注：{len(export_df)}")
    st.download_button(
        "下载标注 CSV",
        data=export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="teacher_recommendation_annotations.csv",
        mime="text/csv",
    )
    if st.button("保存到本地文件"):
        path = save_local(export_df)
        st.success(f"已保存到 {path}")
    with st.expander("当前会话标注预览", expanded=False):
        st.dataframe(export_df, use_container_width=True)


if __name__ == "__main__":
    main()
