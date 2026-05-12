from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
CSV_CANDIDATES = [DATA_DIR / "review_samples.csv"]

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
        "basis": ["优先关注历史中暴露出的薄弱知识点。", "候选题如果能针对这些知识点进行练习，则值得推荐。", "请选最值得优先推荐的一题。"],
    },
    "KP_Stage": {
        "goal": "根据学生历史题目与表现，选择最符合当前教学章节 / 知识阶段的候选题。",
        "basis": ["重点看候选题是否符合当前教学章节。", "判断标准不是单纯难度高低。", "明显过早或过晚的题不应优先推荐。"],
    },
    "Planning_Progress": {
        "goal": "根据学生最近学习轨迹，选择最适合作为下一步练习的候选题。",
        "basis": ["判断学生当前更适合推进、巩固，还是回退修复。", "避免过难题或无效重复题。", "从教学推进角度选择最合理的一题。"],
    },
    "Planning_Target": {
        "goal": "在给定学习目标下，选择最能推进该目标的候选题。",
        "basis": ["显式学习目标是核心条件。", "优先考虑候选题是否真正服务目标。", "不要只看与历史题的表面相似。"],
    },
    "Personality_Individual": {
        "goal": "给定目标题目，从候选学生中选择最适合推荐这道题的学生。",
        "basis": ["比较每个候选学生的历史表现与目标题所需能力。", "不要只看总体成绩。", "判断谁最适合这道具体题目。"],
    },
    "Personality_Group": {
        "goal": "给定目标题目，从候选学生组中选择最适合推荐这道题的学生组。",
        "basis": ["比较小组整体能力结构与目标题的匹配程度。", "不要只看单个成员。", "判断哪个组最适合这道具体题目。"],
    },
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUP = str.maketrans("0123456789+-=()nix", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣ")
SUB = str.maketrans("0123456789+-=()nix", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙᵢₓ")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def clean_str(value: Any) -> str:
    return "" if is_blank(value) else str(value).strip()


def load_json(value: Any, default: Any) -> Any:
    if is_blank(value):
        return default
    text = str(value).strip()
    try:
        return json.loads(text)
    except Exception:
        return default


def list_json(value: Any) -> list[dict[str, Any]]:
    obj = load_json(value, [])
    return obj if isinstance(obj, list) else []


def dict_json(value: Any) -> dict[str, Any]:
    obj = load_json(value, {})
    return obj if isinstance(obj, dict) else {}


def find_csv() -> Path:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("找不到 review_samples.csv。请放在 data/review_samples.csv 或 app.py 同目录。")


def load_samples() -> pd.DataFrame:
    df = pd.read_csv(find_csv())
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))
    for col in [
        "benchmark", "task_type", "task_label", "learning_goal", "progress_state",
        "weak_knowledge_points", "stage_knowledge_points", "target_question_json",
        "target_question_text", "history_items_json", "candidate_items_json",
        "candidate_entities_json", "history_preview", "candidate_preview", "teacher_check_focus",
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def parse_prefixed_preview(text: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if is_blank(text):
        return out
    for idx, line in enumerate(str(text).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        label = LETTERS[len(out)] if len(out) < len(LETTERS) else str(len(out) + 1)
        body = line
        m = re.match(r"^([A-Z]|C\d+|\d+)[.:：]\s*(.*)$", line)
        if m:
            label, body = m.group(1), m.group(2)
            if label.startswith("C") and label[1:].isdigit():
                num = int(label[1:]) - 1
                label = LETTERS[num] if 0 <= num < len(LETTERS) else label
        else:
            m2 = re.match(r"^(\d+)\.\s*(?:[（(][^）)]*[）)]\s*)?(.*)$", line)
            if m2:
                body = m2.group(2)
        out.append({"display_index": len(out) + 1, "label": label, "question_text": body, "text": body})
    return out


def ensure_label(item: dict[str, Any], idx: int) -> str:
    label = clean_str(item.get("label"))
    if label:
        return label
    return LETTERS[idx] if idx < len(LETTERS) else f"Option-{idx + 1}"


def get_history_items(row: pd.Series) -> list[dict[str, Any]]:
    items = list_json(row.get("history_items_json", ""))
    if items:
        return items
    return parse_prefixed_preview(row.get("history_preview", ""))


def get_question_candidates(row: pd.Series) -> list[dict[str, Any]]:
    raw = list_json(row.get("candidate_items_json", ""))
    if not raw:
        raw = parse_prefixed_preview(row.get("candidate_preview", ""))
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        text = clean_str(item.get("question_text", item.get("text", "")))
        if not text:
            continue
        out.append({
            "display_index": item.get("display_index", idx + 1),
            "label": ensure_label(item, idx),
            "qid": clean_str(item.get("qid", "")),
            "question_text": text,
            "knowledge_points": item.get("knowledge_points", []),
            "chapter": clean_str(item.get("chapter", "")),
        })
    return out


def get_target_question(row: pd.Series) -> dict[str, Any]:
    obj = dict_json(row.get("target_question_json", ""))
    if obj and clean_str(obj.get("question_text")):
        return obj
    text = clean_str(row.get("target_question_text", "")) or clean_str(row.get("gt_candidate_text_machine", "")) or clean_str(row.get("gt_candidate_text", "")) or clean_str(row.get("gt_signal", ""))
    return {"qid": "", "question_text": text}


def get_learning_goal(row: pd.Series) -> str:
    return clean_str(row.get("learning_goal", "")) or (clean_str(row.get("gt_signal", "")) if clean_str(row.get("benchmark", "")) == "Planning_Target" else "")


def get_personality_candidates(row: pd.Series) -> list[dict[str, Any]]:
    entities = list_json(row.get("candidate_entities_json", ""))
    if not entities:
        # 极端兜底：不要空白。旧数据下从 candidate_preview 拆，但新数据通常不会走这里。
        preview = parse_prefixed_preview(row.get("candidate_preview", ""))
        entities = [
            {"display_index": i + 1, "label": ensure_label(item, i), "title": f"候选 {ensure_label(item, i)}", "summary": item.get("question_text", ""), "history_items": []}
            for i, item in enumerate(preview)
        ]
    out: list[dict[str, Any]] = []
    for idx, ent in enumerate(entities):
        label = ensure_label(ent, idx)
        out.append({
            "display_index": ent.get("display_index", idx + 1),
            "label": label,
            "entity_id": clean_str(ent.get("entity_id", "")),
            "entity_type": clean_str(ent.get("entity_type", "")),
            "title": clean_str(ent.get("title", f"候选 {label}")),
            "members": ent.get("members", []),
            "summary": clean_str(ent.get("summary", "")),
            "history_items": ent.get("history_items", []) if isinstance(ent.get("history_items", []), list) else [],
        })
    return out


def _find_brace_end(s: str, start: int) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _replace_one_arg(s: str, cmd: str, renderer) -> str:
    needle = "\\" + cmd
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s.startswith(needle, i):
            out.append(s[i]); i += 1; continue
        j = i + len(needle)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            out.append(s[i]); i += 1; continue
        end = _find_brace_end(s, j)
        if end < 0:
            out.append(s[i]); i += 1; continue
        out.append(renderer(s[j + 1:end]))
        i = end + 1
    return "".join(out)


def _replace_two_args(s: str, cmd: str, renderer) -> str:
    needle = "\\" + cmd
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s.startswith(needle, i):
            out.append(s[i]); i += 1; continue
        j = i + len(needle)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            out.append(s[i]); i += 1; continue
        end1 = _find_brace_end(s, j)
        if end1 < 0:
            out.append(s[i]); i += 1; continue
        k = end1 + 1
        while k < len(s) and s[k].isspace():
            k += 1
        if k >= len(s) or s[k] != "{":
            out.append(s[i]); i += 1; continue
        end2 = _find_brace_end(s, k)
        if end2 < 0:
            out.append(s[i]); i += 1; continue
        out.append(renderer(s[j + 1:end1], s[k + 1:end2]))
        i = end2 + 1
    return "".join(out)


def normalize_latex_text(raw: Any) -> str:
    s = clean_str(raw)
    if not s:
        return ""

    # 删除公式分隔符，但保留内部内容。Planning_Progress 的 $$5$$ 会显示为 5。
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("$$", "").replace("$", "")

    # cases 环境转成多行可读文本。
    s = re.sub(r"\\begin\{cases\}", "{ ", s)
    s = re.sub(r"\\end\{cases\}", " }", s)
    s = s.replace("&", " ").replace("\\\\", "； ")

    def frac(a: str, b: str) -> str:
        return f"@@FRAC:{normalize_latex_text(a)}|{normalize_latex_text(b)}@@"

    s = _replace_two_args(s, "dfrac", frac)
    s = _replace_two_args(s, "frac", frac)
    s = _replace_one_arg(s, "sqrt", lambda a: f"√({normalize_latex_text(a)})")
    for cmd in ["text", "mathrm", "rm", "boldsymbol"]:
        s = _replace_one_arg(s, cmd, normalize_latex_text)
    s = _replace_one_arg(s, "overline", lambda a: f"{normalize_latex_text(a)}̅")
    s = _replace_one_arg(s, "overparen", lambda a: f"⌒{normalize_latex_text(a)}")

    repl = {
        r"\vartriangle": "△", r"\triangle": "△", r"\angle": "∠",
        r"\bot": "⊥", r"\perp": "⊥", r"\parallel": "∥", r"/\!/": "∥",
        r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\circ": "°",
        r"\leqslant": "≤", r"\leq": "≤", r"\geqslant": "≥", r"\geq": "≥", r"\neq": "≠",
        r"\pi": "π", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\theta": "θ",
        r"\left": "", r"\right": "", r"\quad": "　", r"\qquad": "　　",
        r"\sim": "∼", r"\infty": "∞", r"\cdots": "⋯", r"\dots": "…", r"\odot": "⊙",
        r"\#": "#",
    }
    for old, new in repl.items():
        s = s.replace(old, new)

    s = re.sub(r"\{\s*\^\s*°\s*\}", "°", s)
    s = re.sub(r"\^\s*\{\s*°\s*\}", "°", s)

    def sup(m: re.Match) -> str:
        body = normalize_latex_text(m.group(1) or m.group(2))
        if re.fullmatch(r"[0-9+\-=()nix]+", body):
            return body.translate(SUP)
        return f"@@SUP:{body}@@"

    def sub(m: re.Match) -> str:
        body = normalize_latex_text(m.group(1) or m.group(2))
        if re.fullmatch(r"[0-9+\-=()nix]+", body):
            return body.translate(SUB)
        return f"@@SUB:{body}@@"

    s = re.sub(r"\^\{([^{}]{1,60})\}|\^([A-Za-z0-9+\-=()])", sup, s)
    s = re.sub(r"_\{([^{}]{1,60})\}|_([A-Za-z0-9+\-=()])", sub, s)
    s = re.sub(r"\{([A-Za-z0-9+\-*/=<>≤≥.,，。:：_()（）\[\]αβγθπ°| ]{1,80})\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return s


def render_text_html(raw: Any) -> str:
    s = normalize_latex_text(raw).replace("\r\n", "\n").replace("\r", "\n")
    protected: dict[str, str] = {}

    def protect(pattern: str, repl_func) -> None:
        nonlocal s
        def repl(m: re.Match) -> str:
            token = f"@@HTML{len(protected)}@@"
            protected[token] = repl_func(m)
            return token
        s = re.sub(pattern, repl, s, flags=re.S)

    protect(r"@@FRAC:(.*?)\|(.*?)@@", lambda m: f"<span class='frac'><span class='num'>{html.escape(m.group(1))}</span><span class='den'>{html.escape(m.group(2))}</span></span>")
    protect(r"@@SUP:(.*?)@@", lambda m: f"<sup>{html.escape(m.group(1))}</sup>")
    protect(r"@@SUB:(.*?)@@", lambda m: f"<sub>{html.escape(m.group(1))}</sub>")
    out = html.escape(s, quote=False).replace("\n", "<br>")
    for token, value in protected.items():
        out = out.replace(token, value)
    out = re.sub(r"(?<![A-Za-z])([ABCD])\.", r"<span class='choice'>\1.</span>", out)
    return out


def css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stVerticalBlock"] {gap: .65rem;}
        .top-caption {color:#64748b; font-size:.92rem;}
        .task-card,.goal-card,.target-card,.section-card,.candidate-card,.entity-card {
            border:1px solid #e5e7eb; border-radius:18px; padding:14px 16px; background:#fff;
            box-shadow:0 1px 2px rgba(15,23,42,.045); margin:.55rem 0;
        }
        .task-card {background:linear-gradient(180deg,#eff6ff 0%,#ffffff 90%); border-color:#bfdbfe;}
        .goal-card {background:#fffbeb; border-color:#f59e0b; border-left:7px solid #f59e0b;}
        .target-card {background:#f0fdf4; border-color:#22c55e; border-left:7px solid #22c55e;}
        .section-title {font-weight:900; font-size:1.12rem; margin:1.1rem 0 .35rem; display:flex; align-items:center; gap:.45rem;}
        .section-title:before {content:""; display:inline-block; width:5px; height:18px; border-radius:999px; background:#2563eb;}
        .candidate-card {border-left:7px solid #2563eb;}
        .entity-card {border-left:7px solid #7c3aed;}
        .history-card {border-left:6px solid #94a3b8;}
        .card-head {display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:8px;}
        .card-title {font-weight:900; color:#0f172a;}
        .meta {font-size:.82rem; color:#64748b; background:#f1f5f9; padding:3px 9px; border-radius:999px; white-space:nowrap;}
        .badge {display:inline-flex; align-items:center; justify-content:center; min-width:28px; height:28px; padding:0 8px; margin-right:8px; border-radius:999px; background:#2563eb; color:white; font-weight:900;}
        .entity-card .badge {background:#7c3aed;}
        .qtext {font-size:1.02rem; line-height:1.85; word-break:break-word; overflow-wrap:anywhere;}
        .smalltext {font-size:.95rem; line-height:1.7; color:#334155;}
        .frac {display:inline-grid; grid-template-rows:auto auto; align-items:center; justify-items:center; vertical-align:middle; margin:0 .14em; line-height:1.05;}
        .frac .num {border-bottom:1.4px solid currentColor; padding:0 .22em .08em;}
        .frac .den {padding:.08em .22em 0;}
        sup, sub {font-size:.72em; line-height:0;}
        .choice {display:inline-block; margin-left:.45em; margin-right:.08em; padding:0 .28em; border-radius:6px; background:#eef2ff; color:#3730a3; font-weight:800;}
        .history-grid {display:grid; gap:8px;}
        .member {font-size:.88rem; color:#475569; margin:.25rem 0 .1rem; font-weight:800;}
        .sticky-box {position: sticky; top: 1rem;}
        .stRadio [role="radiogroup"] {gap:.35rem .65rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def card(title: str, text: Any, *, meta: str = "", kind: str = "section-card", badge: str = "") -> None:
    badge_html = f"<span class='badge'>{html.escape(badge)}</span>" if badge else ""
    meta_html = f"<span class='meta'>{html.escape(meta)}</span>" if meta else ""
    html_block(f"""
    <div class='{kind}'>
      <div class='card-head'><div class='card-title'>{badge_html}{html.escape(title)}</div>{meta_html}</div>
      <div class='qtext'>{render_text_html(text)}</div>
    </div>
    """)


def render_task_guidance(row: pd.Series) -> None:
    benchmark = clean_str(row.get("benchmark", ""))
    label = clean_str(row.get("task_label", "")) or TASK_LABELS.get(benchmark, benchmark)
    guide = TASK_GUIDANCE.get(benchmark, {"goal": "请根据页面信息选择最合理的推荐对象。", "basis": []})
    basis = "".join(f"<li>{html.escape(x)}</li>" for x in guide.get("basis", []))
    focus = clean_str(row.get("teacher_check_focus", ""))
    focus_html = f"<div class='smalltext'><b>检查重点：</b>{html.escape(focus)}</div>" if focus else ""
    html_block(f"""
    <div class='task-card'>
      <div class='card-title'>{html.escape(label)}</div>
      <div class='smalltext'><b>任务：</b>{html.escape(guide.get('goal', ''))}</div>
      <ul class='smalltext' style='margin:.35rem 0 0 1.1rem;'>{basis}</ul>
      {focus_html}
    </div>
    """)


def score_meta(item: dict[str, Any]) -> str:
    score, total = clean_str(item.get("score", "")), clean_str(item.get("total", ""))
    qid = clean_str(item.get("qid", ""))
    parts = []
    if score or total:
        parts.append(f"得分 {score}/{total}" if total else f"得分 {score}")
    if qid:
        parts.append(qid)
    return " · ".join(parts)


def render_history(items: list[dict[str, Any]], title: str = "学生历史 / 学习轨迹") -> None:
    html_block(f"<div class='section-title'>{html.escape(title)}</div>")
    if not items:
        st.info("没有可展示的历史信息。")
        return
    for idx, item in enumerate(items, start=1):
        label = clean_str(item.get("index", "")) or str(idx)
        text = item.get("question_text", item.get("text", ""))
        member = clean_str(item.get("member_label", ""))
        prefix = f"成员 {member} · " if member else ""
        card(f"{prefix}历史 {label}", text, meta=score_meta(item), kind="section-card history-card")


def render_question_candidates(candidates: list[dict[str, Any]], title: str = "候选题") -> None:
    html_block(f"<div class='section-title'>{html.escape(title)}</div>")
    if not candidates:
        st.error("当前样本没有候选题。请检查 candidate_items_json。")
        return
    for item in candidates:
        label = clean_str(item.get("label", ""))
        qid = clean_str(item.get("qid", ""))
        kp = item.get("knowledge_points", [])
        kp_text = "、".join(map(str, kp)) if isinstance(kp, list) and kp else ""
        meta = " · ".join(x for x in [qid, kp_text] if x)
        card(f"候选 {label}", item.get("question_text", ""), meta=meta, kind="candidate-card", badge=label)


def render_entity_candidates(entities: list[dict[str, Any]], benchmark: str) -> None:
    title = "候选学生组" if benchmark == "Personality_Group" else "候选学生"
    html_block(f"<div class='section-title'>{title}</div>")
    if not entities:
        st.error("当前样本没有候选实体。请检查 candidate_entities_json。")
        return
    for ent in entities:
        label = ent["label"]
        members = ent.get("members", [])
        members_text = "成员：" + "、".join(map(str, members)) if members else ""
        entity_id = clean_str(ent.get("entity_id", ""))
        meta = " · ".join(x for x in [entity_id, members_text] if x)
        summary = ent.get("summary", "")
        html_block(f"""
        <div class='entity-card'>
          <div class='card-head'><div class='card-title'><span class='badge'>{html.escape(label)}</span>{html.escape(ent.get('title') or f'候选 {label}')}</div><span class='meta'>{html.escape(meta)}</span></div>
          {f"<div class='smalltext'>{render_text_html(summary)}</div>" if summary else ""}
        """)
        histories = ent.get("history_items", [])
        if histories:
            # 直接在卡片内列前 10 条，完整但不至于爆炸；这份数据每候选通常 10 条。
            for i, item in enumerate(histories, start=1):
                idx = clean_str(item.get("index", "")) or str(i)
                member = clean_str(item.get("member_label", ""))
                member_html = f"<div class='member'>成员 {html.escape(member)}</div>" if member else ""
                html_block(f"""
                    {member_html}
                    <div class='section-card history-card' style='margin:.42rem 0; padding:10px 12px;'>
                      <div class='card-head'><div class='card-title'>历史 {html.escape(idx)}</div><span class='meta'>{html.escape(score_meta(item))}</span></div>
                      <div class='qtext'>{render_text_html(item.get('question_text', ''))}</div>
                    </div>
                """)
        html_block("</div>")


def render_left(row: pd.Series) -> tuple[list[dict[str, Any]], str]:
    benchmark = clean_str(row.get("benchmark", ""))
    render_task_guidance(row)

    if benchmark == "Planning_Target":
        goal = get_learning_goal(row)
        card("显式学习目标", goal or "（缺失）", kind="goal-card")
    elif benchmark == "Planning_Progress":
        state = clean_str(row.get("progress_state", "")) or clean_str(row.get("gt_signal", ""))
        if state:
            card("学习进程状态", state, kind="goal-card")
    elif benchmark == "KP_Weak":
        weak = clean_str(row.get("weak_knowledge_points", "")) or clean_str(row.get("gt_signal", ""))
        if weak:
            card("薄弱知识点", weak, kind="goal-card")
    elif benchmark == "KP_Stage":
        stage = clean_str(row.get("stage_knowledge_points", "")) or clean_str(row.get("gt_signal", ""))
        if stage:
            card("当前知识阶段 / 章节", stage, kind="goal-card")

    if benchmark in {"Personality_Individual", "Personality_Group"}:
        target = get_target_question(row)
        card("目标题目", target.get("question_text", "（缺失）"), meta=clean_str(target.get("qid", "")), kind="target-card")
        entities = get_personality_candidates(row)
        render_entity_candidates(entities, benchmark)
        return entities, "entity"

    history = get_history_items(row)
    render_history(history, "学生历史作答 / 学习轨迹")
    candidates = get_question_candidates(row)
    render_question_candidates(candidates, "候选下一步练习题" if benchmark.startswith("Planning") else "候选题")
    return candidates, "question"


def init_state(df: pd.DataFrame) -> None:
    st.session_state.setdefault("annotations", {})
    st.session_state.setdefault("current_index", 0)
    valid = set(map(int, df["row_id"].tolist()))
    st.session_state.annotations = {int(k): v for k, v in st.session_state.annotations.items() if int(k) in valid}


def default_annotation() -> dict[str, Any]:
    return {"teacher_choice_label": "", "teacher_choice_text": "", "teacher_confidence": 3, "teacher_reason": "", "teacher_comment": ""}


def render_annotation_form(row: pd.Series, choices: list[dict[str, Any]], choice_type: str) -> None:
    row_id = int(row["row_id"])
    current = {**default_annotation(), **st.session_state.annotations.get(row_id, {})}
    labels = [clean_str(x.get("label", "")) for x in choices if clean_str(x.get("label", ""))]
    options = [""] + labels + ["无法判断 / 暂不推荐"]
    label_to_text = {}
    for x in choices:
        label = clean_str(x.get("label", ""))
        if choice_type == "entity":
            label_to_text[label] = clean_str(x.get("title", "")) or clean_str(x.get("summary", "")) or label
        else:
            label_to_text[label] = clean_str(x.get("question_text", ""))

    html_block("<div class='sticky-box'>")
    st.subheader("教师标注")
    choice = st.radio(
        "请选择你会推荐的候选项",
        options=options,
        index=options.index(current.get("teacher_choice_label", "")) if current.get("teacher_choice_label", "") in options else 0,
        format_func=lambda x: "请选择" if x == "" else x,
        horizontal=True,
    )
    if choice in label_to_text:
        st.markdown(f"**已选：{choice}**")
        st.markdown(f"<div class='section-card'><div class='qtext'>{render_text_html(label_to_text[choice])}</div></div>", unsafe_allow_html=True)
    confidence = st.slider("推荐把握度", 1, 5, int(current.get("teacher_confidence", 3)))
    reason = st.text_area("推荐理由", value=clean_str(current.get("teacher_reason", "")), height=160, placeholder="请说明为什么推荐这个候选。")
    comment = st.text_area("补充备注", value=clean_str(current.get("teacher_comment", "")), height=110, placeholder="记录歧义、样本问题或其他备选意见。")
    st.session_state.annotations[row_id] = {
        "teacher_choice_label": choice,
        "teacher_choice_text": label_to_text.get(choice, choice),
        "teacher_confidence": confidence,
        "teacher_reason": reason,
        "teacher_comment": comment,
    }
    html_block("</div>")


def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_id = df.set_index("row_id").to_dict(orient="index")
    for row_id, ann in sorted(st.session_state.annotations.items()):
        base = by_id.get(row_id, {})
        rows.append({"row_id": row_id, **base, **ann, "saved_at": datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(rows)


def save_local(export_df: pd.DataFrame) -> Path:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATION_DIR / f"teacher_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def sidebar_filter(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("筛选")
    options = ["全部"] + [TASK_LABELS.get(b, b) for b in sorted(df["benchmark"].dropna().unique())]
    selected = st.sidebar.selectbox("任务维度", options)
    if selected == "全部":
        out = df.copy()
    else:
        reverse = {TASK_LABELS.get(b, b): b for b in df["benchmark"].dropna().unique()}
        out = df[df["benchmark"] == reverse[selected]].copy()
    return out.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="教师推荐标注", layout="wide")
    css()
    st.title("教师推荐标注")
    st.markdown("<div class='top-caption'>按任务语义展示：学习目标、学习进程、目标题目、候选学生 / 组会分开呈现；页面不展示 GT 或 is_gt。</div>", unsafe_allow_html=True)

    df = load_samples()
    init_state(df)
    filtered = sidebar_filter(df)
    if filtered.empty:
        st.warning("当前筛选下没有样本。")
        return

    annotated = sum(1 for rid in filtered["row_id"].tolist() if st.session_state.annotations.get(int(rid), {}).get("teacher_choice_label"))
    st.sidebar.metric("当前筛选已标注", f"{annotated}/{len(filtered)}")
    if st.session_state.current_index >= len(filtered):
        st.session_state.current_index = 0

    row_ids = filtered["row_id"].tolist()
    selected_row_id = st.sidebar.selectbox(
        "跳转到样本",
        row_ids,
        index=min(st.session_state.current_index, len(row_ids) - 1),
        format_func=lambda rid: f"{rid} | {TASK_LABELS.get(filtered[filtered['row_id'] == rid].iloc[0]['benchmark'], filtered[filtered['row_id'] == rid].iloc[0]['benchmark'])}",
    )
    st.session_state.current_index = int(filtered.index[filtered["row_id"] == selected_row_id][0])

    nav1, nav2, nav3 = st.columns([1, 1, 5])
    if nav1.button("上一条", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    if nav2.button("下一条", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
    nav3.progress((st.session_state.current_index + 1) / len(filtered))
    st.caption(f"样本 {st.session_state.current_index + 1} / {len(filtered)}")

    row = filtered.iloc[st.session_state.current_index]
    left, right = st.columns([1.58, 1.0], gap="large")
    with left:
        choices, choice_type = render_left(row)
    with right:
        render_annotation_form(row, choices, choice_type)

    st.divider()
    export_df = build_export_df(df)
    st.subheader("导出")
    st.write(f"当前会话已保存标注：{len(export_df)}")
    st.download_button(
        "下载标注 CSV",
        data=export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="teacher_recommendation_annotations.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.button("保存到本地 data/annotations", use_container_width=True):
        path = save_local(export_df)
        st.success(f"已保存到 {path}")
    with st.expander("当前会话标注预览", expanded=False):
        st.dataframe(export_df, use_container_width=True)


if __name__ == "__main__":
    main()
