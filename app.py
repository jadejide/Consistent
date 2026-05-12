from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
REVIEW_CSV = DATA_DIR / "review_samples.csv"

DIMENSION_LABELS = {
    "KP_Weak": "知识匹配：薄弱知识匹配",
    "KP_Stage": "知识匹配：阶段知识匹配",
    "Planning_Progress": "规划适配：学习进度适配",
    "Planning_Target": "规划适配：学习目标适配",
    "Personality_Individual": "个性区分：个体区分",
    "Personality_Group": "个性区分：群体区分",
}

TASK_GUIDANCE = {
    "KP_Weak": {
        "title": "任务说明",
        "goal": "请根据学生历史作答表现，选择最适合用于针对性补弱的候选题。",
        "basis": [
            "优先关注历史中暴露出的薄弱知识点。",
            "不要只看表面相似度，要判断是否真的能补当前最关键的弱点。",
            "如果多个候选都相关，请选你认为最值得优先推荐的一道。",
        ],
    },
    "KP_Stage": {
        "title": "任务说明",
        "goal": "请根据学生历史题目与表现，选择最符合当前学习阶段的候选题。",
        "basis": [
            "重点看候选题与当前学习阶段是否一致，而不是只看难度高低。",
            "优先选择最适合作为当前阶段继续学习的一题。",
            "如果候选题明显过早或过晚，不应优先推荐。",
        ],
    },
    "Planning_Progress": {
        "title": "任务说明",
        "goal": "请根据学生最近学习轨迹，选择最适合作为下一步练习的候选题。",
        "basis": [
            "考虑学生当前更适合推进、巩固，还是先回退修复。",
            "不要选过难或无效重复的题目。",
            "请从教学推进角度选择最合理的一题。",
        ],
    },
    "Planning_Target": {
        "title": "任务说明",
        "goal": "请在给定学习目标下，选择最能推进该目标的候选题。",
        "basis": [
            "优先考虑是否真正服务于学习目标，而不是只看与历史的表面相似。",
            "更看重能否补足目标知识覆盖，而不是重复已经掌握的内容。",
            "请选择最适合作为“朝目标再前进一步”的题目。",
        ],
    },
    "Personality_Individual": {
        "title": "任务说明",
        "goal": "请根据多个候选学生的历史表现，选择最适合当前目标题目的学生。",
        "basis": [
            "比较候选学生与目标题所需知识、能力之间的匹配程度。",
            "不要只看总体成绩，要看谁最适合这道具体题目。",
            "请从“把这道题推荐给谁最合理”的角度做选择。",
        ],
    },
    "Personality_Group": {
        "title": "任务说明",
        "goal": "请根据多个候选学生组的历史表现，选择最适合当前目标题目的学生组。",
        "basis": [
            "考虑小组整体与目标题的匹配程度，而不是只看单个成员。",
            "关注组内能力结构是否适合这道题。",
            "请从“把这道题推荐给哪个组最合理”的角度做选择。",
        ],
    },
}

VISIBLE_CONTEXT_HINT = {"Planning_Target": "学习目标"}

# 这次数据量不大：启动时把所有题目文本枚举转换成 HTML，避免每次渲染时再猜高度/猜公式。
RENDER_CACHE: dict[str, str] = {}

SUPERSCRIPT = str.maketrans("0123456789+-=()nix", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣ")
SUBSCRIPT = str.maketrans("0123456789+-=()nix", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙᵢₓ")


def load_samples() -> pd.DataFrame:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"缺少样本文件: {REVIEW_CSV}")
    df = pd.read_csv(REVIEW_CSV)
    if "row_id" not in df.columns:
        df = df.copy()
        df.insert(0, "row_id", range(1, len(df) + 1))
    return df


def init_state(df: pd.DataFrame) -> None:
    st.session_state.setdefault("annotations", {})
    st.session_state.setdefault("current_index", 0)
    st.session_state.setdefault("reviewer_name", "")
    known_ids = set(df["row_id"].tolist())
    st.session_state.annotations = {
        key: value for key, value in st.session_state.annotations.items() if key in known_ids
    }


def pretty_dimension(benchmark: str) -> str:
    return DIMENSION_LABELS.get(str(benchmark), str(benchmark))


def guidance_for(benchmark: str) -> dict:
    return TASK_GUIDANCE.get(
        benchmark,
        {
            "title": "任务说明",
            "goal": "请根据给定信息，选择你认为最合适的推荐对象。",
            "basis": ["请按照真实教学推荐习惯进行判断。"],
        },
    )


def load_json_items(value: Any) -> list[dict]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_prefixed_lines(text: Any) -> list[tuple[str, str]]:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    parsed: list[tuple[str, str]] = []
    for line in lines:
        prefix, body = "", line
        if ":" in line:
            prefix, body = line.split(":", 1)
        elif "：" in line:
            prefix, body = line.split("：", 1)
        parsed.append((prefix.strip(), body.strip()))
    return parsed


def build_history_items(text: Any) -> list[dict]:
    items = []
    for idx, (_, body) in enumerate(parse_prefixed_lines(text), start=1):
        items.append({"index": idx, "score": "", "total": "", "question_text": body})
    return items


def build_blind_candidates(text: Any) -> list[dict]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidates: list[dict] = []
    for idx, (_, body) in enumerate(parse_prefixed_lines(text)):
        label = letters[idx] if idx < len(letters) else f"Option-{idx + 1}"
        candidates.append({"label": label, "text": body, "qid": "", "is_gt": False})
    return candidates


def _find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _replace_command_two_args(text: str, command: str, renderer) -> str:
    out: list[str] = []
    i = 0
    needle = "\\" + command
    while i < len(text):
        if not text.startswith(needle, i):
            out.append(text[i])
            i += 1
            continue
        j = i + len(needle)
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            out.append(text[i])
            i += 1
            continue
        end1 = _find_matching_brace(text, j)
        if end1 < 0:
            out.append(text[i])
            i += 1
            continue
        k = end1 + 1
        while k < len(text) and text[k].isspace():
            k += 1
        if k >= len(text) or text[k] != "{":
            out.append(text[i])
            i += 1
            continue
        end2 = _find_matching_brace(text, k)
        if end2 < 0:
            out.append(text[i])
            i += 1
            continue
        a = text[j + 1 : end1]
        b = text[k + 1 : end2]
        out.append(renderer(a, b))
        i = end2 + 1
    return "".join(out)


def _replace_command_one_arg(text: str, command: str, renderer) -> str:
    out: list[str] = []
    i = 0
    needle = "\\" + command
    while i < len(text):
        if not text.startswith(needle, i):
            out.append(text[i])
            i += 1
            continue
        j = i + len(needle)
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            out.append(text[i])
            i += 1
            continue
        end = _find_matching_brace(text, j)
        if end < 0:
            out.append(text[i])
            i += 1
            continue
        a = text[j + 1 : end]
        out.append(renderer(a))
        i = end + 1
    return "".join(out)


def _simple_math_escape(s: str) -> str:
    return html.escape(latex_to_readable_text(s), quote=False)


def latex_to_readable_text(raw: Any) -> str:
    """把本题库常见的裸 LaTeX 转成中文题面可读文本。

    关键点：不要求源文本给 $...$，所以比 MathJax 自动识别更稳。
    """
    s = "" if raw is None else str(raw)
    if s.lower() == "nan":
        return ""

    # 先处理带参数命令。
    s = _replace_command_two_args(
        s,
        "dfrac",
        lambda a, b: f"⟦FRAC:{_simple_math_escape(a)}|{_simple_math_escape(b)}⟧",
    )
    s = _replace_command_two_args(
        s,
        "frac",
        lambda a, b: f"⟦FRAC:{_simple_math_escape(a)}|{_simple_math_escape(b)}⟧",
    )
    s = _replace_command_one_arg(s, "sqrt", lambda a: f"√({_simple_math_escape(a)})")
    s = _replace_command_one_arg(s, "boldsymbol", lambda a: latex_to_readable_text(a))
    s = _replace_command_one_arg(s, "mathrm", lambda a: latex_to_readable_text(a))
    s = _replace_command_one_arg(s, "rm", lambda a: latex_to_readable_text(a))
    s = _replace_command_one_arg(s, "text", lambda a: latex_to_readable_text(a))
    s = _replace_command_one_arg(s, "overline", lambda a: f"{latex_to_readable_text(a)}̅")
    s = _replace_command_one_arg(s, "overparen", lambda a: f"⌒{latex_to_readable_text(a)}")

    replacements = {
        r"\vartriangle": "△",
        r"\triangle": "△",
        r"\angle": "∠",
        r"\bot": "⊥",
        r"\perp": "⊥",
        r"\/\!\/": "∥",
        r"/\!/": "∥",
        r"\parallel": "∥",
        r"\times": "×",
        r"\div": "÷",
        r"\cdot": "·",
        r"\boldsymbol{⋅}": "·",
        r"\circ": "°",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\theta": "θ",
        r"\pi": "π",
        r"\odot": "⊙",
        r"\leqslant": "≤",
        r"\leq": "≤",
        r"\geqslant": "≥",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\sim": "∼",
        r"\infty": "∞",
        r"\cdots": "⋯",
        r"\dots": "…",
        r"\quad": "　",
        r"\left": "",
        r"\right": "",
        r"\blacksquare": "■",
        r"\square": "□",
        r"\#": "#",
        r"\\": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # {^\circ} / {^\circ} 这类先转成 °。
    s = re.sub(r"\{\s*\^\s*°\s*\}", "°", s)
    s = re.sub(r"\^\s*\{\s*°\s*\}", "°", s)

    # 常见大括号只是 LaTeX 分组：{a^2} -> a^2，{\left(...\right)^2} -> (... )^2。
    s = re.sub(r"\{([A-Za-z0-9+\-*/=<>≤≥.,，。:：_()（）\[\]αβγθπ°| ]{1,60})\}", r"\1", s)

    # 上下标。复杂内容用括号显示，简单数字/字母用 unicode。
    def sup_repl(m: re.Match) -> str:
        body = m.group(1) or m.group(2)
        body = latex_to_readable_text(body)
        if re.fullmatch(r"[0-9+\-=()nix]+", body):
            return body.translate(SUPERSCRIPT)
        return f"<sup>{html.escape(body)}</sup>"

    def sub_repl(m: re.Match) -> str:
        body = m.group(1) or m.group(2)
        body = latex_to_readable_text(body)
        if re.fullmatch(r"[0-9+\-=()nix]+", body):
            return body.translate(SUBSCRIPT)
        return f"<sub>{html.escape(body)}</sub>"

    s = re.sub(r"\^\{([^{}]{1,40})\}|\^([A-Za-z0-9+\-=()])", sup_repl, s)
    s = re.sub(r"_\{([^{}]{1,40})\}|_([A-Za-z0-9+\-=()])", sub_repl, s)

    # 处理残留的 LaTeX 转义命令，避免页面上出现反斜杠。
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return s


def question_to_html(raw: Any) -> str:
    key = str(raw)
    if key in RENDER_CACHE:
        return RENDER_CACHE[key]

    converted = latex_to_readable_text(raw)
    # 不能整体 escape，因为上面可能插入 <sup>/<sub> 和 FRAC token；先保护 token。
    converted = converted.replace("\r\n", "\n").replace("\r", "\n")
    placeholders: dict[str, str] = {}

    def protect_frac(m: re.Match) -> str:
        token = f"@@FRAC{len(placeholders)}@@"
        num, den = m.group(1), m.group(2)
        placeholders[token] = (
            '<span class="frac"><span class="num">'
            + num
            + '</span><span class="den">'
            + den
            + '</span></span>'
        )
        return token

    converted = re.sub(r"⟦FRAC:(.*?)\|(.*?)⟧", protect_frac, converted)

    # 保护允许的 sup/sub 标签。
    def protect_tag(m: re.Match) -> str:
        token = f"@@TAG{len(placeholders)}@@"
        placeholders[token] = m.group(0)
        return token

    converted = re.sub(r"</?(?:sup|sub)>", protect_tag, converted)
    escaped = html.escape(converted, quote=False)
    escaped = escaped.replace("\n", "<br>")
    for token, value in placeholders.items():
        escaped = escaped.replace(token, value)

    # 选择题选项稍微拉开：A. / B. / C. / D. 单独高亮。
    escaped = re.sub(
        r"(?<![A-Za-z])([ABCD])\.",
        r'<span class="option-mark">\1.</span>',
        escaped,
    )
    RENDER_CACHE[key] = escaped
    return escaped


def enumerate_render_cache(df: pd.DataFrame) -> None:
    texts: list[str] = []
    for col in ["gt_signal", "gt_candidate_text", "history_preview", "candidate_preview"]:
        if col in df.columns:
            texts.extend(str(x) for x in df[col].dropna().tolist())
    for col in ["history_items_json", "candidate_items_json"]:
        if col not in df.columns:
            continue
        for value in df[col].dropna().tolist():
            for item in load_json_items(value):
                text = item.get("question_text")
                if text:
                    texts.append(str(text))
    for text in dict.fromkeys(texts):
        question_to_html(text)


def card_html(title: str, body_html: str, meta: str = "", kind: str = "normal", badge: str = "") -> str:
    badge_html = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
    meta_html = f'<span class="meta">{html.escape(meta)}</span>' if meta else ""
    return f"""
    <section class="q-card {kind}">
      <div class="q-head">
        <div>{badge_html}<span class="q-title">{html.escape(title)}</span></div>
        {meta_html}
      </div>
      <div class="q-body">{body_html}</div>
    </section>
    """


def page_css() -> str:
    return """
    <style>
      :root { color-scheme: light; }
      body {
        margin: 0;
        background: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
        color: #0f172a;
      }
      .wrap { padding: 4px 2px 18px; }
      .guide {
        border: 1px solid #dbeafe;
        background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 14px;
      }
      .guide h3 { margin: 0 0 8px; font-size: 19px; }
      .guide p { margin: 4px 0; line-height: 1.72; }
      .guide ul { margin: 7px 0 0 22px; padding: 0; line-height: 1.7; }
      .section-title {
        font-size: 19px;
        font-weight: 800;
        margin: 18px 0 10px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .section-title:before {
        content: "";
        display: inline-block;
        width: 5px;
        height: 18px;
        border-radius: 999px;
        background: #2563eb;
      }
      .q-card {
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        background: #ffffff;
        padding: 13px 15px 14px;
        margin: 10px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      }
      .q-card.history { border-left: 5px solid #94a3b8; }
      .q-card.candidate { border-left: 5px solid #2563eb; }
      .q-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }
      .q-title { font-weight: 800; font-size: 16px; color: #111827; }
      .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 26px;
        height: 26px;
        margin-right: 8px;
        padding: 0 8px;
        border-radius: 999px;
        background: #2563eb;
        color: white;
        font-weight: 900;
        font-size: 14px;
      }
      .meta {
        flex: none;
        color: #64748b;
        background: #f1f5f9;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 13px;
      }
      .q-body {
        font-size: 18px;
        line-height: 1.86;
        letter-spacing: 0.01em;
        word-break: break-word;
        overflow-wrap: anywhere;
      }
      .frac {
        display: inline-grid;
        grid-template-rows: auto auto;
        align-items: center;
        justify-items: center;
        vertical-align: middle;
        margin: 0 0.12em;
        line-height: 1.05;
        font-size: 0.94em;
      }
      .frac .num {
        border-bottom: 1.5px solid currentColor;
        padding: 0 0.22em 0.08em;
      }
      .frac .den { padding: 0.08em 0.22em 0; }
      sup, sub { line-height: 0; font-size: 0.72em; }
      .option-mark {
        display: inline-block;
        margin-left: 0.55em;
        margin-right: 0.1em;
        padding: 0 0.28em;
        border-radius: 6px;
        background: #eef2ff;
        color: #3730a3;
        font-weight: 800;
      }
    </style>
    """


def render_html_panel(html_body: str, min_height: int = 560) -> None:
    # 直接给足高度，避免 iframe 内部出现二级滚动；题目少，宁可页面长一点。
    height = max(min_height, 220 + html_body.count("q-card") * 170 + html_body.count("<br>") * 18)
    components.html(
        f"<html><head><meta charset='utf-8'>{page_css()}</head><body><main class='wrap'>{html_body}</main></body></html>",
        height=height,
        scrolling=False,
    )


def get_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("筛选")
    dimension_options = ["全部"] + [pretty_dimension(x) for x in sorted(df["benchmark"].dropna().unique().tolist())]
    selected_dimension = st.sidebar.selectbox("维度", dimension_options)
    out = df
    if selected_dimension != "全部":
        reverse_map = {pretty_dimension(x): x for x in df["benchmark"].dropna().unique().tolist()}
        out = out[out["benchmark"] == reverse_map[selected_dimension]]
    return out.reset_index(drop=True)


def context_text(row: pd.Series) -> str:
    benchmark = str(row.get("benchmark", ""))
    if benchmark == "Planning_Target":
        goal = str(row.get("gt_signal", "")).strip()
        if goal and goal.lower() != "nan":
            return goal
    return ""


def get_history_items(row: pd.Series) -> list[dict]:
    items = load_json_items(row.get("history_items_json", ""))
    if items:
        return items
    return build_history_items(row.get("history_preview", ""))


def get_candidates(row: pd.Series) -> list[dict]:
    raw_candidates = load_json_items(row.get("candidate_items_json", ""))
    if raw_candidates:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [
            {
                "label": letters[idx] if idx < len(letters) else f"Option-{idx + 1}",
                "text": str(item.get("question_text", "")),
                "qid": str(item.get("qid", "")),
                # 保留在导出，不展示在界面。
                "is_gt": bool(item.get("is_gt", False)),
            }
            for idx, item in enumerate(raw_candidates)
        ]
    return build_blind_candidates(row.get("candidate_preview", ""))


def build_question_panel(row: pd.Series, candidates: list[dict]) -> str:
    benchmark = str(row.get("benchmark", ""))
    guide = guidance_for(benchmark)
    body = [
        "<div class='guide'>",
        f"<h3>{html.escape(pretty_dimension(benchmark))}</h3>",
        f"<p><b>{html.escape(guide['title'])}</b>：{html.escape(guide['goal'])}</p>",
        "<ul>",
    ]
    for item in guide["basis"]:
        body.append(f"<li>{html.escape(item)}</li>")
    body.append("</ul>")
    extra = context_text(row)
    if extra:
        hint = VISIBLE_CONTEXT_HINT.get(benchmark, "补充信息")
        body.append(f"<p><b>{html.escape(hint)}：</b>{question_to_html(extra)}</p>")
    focus = str(row.get("teacher_check_focus", "")).strip()
    if focus and focus.lower() != "nan":
        body.append(f"<p><b>检查重点：</b>{html.escape(focus)}</p>")
    body.append("</div>")

    body.append("<div class='section-title'>历史信息</div>")
    for item in get_history_items(row):
        idx = item.get("index", "")
        score = item.get("score", "")
        total = item.get("total", "")
        meta = ""
        if score != "" and total != "":
            meta = f"得分 {score}/{total}"
        title = f"历史 {idx}" if idx != "" else "历史"
        body.append(card_html(title, question_to_html(item.get("question_text", "")), meta=meta, kind="history"))

    body.append("<div class='section-title'>候选项</div>")
    for candidate in candidates:
        label = candidate["label"]
        meta = candidate.get("qid", "")
        body.append(
            card_html(
                f"候选 {label}",
                question_to_html(candidate.get("text", "")),
                meta=meta,
                kind="candidate",
                badge=label,
            )
        )
    return "\n".join(body)


def default_annotation(row_id: int) -> dict:
    return {
        "reviewer_name": st.session_state.reviewer_name,
        "teacher_choice_label": "",
        "teacher_choice_text": "",
        "teacher_confidence": 3,
        "teacher_reason": "",
        "teacher_comment": "",
    }


def get_annotation(row_id: int) -> dict:
    return st.session_state.annotations.get(row_id, default_annotation(row_id))


def set_annotation(row_id: int, payload: dict) -> None:
    st.session_state.annotations[row_id] = payload


def render_annotation_form(row: pd.Series, candidates: list[dict]) -> None:
    row_id = int(row["row_id"])
    default = get_annotation(row_id)
    label_to_text = {item["label"]: item["text"] for item in candidates}
    choice_labels = [""] + [item["label"] for item in candidates] + ["无法判断 / 暂不推荐"]

    st.subheader("老师标注")
    reviewer_name = st.text_input("标注人", value=default["reviewer_name"] or st.session_state.reviewer_name)
    st.session_state.reviewer_name = reviewer_name

    selected_label = st.radio(
        "请选择你会推荐的候选项",
        options=choice_labels,
        index=choice_labels.index(default["teacher_choice_label"])
        if default["teacher_choice_label"] in choice_labels
        else 0,
        format_func=lambda x: "请选择" if x == "" else x,
        horizontal=True,
    )
    if selected_label and selected_label in label_to_text:
        with st.expander(f"已选候选 {selected_label} 预览", expanded=True):
            components.html(
                f"<html><head><meta charset='utf-8'>{page_css()}</head><body><div class='q-body'>{question_to_html(label_to_text[selected_label])}</div></body></html>",
                height=220,
                scrolling=True,
            )

    teacher_confidence = st.slider("推荐把握度", 1, 5, int(default["teacher_confidence"]))
    teacher_reason = st.text_area(
        "推荐理由",
        value=default["teacher_reason"],
        height=150,
        placeholder="请说明你为什么会推荐这个候选项。",
    )
    teacher_comment = st.text_area(
        "补充备注",
        value=default["teacher_comment"],
        height=120,
        placeholder="可记录歧义、样本问题、其他备选意见等。",
    )

    payload = {
        "reviewer_name": reviewer_name,
        "teacher_choice_label": selected_label,
        "teacher_choice_text": label_to_text.get(selected_label, selected_label),
        "teacher_confidence": teacher_confidence,
        "teacher_reason": teacher_reason,
        "teacher_comment": teacher_comment,
    }
    set_annotation(row_id, payload)


def render_progress(filtered: pd.DataFrame) -> None:
    selected_ids = set(filtered["row_id"].tolist())
    annotated = 0
    for row_id in selected_ids:
        item = st.session_state.annotations.get(row_id)
        if item and item.get("teacher_choice_label"):
            annotated += 1
    st.sidebar.metric("当前筛选下已标注", f"{annotated}/{len(filtered)}")


def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = df.set_index("row_id").to_dict(orient="index")
    for row_id, annotation in st.session_state.annotations.items():
        base = by_id.get(row_id, {})
        rows.append(
            {
                "row_id": row_id,
                **base,
                **annotation,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def save_local_annotations(export_df: pd.DataFrame, reviewer_name: str) -> Path:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = reviewer_name.strip() or "anonymous"
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in safe_name)
    path = ANNOTATION_DIR / f"{safe_name}_annotations.csv"
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def inject_streamlit_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1480px; }
          div[data-testid="stVerticalBlock"] { gap: 0.65rem; }
          section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
          .stRadio [role="radiogroup"] { gap: 0.38rem 0.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="老师推荐标注", layout="wide")
    inject_streamlit_css()
    st.title("老师推荐标注")
    st.caption("已按当前 CSV 穷举预渲染题目文本；裸 LaTeX 会转成可读 HTML，不再依赖题目自带 $...$。")

    df = load_samples()
    enumerate_render_cache(df)
    init_state(df)
    filtered = get_filtered_df(df)

    if filtered.empty:
        st.warning("当前筛选下没有样本。")
        return

    render_progress(filtered)

    if st.session_state.current_index >= len(filtered):
        st.session_state.current_index = 0

    row_ids = filtered["row_id"].tolist()
    selected_row_id = st.sidebar.selectbox(
        "跳转到样本",
        row_ids,
        index=min(st.session_state.current_index, len(row_ids) - 1),
        format_func=lambda rid: f"{rid} | {pretty_dimension(filtered[filtered['row_id'] == rid].iloc[0]['benchmark'])}",
    )
    st.session_state.current_index = int(filtered.index[filtered["row_id"] == selected_row_id][0])

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    if nav1.button("上一条", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    if nav2.button("下一条", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
    nav3.progress((st.session_state.current_index + 1) / len(filtered))
    st.caption(f"样本 {st.session_state.current_index + 1} / {len(filtered)}")

    row = filtered.iloc[st.session_state.current_index]
    candidates = get_candidates(row)

    left, right = st.columns([1.55, 1.0], gap="large")
    with left:
        render_html_panel(build_question_panel(row, candidates))
    with right:
        render_annotation_form(row, candidates)

    st.divider()
    export_df = build_export_df(df)
    st.subheader("导出")
    st.write(f"当前会话已保存标注：{len(export_df)}")
    download_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下载标注 CSV",
        data=download_bytes,
        file_name="teacher_recommendation_annotations.csv",
        mime="text/csv",
    )
    if st.button("保存到本地文件"):
        path = save_local_annotations(export_df, st.session_state.reviewer_name)
        st.success(f"已保存到 {path}")
    with st.expander("当前会话标注预览", expanded=False):
        st.dataframe(export_df, use_container_width=True)


if __name__ == "__main__":
    main()
