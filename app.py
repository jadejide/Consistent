from __future__ import annotations

import html
import json
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

BENCHMARK_LABELS = {
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
            "优先关注历史中暴露出的薄弱知识点。候选题如果能针对这些知识点进行练习，则值得推荐，难度高低暂时不是首要考虑的因素。",
            "请选你您认为最值得优先推荐的一道。",
        ],
    },
    "KP_Stage": {
        "title": "任务说明",
        "goal": "请根据学生历史题目与表现，选择最符合当前教学章节的候选题。",
        "basis": [
            "重点看候选题与当前教学章节是否一致，教学章节以历史出现频率最高的为准，不考虑难度高低。",
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


def load_samples() -> pd.DataFrame:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"缺少样本文件：{REVIEW_CSV}")
    df = pd.read_csv(REVIEW_CSV)
    if "row_id" not in df.columns:
        df = df.copy()
        df.insert(0, "row_id", [f"row_{idx + 1}" for idx in range(len(df))])
    df["row_id"] = df["row_id"].astype(str)
    return df


def init_state(df: pd.DataFrame) -> None:
    st.session_state.setdefault("annotations", {})
    st.session_state.setdefault("current_index", 0)
    known_ids = set(df["row_id"].tolist())
    st.session_state.annotations = {
        key: value for key, value in st.session_state.annotations.items() if key in known_ids
    }


def pretty_benchmark(benchmark: str) -> str:
    return BENCHMARK_LABELS.get(str(benchmark), str(benchmark))


def guidance_for(benchmark: str) -> dict[str, Any]:
    return TASK_GUIDANCE.get(
        benchmark,
        {
            "title": "任务说明",
            "goal": "请根据给定信息，选择你认为最合适的推荐对象。",
            "basis": ["请按真实教学推荐习惯进行判断。"],
        },
    )


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def parse_json_value(value: Any) -> Any:
    if is_blank(value):
        return ""
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return ""


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_value(value)
    return parsed if isinstance(parsed, list) else []


def parse_json_dict(value: Any) -> dict[str, Any]:
    parsed = parse_json_value(value)
    return parsed if isinstance(parsed, dict) else {}


def text_or_empty(value: Any) -> str:
    return "" if is_blank(value) else str(value)


def html_text(text: Any) -> str:
    content = text_or_empty(text)
    escaped = html.escape(content, quote=False).replace("\r\n", "\n").replace("\r", "\n")
    return escaped.replace("\n", "<br>")


def compact_items_meta(item: dict[str, Any]) -> str:
    meta_parts: list[str] = []
    qid = text_or_empty(item.get("qid"))
    if qid:
        meta_parts.append(qid)
    score = text_or_empty(item.get("score"))
    total = text_or_empty(item.get("total"))
    if score or total:
        meta_parts.append(f"得分 {score}/{total}")
    chapter = text_or_empty(item.get("chapter"))
    if chapter:
        meta_parts.append(f"章节 {chapter}")
    knowledge_points = item.get("knowledge_points")
    if isinstance(knowledge_points, list) and knowledge_points:
        meta_parts.append("知识点：" + "；".join(str(x) for x in knowledge_points))
    return " | ".join(meta_parts)


def card_html(title: str, body_html: str, meta: str = "", badge: str = "", kind: str = "normal") -> str:
    badge_html = f"<span class='badge'>{html.escape(badge)}</span>" if badge else ""
    meta_html = f"<span class='meta'>{html.escape(meta)}</span>" if meta else ""
    return f"""
    <section class="card {kind}">
      <div class="card-head">
        <div class="card-title">{badge_html}{html.escape(title)}</div>
        {meta_html}
      </div>
      <div class="card-body">{body_html}</div>
    </section>
    """


def section_title_html(title: str) -> str:
    return f"<div class='section-title'>{html.escape(title)}</div>"


def page_css() -> str:
    return """
    <style>
      :root { color-scheme: light; }
      body {
        margin: 0;
        background: #f8fafc;
        color: #0f172a;
        font-family: "PingFang SC", "Microsoft YaHei", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .wrap { padding: 8px 4px 18px; }
      .guide {
        border: 1px solid #dbeafe;
        background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 16px;
      }
      .guide h3 {
        margin: 0 0 8px;
        font-size: 20px;
      }
      .guide p {
        margin: 5px 0;
        line-height: 1.8;
        font-size: 16px;
      }
      .guide ul {
        margin: 8px 0 0 20px;
        padding: 0;
        line-height: 1.8;
        font-size: 15px;
      }
      .context-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }
      .context-box {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        background: #ffffff;
        padding: 10px 12px;
      }
      .context-box .label {
        display: block;
        color: #475569;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 5px;
      }
      .context-box .value {
        font-size: 15px;
        line-height: 1.8;
      }
      .section-title {
        margin: 18px 0 10px;
        font-size: 19px;
        font-weight: 800;
        color: #0f172a;
        border-left: 5px solid #2563eb;
        padding-left: 10px;
      }
      .card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        background: #ffffff;
        padding: 13px 15px 14px;
        margin: 10px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      }
      .card.history { border-left: 5px solid #94a3b8; }
      .card.candidate { border-left: 5px solid #2563eb; }
      .card.target { border-left: 5px solid #0f766e; }
      .card.entity { border-left: 5px solid #7c3aed; }
      .card.member-history { border-left: 4px solid #c4b5fd; background: #faf5ff; }
      .card-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 8px;
      }
      .card-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
      }
      .card-body {
        font-size: 17px;
        line-height: 1.85;
        word-break: break-word;
        overflow-wrap: anywhere;
        white-space: normal;
      }
      .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 26px;
        height: 26px;
        padding: 0 8px;
        margin-right: 8px;
        border-radius: 999px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 800;
        font-size: 14px;
      }
      .meta {
        flex: none;
        color: #64748b;
        background: #f1f5f9;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 12px;
        line-height: 1.5;
      }
      .subtle {
        color: #475569;
        font-size: 15px;
      }
      .history-group {
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed #d8b4fe;
      }
      .history-group:first-of-type {
        margin-top: 0;
        padding-top: 0;
        border-top: none;
      }
    </style>
    """


def katex_head() -> str:
    return """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
      onload="renderMathInElement(document.body, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '\\\\[', right: '\\\\]', display: true},
          {left: '\\\\(', right: '\\\\)', display: false}
        ],
        throwOnError: false
      });">
    </script>
    """


def render_html_panel(html_body: str, min_height: int = 720) -> None:
    height = max(min_height, 260 + html_body.count("class=\"card") * 150 + html_body.count("<br>") * 12)
    components.html(
        f"""
        <html>
          <head>
            <meta charset="utf-8">
            {katex_head()}
            {page_css()}
          </head>
          <body>
            <main class="wrap">{html_body}</main>
          </body>
        </html>
        """,
        height=height,
        scrolling=False,
    )


def get_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("筛选")
    benchmark_values = sorted(df["benchmark"].dropna().unique().tolist())
    benchmark_options = ["全部"] + [pretty_benchmark(x) for x in benchmark_values]
    selected_benchmark = st.sidebar.selectbox("维度", benchmark_options)

    case_type_options = ["全部"] + sorted(df["case_type"].dropna().unique().tolist())
    selected_case_type = st.sidebar.selectbox("样本类型", case_type_options)

    filtered = df
    if selected_benchmark != "全部":
        reverse_map = {pretty_benchmark(x): x for x in benchmark_values}
        filtered = filtered[filtered["benchmark"] == reverse_map[selected_benchmark]]
    if selected_case_type != "全部":
        filtered = filtered[filtered["case_type"] == selected_case_type]
    return filtered.reset_index(drop=True)


def build_context_blocks(row: pd.Series) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    if text_or_empty(row.get("learning_goal")):
        blocks.append(("学习目标", text_or_empty(row.get("learning_goal"))))
    if text_or_empty(row.get("progress_state")):
        blocks.append(("学习进度状态", text_or_empty(row.get("progress_state"))))
    if text_or_empty(row.get("weak_knowledge_points")):
        blocks.append(("薄弱知识点", text_or_empty(row.get("weak_knowledge_points"))))
    if text_or_empty(row.get("stage_knowledge_points")):
        blocks.append(("阶段信息", text_or_empty(row.get("stage_knowledge_points"))))
    if text_or_empty(row.get("teacher_check_focus")):
        blocks.append(("核验重点", text_or_empty(row.get("teacher_check_focus"))))
    return blocks


def get_history_items(row: pd.Series) -> list[dict[str, Any]]:
    return parse_json_list(row.get("history_items_json", ""))


def get_question_candidates(row: pd.Series) -> list[dict[str, Any]]:
    candidates = parse_json_list(row.get("candidate_items_json", ""))
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        normalized.append(
            {
                "label": text_or_empty(item.get("label")) or chr(ord("A") + idx),
                "display_index": item.get("display_index", idx + 1),
                "qid": text_or_empty(item.get("qid")),
                "question_text": text_or_empty(item.get("question_text")),
                "knowledge_points": item.get("knowledge_points", []),
                "chapter": text_or_empty(item.get("chapter")),
            }
        )
    return normalized


def get_target_question(row: pd.Series) -> dict[str, Any]:
    return parse_json_dict(row.get("target_question_json", ""))


def get_entity_candidates(row: pd.Series) -> list[dict[str, Any]]:
    entities = parse_json_list(row.get("candidate_entities_json", ""))
    normalized: list[dict[str, Any]] = []
    for idx, entity in enumerate(entities):
        normalized.append(
            {
                "label": text_or_empty(entity.get("label")) or chr(ord("A") + idx),
                "entity_id": text_or_empty(entity.get("entity_id")),
                "entity_type": text_or_empty(entity.get("entity_type")),
                "title": text_or_empty(entity.get("title")) or f"候选对象 {idx + 1}",
                "summary": text_or_empty(entity.get("summary")),
                "members": entity.get("members", []),
                "history_items": entity.get("history_items", []),
            }
        )
    return normalized


def build_guide_html(row: pd.Series) -> str:
    benchmark = str(row.get("benchmark", ""))
    guide = guidance_for(benchmark)
    blocks = build_context_blocks(row)
    body = [
        "<div class='guide'>",
        f"<h3>{html.escape(pretty_benchmark(benchmark))}</h3>",
        f"<p><b>{html.escape(guide['title'])}</b>：{html.escape(guide['goal'])}</p>",
        "<ul>",
    ]
    for item in guide["basis"]:
        body.append(f"<li>{html.escape(item)}</li>")
    body.append("</ul>")
    if blocks:
        body.append("<div class='context-grid'>")
        for label, value in blocks:
            body.append(
                f"<div class='context-box'><span class='label'>{html.escape(label)}</span>"
                f"<div class='value'>{html_text(value)}</div></div>"
            )
        body.append("</div>")
    body.append("</div>")
    return "\n".join(body)


def build_question_task_panel(row: pd.Series) -> str:
    parts: list[str] = [build_guide_html(row)]

    history_items = get_history_items(row)
    parts.append(section_title_html("历史题目"))
    for item in history_items:
        title = f"历史 {item.get('index', '')}".strip()
        parts.append(
            card_html(
                title=title or "历史题目",
                body_html=html_text(item.get("question_text", "")),
                meta=compact_items_meta(item),
                kind="history",
            )
        )

    candidates = get_question_candidates(row)
    parts.append(section_title_html("候选题目"))
    for item in candidates:
        parts.append(
            card_html(
                title=f"候选题 {item['label']}",
                body_html=html_text(item.get("question_text", "")),
                meta=compact_items_meta(item),
                badge=item["label"],
                kind="candidate",
            )
        )
    return "\n".join(parts)


def build_entity_history_html(entity: dict[str, Any]) -> str:
    history_items = entity.get("history_items", [])
    if not isinstance(history_items, list) or not history_items:
        return "<div class='subtle'>未提供结构化历史记录。</div>"

    if entity.get("entity_type") == "group":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in history_items:
            member_label = text_or_empty(item.get("member_label")) or "成员"
            grouped.setdefault(member_label, []).append(item)
        sections: list[str] = []
        for member_label, items in grouped.items():
            member_lines = [f"<div class='history-group'><div class='subtle'><b>成员 {html.escape(member_label)}</b></div>"]
            for history_item in items:
                meta = compact_items_meta(history_item)
                body = html_text(history_item.get("question_text", ""))
                member_lines.append(
                    f"<div class='subtle' style='margin-top:6px;'><b>历史 {html.escape(str(history_item.get('index', '')))}</b>"
                    f"{'｜' + html.escape(meta) if meta else ''}</div>"
                )
                member_lines.append(f"<div>{body}</div>")
            member_lines.append("</div>")
            sections.append("".join(member_lines))
        return "".join(sections)

    lines: list[str] = []
    for history_item in history_items:
        meta = compact_items_meta(history_item)
        lines.append(
            f"<div class='subtle' style='margin-top:8px;'><b>历史 {html.escape(str(history_item.get('index', '')))}</b>"
            f"{'｜' + html.escape(meta) if meta else ''}</div>"
        )
        lines.append(f"<div>{html_text(history_item.get('question_text', ''))}</div>")
    return "".join(lines)


def build_entity_task_panel(row: pd.Series) -> str:
    parts: list[str] = [build_guide_html(row)]

    target_question = get_target_question(row)
    parts.append(section_title_html("目标题目"))
    parts.append(
        card_html(
            title="目标题目",
            body_html=html_text(target_question.get("question_text", row.get("target_question_text", ""))),
            meta=compact_items_meta(target_question),
            kind="target",
        )
    )

    parts.append(section_title_html("候选对象"))
    entities = get_entity_candidates(row)
    for entity in entities:
        members = entity.get("members", [])
        meta_parts = []
        if entity.get("entity_id"):
            meta_parts.append(entity["entity_id"])
        if isinstance(members, list) and members:
            meta_parts.append("成员：" + "、".join(str(x) for x in members))
        summary_block = ""
        if entity.get("summary"):
            summary_block = f"<div style='margin-bottom:10px;'>{html_text(entity['summary'])}</div>"
        body_html = summary_block + build_entity_history_html(entity)
        parts.append(
            card_html(
                title=entity.get("title", f"候选对象 {entity['label']}"),
                body_html=body_html,
                meta=" | ".join(meta_parts),
                badge=entity["label"],
                kind="entity",
            )
        )
    return "\n".join(parts)


def build_main_panel(row: pd.Series) -> str:
    task_type = str(row.get("task_type", ""))
    if task_type == "entity_recommendation":
        return build_entity_task_panel(row)
    return build_question_task_panel(row)


def default_annotation() -> dict[str, Any]:
    return {
        "teacher_choice_label": "",
        "teacher_choice_text": "",
        "teacher_confidence": 3,
        "teacher_reason": "",
        "teacher_comment": "",
    }


def get_annotation(row_id: str) -> dict[str, Any]:
    return st.session_state.annotations.get(row_id, default_annotation())


def set_annotation(row_id: str, payload: dict[str, Any]) -> None:
    st.session_state.annotations[row_id] = payload


def choice_options_for_row(row: pd.Series) -> list[tuple[str, str]]:
    task_type = str(row.get("task_type", ""))
    if task_type == "entity_recommendation":
        entities = get_entity_candidates(row)
        return [(item["label"], item.get("title", item["label"])) for item in entities]
    candidates = get_question_candidates(row)
    return [(item["label"], item.get("question_text", "")) for item in candidates]


def render_annotation_form(row: pd.Series) -> None:
    row_id = str(row["row_id"])
    default = get_annotation(row_id)
    options = choice_options_for_row(row)
    labels = [""] + [label for label, _ in options] + ["无法判断 / 暂不推荐"]
    label_to_text = {label: text for label, text in options}

    st.subheader("教师推荐标注")
    selected_label = st.radio(
        "请选择你会推荐的对象",
        options=labels,
        index=labels.index(default["teacher_choice_label"]) if default["teacher_choice_label"] in labels else 0,
        format_func=lambda value: "请选择" if value == "" else value,
        horizontal=True,
    )

    if selected_label and selected_label in label_to_text:
        st.caption("当前选择预览")
        st.markdown(
            f"""
            <div style="border:1px solid #dbeafe;border-radius:14px;padding:12px 14px;background:#f8fbff;line-height:1.8;">
              <b>{html.escape(selected_label)}</b><br>
              {html_text(label_to_text[selected_label])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    teacher_confidence = st.slider("推荐把握度", 1, 5, int(default["teacher_confidence"]))
    teacher_reason = st.text_area(
        "推荐理由",
        value=default["teacher_reason"],
        height=160,
        placeholder="请说明你为什么会推荐这个对象。",
    )
    teacher_comment = st.text_area(
        "补充备注",
        value=default["teacher_comment"],
        height=120,
        placeholder="可记录歧义、数据问题、其他候选意见等。",
    )

    payload = {
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


def save_local_annotations(export_df: pd.DataFrame) -> Path:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATION_DIR / "teacher_annotations_latest.csv"
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def inject_streamlit_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
          div[data-testid="stVerticalBlock"] { gap: 0.65rem; }
          section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.65rem; }
          .stRadio [role="radiogroup"] { gap: 0.4rem 0.7rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="教师推荐标注", layout="wide")
    inject_streamlit_css()
    st.title("教师推荐标注")

    df = load_samples()
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
        format_func=lambda rid: f"{rid} | {pretty_benchmark(filtered[filtered['row_id'] == rid].iloc[0]['benchmark'])}",
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

    left, right = st.columns([1.7, 1.0], gap="large")
    with left:
        render_html_panel(build_main_panel(row))
    with right:
        render_annotation_form(row)

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
        path = save_local_annotations(export_df)
        st.success(f"已保存到 {path}")
    with st.expander("当前会话标注预览", expanded=False):
        st.dataframe(export_df, use_container_width=True)


if __name__ == "__main__":
    main()
