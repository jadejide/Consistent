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
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def guidance_for(benchmark: str) -> dict[str, Any]:
    return TASK_GUIDANCE.get(
        str(benchmark),
        {"title": "任务说明", "goal": "请根据给定信息，选择你认为最合适的推荐对象。", "basis": ["请按照真实教学推荐习惯进行判断。"]},
    )


def load_json_items(value: Any) -> list[dict[str, Any]]:
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


def build_history_items(row: pd.Series) -> list[dict[str, Any]]:
    json_items = load_json_items(row.get("history_items_json", ""))
    if json_items:
        return [
            {
                "index": item.get("index", idx),
                "text": str(item.get("question_text", "")),
                "score": item.get("score", ""),
                "total": item.get("total", ""),
                "qid": item.get("qid", ""),
            }
            for idx, item in enumerate(json_items, start=1)
        ]
    return [
        {"index": idx, "text": body, "score": "", "total": "", "qid": ""}
        for idx, (_, body) in enumerate(parse_prefixed_lines(row.get("history_preview", "")), start=1)
    ]


def build_candidates(row: pd.Series) -> list[dict[str, Any]]:
    json_items = load_json_items(row.get("candidate_items_json", ""))
    if json_items:
        return [
            {
                "label": LETTERS[idx] if idx < len(LETTERS) else f"Option-{idx + 1}",
                "text": str(item.get("question_text", "")),
                "qid": str(item.get("qid", "")),
                "is_gt": bool(item.get("is_gt", False)),
            }
            for idx, item in enumerate(json_items)
        ]
    return [
        {"label": LETTERS[idx] if idx < len(LETTERS) else f"Option-{idx + 1}", "text": body, "qid": "", "is_gt": False}
        for idx, (_, body) in enumerate(parse_prefixed_lines(row.get("candidate_preview", "")))
    ]


def context_text(row: pd.Series) -> str:
    if str(row.get("benchmark", "")) == "Planning_Target":
        return str(row.get("gt_signal", "")).strip()
    return ""


def safe_text(value: Any) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def question_card_html(
    title: str,
    text: Any,
    *,
    badge: str = "",
    meta: str = "",
    tone: str = "neutral",
) -> str:
    tone_class = {
        "history": "card-history",
        "candidate": "card-candidate",
        "target": "card-target",
    }.get(tone, "card-neutral")
    badge_html = f'<span class="badge">{safe_text(badge)}</span>' if badge else ""
    meta_html = f'<span class="meta">{safe_text(meta)}</span>' if meta else ""
    return f"""
    <article class="q-card {tone_class}">
      <header class="q-head">
        <div class="q-title">{safe_text(title)} {badge_html}</div>
        {meta_html}
      </header>
      <div class="q-body tex2jax_process">{safe_text(text)}</div>
    </article>
    """


def page_html(row: pd.Series, history_items: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    benchmark = str(row.get("benchmark", ""))
    guide = guidance_for(benchmark)
    basis_items = "".join(f"<li>{safe_text(item)}</li>" for item in guide["basis"])

    extra_context = context_text(row)
    context_block = ""
    if extra_context:
        context_block = question_card_html(
            VISIBLE_CONTEXT_HINT.get(benchmark, "补充信息"), extra_context, tone="target"
        )

    history_cards = []
    for item in history_items:
        score = ""
        if item.get("score") != "" or item.get("total") != "":
            score = f"得分 {item.get('score', '')}/{item.get('total', '')}"
        qid = f"题号 {item.get('qid')}" if item.get("qid") else ""
        meta = " · ".join(x for x in [score, qid] if x)
        history_cards.append(
            question_card_html(f"历史 {item.get('index', '')}", item.get("text", ""), meta=meta, tone="history")
        )

    candidate_cards = []
    for item in candidates:
        meta = f"题号 {item.get('qid')}" if item.get("qid") else ""
        # 不展示 is_gt，避免标注泄漏。只保留在导出/调试数据里。
        candidate_cards.append(
            question_card_html(
                f"候选 {item['label']}", item.get("text", ""), badge=item["label"], meta=meta, tone="candidate"
            )
        )

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script>
        window.MathJax = {{
          tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
          }},
          svg: {{ fontCache: 'global' }},
          options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
        }};
      </script>
      <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
      <style>
        :root {{
          --border: #e5e7eb;
          --muted: #6b7280;
          --text: #111827;
          --bg-soft: #f9fafb;
          --blue-soft: #eff6ff;
          --blue-border: #bfdbfe;
          --amber-soft: #fffbeb;
          --amber-border: #fde68a;
          --green-soft: #f0fdf4;
          --green-border: #bbf7d0;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          padding: 0 4px 18px 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
          color: var(--text);
          background: white;
        }}
        .top-card {{
          border: 1px solid var(--border);
          background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
          border-radius: 18px;
          padding: 16px 18px;
          margin-bottom: 14px;
        }}
        .dim {{
          font-size: 22px;
          font-weight: 800;
          letter-spacing: .2px;
          margin-bottom: 8px;
        }}
        .goal {{
          font-size: 17px;
          line-height: 1.65;
          margin: 8px 0 10px;
        }}
        .basis-title {{
          font-weight: 750;
          margin-top: 10px;
        }}
        ul {{ margin: 8px 0 0 22px; padding: 0; }}
        li {{ line-height: 1.7; margin: 2px 0; }}
        .section-title {{
          display: flex;
          align-items: center;
          gap: 9px;
          margin: 20px 0 10px;
          font-size: 20px;
          font-weight: 800;
        }}
        .section-title:before {{
          content: "";
          width: 5px;
          height: 21px;
          border-radius: 999px;
          background: #2563eb;
        }}
        .q-card {{
          border: 1px solid var(--border);
          border-radius: 18px;
          margin: 10px 0 14px;
          padding: 0;
          overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,.045);
        }}
        .card-history {{ background: var(--bg-soft); }}
        .card-candidate {{ background: #fff; border-color: var(--blue-border); }}
        .card-target {{ background: var(--green-soft); border-color: var(--green-border); }}
        .q-head {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 10px 14px;
          border-bottom: 1px solid rgba(229,231,235,.9);
          background: rgba(255,255,255,.72);
        }}
        .q-title {{
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 800;
          font-size: 16px;
          white-space: nowrap;
        }}
        .badge {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 27px;
          height: 27px;
          padding: 0 9px;
          border-radius: 999px;
          color: #1d4ed8;
          background: #dbeafe;
          font-weight: 850;
        }}
        .meta {{
          color: var(--muted);
          font-size: 13px;
          text-align: right;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }}
        .q-body {{
          font-size: 18px;
          line-height: 1.82;
          padding: 15px 16px 18px;
          word-break: break-word;
          overflow-wrap: anywhere;
        }}
        .card-candidate .q-body {{ font-size: 19px; }}
        mjx-container[jax="SVG"][display="true"] {{
          overflow-x: auto;
          overflow-y: hidden;
          max-width: 100%;
          padding: 4px 0;
        }}
        mjx-container {{ outline: none; }}
      </style>
    </head>
    <body>
      <section class="top-card">
        <div class="dim">{safe_text(pretty_dimension(benchmark))}</div>
        <div class="basis-title">{safe_text(guide['title'])}</div>
        <div class="goal">{safe_text(guide['goal'])}</div>
        <div class="basis-title">推荐依据</div>
        <ul>{basis_items}</ul>
      </section>
      {context_block}
      <div class="section-title">历史信息</div>
      {''.join(history_cards) if history_cards else '<div class="q-card"><div class="q-body">暂无历史信息</div></div>'}
      <div class="section-title">候选题目</div>
      {''.join(candidate_cards) if candidate_cards else '<div class="q-card"><div class="q-body">暂无候选项</div></div>'}
    </body>
    </html>
    """


def render_question_panel(row: pd.Series, history_items: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    # 只用一个 iframe 渲染所有题目，避免每道题一个 components.html 导致高度错乱和闪烁。
    item_count = len(history_items) + len(candidates)
    text_len = sum(len(str(x.get("text", ""))) for x in history_items + candidates)
    height = min(3600, max(760, 430 + item_count * 145 + text_len // 12))
    components.html(page_html(row, history_items, candidates), height=height, scrolling=True)


def get_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("筛选")
    dimension_options = ["全部"] + [pretty_dimension(x) for x in sorted(df["benchmark"].dropna().unique().tolist())]
    selected_dimension = st.sidebar.selectbox("维度", dimension_options)

    out = df
    if selected_dimension != "全部":
        reverse_map = {pretty_dimension(x): x for x in df["benchmark"].dropna().unique().tolist()}
        out = out[out["benchmark"] == reverse_map[selected_dimension]]
    return out.reset_index(drop=True)


def default_annotation(row_id: int) -> dict[str, Any]:
    return {
        "reviewer_name": st.session_state.reviewer_name,
        "teacher_choice_label": "",
        "teacher_choice_text": "",
        "teacher_confidence": 3,
        "teacher_reason": "",
        "teacher_comment": "",
    }


def get_annotation(row_id: int) -> dict[str, Any]:
    return st.session_state.annotations.get(row_id, default_annotation(row_id))


def set_annotation(row_id: int, payload: dict[str, Any]) -> None:
    st.session_state.annotations[row_id] = payload


def render_annotation_form(row: pd.Series, candidates: list[dict[str, Any]]) -> None:
    row_id = int(row["row_id"])
    default = get_annotation(row_id)
    label_to_text = {item["label"]: item["text"] for item in candidates}
    choice_labels = [""] + [item["label"] for item in candidates] + ["无法判断 / 暂不推荐"]

    with st.container(border=True):
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
        )

        selected_text = label_to_text.get(selected_label, "")
        if selected_text:
            with st.expander("查看已选候选题", expanded=True):
                st.markdown(f"**候选 {selected_label}**")
                st.write(selected_text)

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
            height=110,
            placeholder="可记录歧义、样本问题、其他备选意见等。",
        )

    set_annotation(
        row_id,
        {
            "reviewer_name": reviewer_name,
            "teacher_choice_label": selected_label,
            "teacher_choice_text": label_to_text.get(selected_label, selected_label),
            "teacher_confidence": teacher_confidence,
            "teacher_reason": teacher_reason,
            "teacher_comment": teacher_comment,
        },
    )


def render_progress(filtered: pd.DataFrame) -> None:
    selected_ids = set(filtered["row_id"].tolist())
    annotated = sum(
        1
        for row_id in selected_ids
        if st.session_state.annotations.get(row_id, {}).get("teacher_choice_label")
    )
    st.sidebar.metric("当前筛选下已标注", f"{annotated}/{len(filtered)}")
    if len(filtered):
        st.sidebar.progress(annotated / len(filtered))


def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = df.set_index("row_id").to_dict(orient="index")
    for row_id, annotation in st.session_state.annotations.items():
        base = by_id.get(row_id, {})
        rows.append({"row_id": row_id, **base, **annotation, "saved_at": datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(rows)


def save_local_annotations(export_df: pd.DataFrame, reviewer_name: str) -> Path:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = reviewer_name.strip() or "anonymous"
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in safe_name)
    path = ANNOTATION_DIR / f"{safe_name}_annotations.csv"
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.4rem; max-width: 1500px; }
          [data-testid="stSidebar"] { min-width: 285px; }
          div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
          .stRadio [role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 6px 8px;
          }
          .stRadio label {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 8px 10px;
            background: #fff;
          }
          .stTextArea textarea { line-height: 1.55; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="老师推荐标注", layout="wide")
    inject_app_css()

    st.title("老师推荐标注")
    st.caption("请根据任务说明、历史信息和候选题目，像真实教学推荐一样做选择。页面不会展示模型结果。")

    df = load_samples()
    init_state(df)
    filtered = get_filtered_df(df)

    if filtered.empty:
        st.warning("当前筛选下没有样本。")
        return

    render_progress(filtered)

    current_index = min(st.session_state.current_index, len(filtered) - 1)
    row_ids = filtered["row_id"].tolist()
    selected_row_id = st.sidebar.selectbox(
        "跳转到样本",
        row_ids,
        index=current_index,
        format_func=lambda rid: f"{rid} | {pretty_dimension(filtered[filtered['row_id'] == rid].iloc[0]['benchmark'])}",
    )
    st.session_state.current_index = int(filtered.index[filtered["row_id"] == selected_row_id][0])
    row = filtered.iloc[st.session_state.current_index]

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 4, 1.4])
    if nav1.button("上一条", use_container_width=True, disabled=st.session_state.current_index <= 0):
        st.session_state.current_index -= 1
        st.rerun()
    if nav2.button("下一条", use_container_width=True, disabled=st.session_state.current_index >= len(filtered) - 1):
        st.session_state.current_index += 1
        st.rerun()
    nav3.progress((st.session_state.current_index + 1) / len(filtered))
    nav4.caption(f"样本 {st.session_state.current_index + 1} / {len(filtered)}")

    history_items = build_history_items(row)
    candidates = build_candidates(row)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        render_question_panel(row, history_items, candidates)
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
        use_container_width=True,
    )

    if st.button("保存到本地文件"):
        path = save_local_annotations(export_df, st.session_state.reviewer_name)
        st.success(f"已保存到 {path}")

    with st.expander("当前会话标注预览", expanded=False):
        st.dataframe(export_df, use_container_width=True)


if __name__ == "__main__":
    main()
