from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

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

VISIBLE_CONTEXT_HINT = {
    "Planning_Target": "学习目标",
}


def load_samples() -> pd.DataFrame:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"缺少样本文件: {REVIEW_CSV}")
    df = pd.read_csv(REVIEW_CSV)
    if "row_id" not in df.columns:
        df = df.copy()
        df.insert(0, "row_id", range(1, len(df) + 1))
    return df


def init_state(df: pd.DataFrame) -> None:
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "reviewer_name" not in st.session_state:
        st.session_state.reviewer_name = ""
    known_ids = set(df["row_id"].tolist())
    st.session_state.annotations = {
        key: value for key, value in st.session_state.annotations.items() if key in known_ids
    }


def pretty_dimension(benchmark: str) -> str:
    return DIMENSION_LABELS.get(benchmark, benchmark)


def guidance_for(benchmark: str) -> dict:
    return TASK_GUIDANCE.get(
        benchmark,
        {
            "title": "任务说明",
            "goal": "请根据给定信息，选择你认为最合适的推荐对象。",
            "basis": ["请按照真实教学推荐习惯进行判断。"],
        },
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


def parse_prefixed_lines(text: str) -> list[tuple[str, str]]:
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


def build_blind_candidates(text: str) -> list[dict]:
    parsed = parse_prefixed_lines(text)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidates: list[dict] = []
    for idx, (_, body) in enumerate(parsed):
        label = letters[idx] if idx < len(letters) else f"Option-{idx + 1}"
        candidates.append({"label": label, "text": body})
    return candidates


def build_history_items(text: str) -> list[str]:
    return [body for _, body in parse_prefixed_lines(text)]


def load_json_items(value: str) -> list[dict]:
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def math_html(text: str, font_size: int = 18) -> str:
    safe = html.escape(str(text)).replace("\n", "<br>")
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <script>
          window.MathJax = {{
            tex: {{
              inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
              displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }},
            svg: {{ fontCache: 'global' }}
          }};
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
          body {{
            margin: 0;
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            color: #222;
            font-size: {font_size}px;
            line-height: 1.68;
            background: white;
          }}
          .box {{
            white-space: normal;
            word-break: break-word;
          }}
        </style>
      </head>
      <body>
        <div class="box">{safe}</div>
      </body>
    </html>
    """


def render_math_text(text: str, font_size: int = 18, min_height: int = 90) -> None:
    line_count = max(3, str(text).count("\n") + 1)
    height = max(min_height, 28 * line_count + 30)
    components.html(math_html(text, font_size=font_size), height=height, scrolling=False)


def render_section_card(title: str, body: str, font_size: int = 18, min_height: int = 110) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:12px 14px 8px 14px;
            margin:10px 0 6px 0;
            background:#ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        ">
            <div style="font-weight:700;font-size:17px;color:#111827;margin-bottom:6px;">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_math_text(body, font_size=font_size, min_height=min_height)


def context_text(row: pd.Series) -> str:
    benchmark = str(row["benchmark"])
    if benchmark == "Planning_Target":
        goal = str(row.get("gt_signal", "")).strip()
        if goal:
            return goal
    return ""


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


def render_guidance(row: pd.Series) -> None:
    benchmark = str(row["benchmark"])
    info = guidance_for(benchmark)
    st.subheader(pretty_dimension(benchmark))
    st.markdown(f"**{info['title']}**")
    st.write(info["goal"])
    st.markdown("**推荐依据**")
    for item in info["basis"]:
        st.markdown(f"- {item}")
    extra_context = context_text(row)
    if extra_context:
        hint = VISIBLE_CONTEXT_HINT.get(benchmark, "补充信息")
        st.markdown(f"**{hint}**")
        render_math_text(extra_context, font_size=18, min_height=80)


def render_history(row: pd.Series) -> None:
    st.markdown("**历史信息**")
    history_items = load_json_items(row.get("history_items_json", ""))
    if history_items:
        for item in history_items:
            title = f"历史 {item.get('index', '')}  |  得分 {item.get('score', '')}/{item.get('total', '')}"
            render_section_card(title, str(item.get("question_text", "")), font_size=17, min_height=120)
        return
    for idx, item in enumerate(build_history_items(row["history_preview"]), start=1):
        render_section_card(f"历史 {idx}", item, font_size=17, min_height=120)


def render_candidates(row: pd.Series) -> list[dict]:
    raw_candidates = load_json_items(row.get("candidate_items_json", ""))
    if raw_candidates:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        candidates = [
            {
                "label": letters[idx] if idx < len(letters) else f"Option-{idx + 1}",
                "text": str(item.get("question_text", "")),
                "qid": str(item.get("qid", "")),
                "is_gt": bool(item.get("is_gt", False)),
            }
            for idx, item in enumerate(raw_candidates)
        ]
    else:
        candidates = build_blind_candidates(row["candidate_preview"])
    st.markdown("**候选项**")
    for candidate in candidates:
        render_section_card(f"候选 {candidate['label']}", candidate["text"], font_size=18, min_height=140)
    return candidates


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
        index=choice_labels.index(default["teacher_choice_label"]) if default["teacher_choice_label"] in choice_labels else 0,
        format_func=lambda x: "请选择" if x == "" else x,
    )
    teacher_confidence = st.slider("推荐把握度", 1, 5, int(default["teacher_confidence"]))
    teacher_reason = st.text_area(
        "推荐理由",
        value=default["teacher_reason"],
        height=140,
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


def main() -> None:
    st.set_page_config(page_title="老师推荐标注", layout="wide")
    st.title("老师推荐标注")
    st.write("请根据页面给出的任务说明、历史信息和候选项，像真实教学推荐一样做选择。页面不会展示模型结果。")

    df = load_samples()
    init_state(df)
    filtered = get_filtered_df(df)

    if filtered.empty:
        st.warning("当前筛选下没有样本。")
        return

    render_progress(filtered)

    current_index = st.session_state.current_index
    if current_index >= len(filtered):
        st.session_state.current_index = 0
        current_index = 0

    row_ids = filtered["row_id"].tolist()
    selected_row_id = st.sidebar.selectbox(
        "跳转到样本",
        row_ids,
        index=min(current_index, len(row_ids) - 1),
        format_func=lambda rid: f"{rid} | {pretty_dimension(filtered[filtered['row_id'] == rid].iloc[0]['benchmark'])}",
    )
    st.session_state.current_index = int(filtered.index[filtered["row_id"] == selected_row_id][0])
    row = filtered.iloc[st.session_state.current_index]

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    if nav1.button("上一条", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    if nav2.button("下一条", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
    nav3.progress((st.session_state.current_index + 1) / len(filtered))

    st.caption(f"样本 {st.session_state.current_index + 1} / {len(filtered)}")

    left, right = st.columns([1.45, 1])
    with left:
        render_guidance(row)
        st.divider()
        render_history(row)
        st.divider()
        candidates = render_candidates(row)
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
