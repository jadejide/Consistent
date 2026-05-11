from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
REVIEW_CSV = DATA_DIR / "review_samples.csv"

ANNOTATION_FIELDS = [
    "reviewer_name",
    "gt_is_reasonable",
    "teacher_preferred_case",
    "teacher_confidence",
    "teacher_would_recommend_gt",
    "teacher_comment",
]


def load_samples() -> pd.DataFrame:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"Missing review sample file: {REVIEW_CSV}")
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


def get_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter")
    dims = ["All"] + sorted(df["维度"].dropna().unique().tolist())
    cases = ["All"] + sorted(df["case_type"].dropna().unique().tolist())
    dim_choice = st.sidebar.selectbox("维度", dims)
    case_choice = st.sidebar.selectbox("样本类型", cases)

    out = df
    if dim_choice != "All":
        out = out[out["维度"] == dim_choice]
    if case_choice != "All":
        out = out[out["case_type"] == case_choice]
    return out.reset_index(drop=True)


def get_annotation(row_id: int) -> dict:
    return st.session_state.annotations.get(
        row_id,
        {
            "reviewer_name": st.session_state.reviewer_name,
            "gt_is_reasonable": "未判断",
            "teacher_preferred_case": "",
            "teacher_confidence": 3,
            "teacher_would_recommend_gt": "未判断",
            "teacher_comment": "",
        },
    )


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


def render_sample(row: pd.Series, current: int, total: int) -> None:
    st.caption(f"Sample {current}/{total}")
    c1, c2, c3 = st.columns(3)
    c1.metric("维度", str(row["维度"]))
    c2.metric("类型", str(row["case_type"]))
    c3.metric("最佳模型", str(row["best_model"]))

    with st.expander("Basic info", expanded=True):
        st.write(
            {
                "benchmark": row["benchmark"],
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "source": row["source"],
                "setting": row["setting"],
                "gt_signal": row["gt_signal"],
                "teacher_check_focus": row["teacher_check_focus"],
            }
        )

    with st.expander("History preview", expanded=True):
        st.text(str(row["history_preview"]))

    with st.expander("GT candidate", expanded=True):
        st.write(str(row["gt_candidate_text"]))

    with st.expander("All candidates", expanded=False):
        st.text(str(row["candidate_preview"]))

    with st.expander("Model predictions", expanded=False):
        model_cols = [col for col in row.index if col.endswith("_top3")]
        model_cols.sort()
        for col in model_cols:
            model_name = col[:-5]
            st.markdown(f"**{model_name}**")
            st.text(str(row[col]))


def render_annotation_form(row: pd.Series) -> None:
    row_id = int(row["row_id"])
    default = get_annotation(row_id)

    st.subheader("Teacher annotation")
    reviewer_name = st.text_input("Reviewer name", value=default["reviewer_name"] or st.session_state.reviewer_name)
    st.session_state.reviewer_name = reviewer_name

    gt_is_reasonable = st.radio(
        "Is the current GT reasonable?",
        ["未判断", "合理", "不太合理", "明显不合理"],
        index=["未判断", "合理", "不太合理", "明显不合理"].index(default["gt_is_reasonable"]),
        horizontal=True,
    )
    teacher_would_recommend_gt = st.radio(
        "Would you recommend the GT candidate to the target case?",
        ["未判断", "会推荐", "不会推荐", "不确定"],
        index=["未判断", "会推荐", "不会推荐", "不确定"].index(default["teacher_would_recommend_gt"]),
        horizontal=True,
    )
    teacher_confidence = st.slider("Confidence", 1, 5, int(default["teacher_confidence"]))
    teacher_preferred_case = st.text_area(
        "If not GT, what would you recommend instead?",
        value=default["teacher_preferred_case"],
        height=100,
    )
    teacher_comment = st.text_area(
        "Comment",
        value=default["teacher_comment"],
        height=140,
    )

    payload = {
        "reviewer_name": reviewer_name,
        "gt_is_reasonable": gt_is_reasonable,
        "teacher_preferred_case": teacher_preferred_case,
        "teacher_confidence": teacher_confidence,
        "teacher_would_recommend_gt": teacher_would_recommend_gt,
        "teacher_comment": teacher_comment,
    }
    set_annotation(row_id, payload)


def main() -> None:
    st.set_page_config(page_title="GT Consistency Review", layout="wide")
    st.title("GT Consistency Review")
    st.write("Teachers can review whether the benchmark GT is reasonable for each selected sample.")

    df = load_samples()
    init_state(df)
    filtered = get_filtered_df(df)

    if filtered.empty:
        st.warning("No samples match the current filters.")
        return

    index = st.session_state.current_index
    if index >= len(filtered):
        st.session_state.current_index = 0
        index = 0

    selected_ids = filtered["row_id"].tolist()
    selected_row_id = st.sidebar.selectbox(
        "Jump to sample",
        selected_ids,
        index=min(index, len(selected_ids) - 1),
        format_func=lambda rid: f"{rid} | {filtered[filtered['row_id'] == rid].iloc[0]['维度']} | {filtered[filtered['row_id'] == rid].iloc[0]['case_type']}",
    )
    selected_index = filtered.index[filtered["row_id"] == selected_row_id][0]
    st.session_state.current_index = int(selected_index)
    row = filtered.iloc[selected_index]

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    if nav1.button("Previous", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    if nav2.button("Next", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
    nav3.progress((st.session_state.current_index + 1) / len(filtered))

    left, right = st.columns([1.35, 1])
    with left:
        render_sample(row, st.session_state.current_index + 1, len(filtered))
    with right:
        render_annotation_form(row)

    st.divider()
    export_df = build_export_df(df)
    st.subheader("Export")
    st.write(f"Annotated samples in this session: {len(export_df)}")

    download_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download annotations CSV",
        data=download_bytes,
        file_name="gt_review_annotations.csv",
        mime="text/csv",
    )

    if st.button("Save annotations to local file"):
        path = save_local_annotations(export_df, st.session_state.reviewer_name)
        st.success(f"Saved to {path}")

    with st.expander("Preview current annotations", expanded=False):
        st.dataframe(export_df, use_container_width=True)
        st.code(json.dumps(st.session_state.annotations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
