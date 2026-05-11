# Consitent

Teacher-facing Streamlit app for checking whether benchmark GT labels are reasonable.

## What is inside

- `app.py`: annotation UI for teachers
- `data/review_samples.csv`: the selected review set
- `data/review_samples.xlsx`: the same data in Excel form
- `data/annotations/`: optional local save location for exported labels

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on GitHub / Streamlit Community Cloud

1. Push the `Consitent/` folder to your repo.
2. Set the app entrypoint to `Consitent/app.py`.
3. Teachers can annotate in the browser and download their own CSV.

## Notes

- Browser session annotations are kept in `st.session_state`.
- Clicking `Save annotations to local file` writes a CSV under `data/annotations/` when the runtime filesystem is writable.
- On some hosted platforms, disk writes may be temporary, so the safest workflow is still: annotate, then click `Download annotations CSV`.
