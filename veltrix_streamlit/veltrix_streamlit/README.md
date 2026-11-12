# Veltrix segmentation: Streamlit app

Interactive version of the Veltrix customer & buyer segmentation project (synthetic data, Jan 2023 to Sep 2026).

## Pages
Overview | Segment explorer | Business questions (Q1 to Q8) | Seasonality (stamp duty spike, August and December) | Demand explorer | Customers (lookup, filter/export, score new customers) | Quality & validation

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
`app_data/` (2 MB) is included, so the app runs without the 19M-row source tables.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (keep `app.py`, `requirements.txt`, `.streamlit/`, `app_data/` at the repo root).
2. At share.streamlit.io choose **New app**, pick the repo, branch and `app.py`, then **Deploy**.
3. Use Python 3.11 or 3.12 in Advanced settings.

Other options: `docker`/any host that can run `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Refreshing the data
After re-running `Veltrix_segmentation.ipynb` (writes `outputs/`), rebuild the app data:
```bash
python prepare_app_data.py --data /path/to/parquet --outputs /path/to/outputs
```

## Notes
- The scoring page loads `app_data/model.pkl`, which was trained with `kmodes==0.12.2` and `scikit-learn==1.8.0`; keep those versions pinned. Uploaded CSVs must follow `app_data/scoring_template.csv` (engineered features, not raw tables). The text value `None` is a genuine category in `lead_source_primary` and `pref_property_type`.
- Segment names are rule-based suggestions. Bootstrap stability is 75.7% (target 80%) and the real ARI against hidden personas has not been measured; the app says so on the Quality page.
- The data is synthetic. No authentication is included: add it (e.g. Streamlit Community Cloud viewer restrictions) before loading real customer data.
