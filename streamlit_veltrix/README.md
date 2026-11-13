# Veltrix Streamlit Dashboard

## Folder layout
Place this folder (or copy its files) in your project root, next to `data/` and `Results/`:

```
veltrix-segmentation_F/
  app.py
  requirements.txt
  .streamlit/config.toml
  data/raw/veltrix_medium/*.parquet
  Results/models/kmeans_k6_assignments.csv        (for the Segmentation tab)
  Results/reports/clustering_experiment_results.csv
```

## Launch from the VS Code terminal (Terminal > New Terminal)

Windows PowerShell:
```powershell
cd path\to\veltrix-segmentation_F
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with Ctrl+C.

## Using a different data folder
Type the path in the sidebar box, or set it before launching:
```powershell
$env:VELTRIX_DATA_DIR = "C:\path\to\data\raw\veltrix_medium"; streamlit run app.py
```

## Notes
- Revenue growth is calculated as the last 12 months vs the prior 12 months of sale revenue. Change this in `app.py` if your Power BI measure is defined differently.
- "High Intent Buyers" uses lead stages in `HIGH_INTENT_STAGES` (top of `app.py`).
- Lead-source and country labels are standardised in the app, so Rightmove spellings are combined.
- Your install showed an invalid `~treamlit` package warning. If Streamlit misbehaves, use the virtual environment above, which avoids it.
