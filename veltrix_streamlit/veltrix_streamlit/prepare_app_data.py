"""Build the small `app_data/` folder the Streamlit app reads.

Run ONCE on a machine that has (a) the Veltrix parquet tables and (b) the `outputs/` folder written by
Veltrix_segmentation.ipynb. The deployed app needs neither the 19M-row source tables nor the notebook.

    python prepare_app_data.py --data /path/to/parquet --outputs /path/to/outputs
"""
import argparse, json, pickle
from pathlib import Path
import duckdb, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/mnt/user-data/uploads")
ap.add_argument("--outputs", default="/home/claude/work/nbrun/outputs")
ap.add_argument("--dest", default="app_data")
a = ap.parse_args()
DATA, OUT, DEST = Path(a.data), Path(a.outputs), Path(a.dest); DEST.mkdir(exist_ok=True)

# friendly names for stakeholders (rule-based names from the notebook -> plain-English labels)
FRIENDLY = {"Home-mover family": "Mortgage buyers", "Window shopper": "Dormant buyers", "Corporate relocation": "Active renters",
            "Buy-to-let investor": "Buy-to-let investors", "Young renter": "Dormant renters", "Downsizer": "Downsizers",
            "Overseas investor": "Overseas investors"}
fr = lambda s: s.map(lambda x: FRIENDLY.get(x, x))

# ---- segments (persona cards + routing rules)
cards = pd.read_csv(OUT/"persona_cards.csv"); rules = pd.read_csv(OUT/"routing_rules.csv")
seg = cards.merge(rules[["persona","priority_tier","desk","route_to","first_reply_sla"]], on="persona")
seg["segment"] = fr(seg.persona); seg.to_csv(DEST/"segments.csv", index=False)

# ---- customers (one row per original record) + features for lookup
crm = pd.read_csv(OUT/"crm_segment_writeback.csv"); crm["segment"] = fr(crm.segment_name)
feat = pd.read_parquet(OUT/"customer_features.parquet")[["master_id","customer_type","finance_type","primary_intent","region_group","country_group","age_band","lead_source_primary","pref_property_type","budget_stated","has_lead"]]
cust = crm.merge(feat, on="master_id", how="left")[["customer_id","master_id","segment","priority_tier","desk","route_to","first_reply_sla",
        "customer_type","finance_type","primary_intent","region_group","country_group","age_band","lead_source_primary","pref_property_type","budget_stated","has_lead"]]
for c in cust.select_dtypes(exclude="number").columns: cust[c] = cust[c].astype("category")
cust.to_parquet(DEST/"customers.parquet", index=False)

# ---- slim model for the scoring page
art = pickle.load(open(OUT/"segmentation_artifacts.pkl","rb"))
names = dict(zip(cards.segment_id, fr(cards.persona)))
pickle.dump({k: art[k] for k in ["scaler","model","NUM_COLS","CAT_COLS","K","GAMMA"]} | {"names": names}, open(DEST/"model.pkl","wb"))
feat_full = pd.read_parquet(OUT/"customer_features.parquet").drop(columns="master_id")
feat_full.head(25).to_csv(DEST/"scoring_template.csv", index=False)

# ---- validation
art["stab"].to_csv(DEST/"stability.csv", index=False); art["kcurve"].to_csv(DEST/"k_curve.csv", index=False)
art["gamma_tbl"].to_csv(DEST/"gamma_grid.csv", index=False)
json.dump({"K": int(art["K"]), "gamma": float(art["GAMMA"]), "stability": float(art["stab"].matched_agreement.mean()),
           "stability_min": float(art["stab"].matched_agreement.min()), "stability_max": float(art["stab"].matched_agreement.max()),
           "bootstrap_ari": float(art["stab"].ARI.mean()), "kmeans_proxy_ari": float(art["kmeans_proxy_ari"])}, open(DEST/"validation.json","w"))
pd.read_csv(OUT/"data_quality_issue_log.csv").to_csv(DEST/"data_quality_issue_log.csv", index=False)

# ---- demand tables
xl = pd.ExcelFile(OUT/"segment_demand_report.xlsx")
for sh in xl.sheet_names:
    d = xl.parse(sh); d.iloc[:, 0] = fr(d.iloc[:, 0]); d.to_csv(DEST/f"demand_{sh}.csv", index=False)
pd.read_csv(OUT/"property_demand_by_region_type.csv").to_csv(DEST/"property_views.csv", index=False)
aud = pd.read_csv(OUT/"powerbi"/"segment_channel_response.csv"); aud["persona"] = fr(aud.persona); aud.to_csv(DEST/"channel_response.csv", index=False)

# ---- time series and lead-level facts straight from the source tables
con = duckdb.connect(); P = lambda t: (DATA/f"{t}.parquet").as_posix()
con.execute(f"""COPY (SELECT strftime(date_trunc('month', created_date),'%Y-%m-01') AS month_start, intent, count(*) n FROM read_parquet('{P('leads')}') GROUP BY 1,2 ORDER BY 1)
                TO '{(DEST/'monthly_leads.csv').as_posix()}' (HEADER)""")
con.execute(f"""COPY (SELECT strftime(date_trunc('month', completion_or_move_in_date),'%Y-%m-01') AS month_start,
                         CASE WHEN transaction_type LIKE '%purchase%' THEN 'Sale completions' ELSE 'Let move-ins' END kind, count(*) n
                  FROM read_parquet('{P('transactions')}') WHERE completion_or_move_in_date IS NOT NULL GROUP BY 1,2 ORDER BY 1)
                TO '{(DEST/'monthly_completions.csv').as_posix()}' (HEADER)""")
SRC = """case lower(trim(lead_source)) when 'rightmove' then 'Rightmove' when 'right move' then 'Rightmove' when 'zoopla' then 'Zoopla' when 'zoopla.co.uk' then 'Zoopla'
 when 'onthemarket' then 'OnTheMarket' when 'on the market' then 'OnTheMarket' when 'otm' then 'OnTheMarket'
 when 'veltrix website' then 'Veltrix website' when 'veltrix.co.uk' then 'Veltrix website' when 'website' then 'Veltrix website'
 when 'google ads' then 'Google Ads' when 'google' then 'Google Ads' when 'facebook' then 'Facebook' when 'fb' then 'Facebook'
 when 'instagram' then 'Instagram' when 'insta' then 'Instagram' when 'tiktok' then 'TikTok' when 'wechat' then 'WeChat'
 when 'email newsletter' then 'Email newsletter' when 'referral' then 'Referral' when 'branch walk-in' then 'Branch walk-in' when 'walk-in' then 'Branch walk-in'
 when 'overseas agent partner' then 'Overseas agent partner' when 'relocation partner' then 'Relocation partner' end"""
con.execute(f"""COPY (SELECT {SRC} source, intent, count(*) closed_leads, sum(converted::int) won FROM read_parquet('{P('leads')}')
                  WHERE lead_stage IN ('Won','Lost') GROUP BY 1,2) TO '{(DEST/'source_conversion.csv').as_posix()}' (HEADER)""")
fr_ = con.execute(f"""WITH f AS (SELECT i.lead_id, arg_min(i.response_time_min, i.interaction_time) mins, any_value(a.seniority) seniority
        FROM read_parquet('{P('interactions')}') i JOIN read_parquet('{P('agents')}') a USING(agent_id)
        WHERE i.direction='Inbound' AND i.response_time_min IS NOT NULL GROUP BY 1)
      SELECT f.mins, f.seniority, l.converted::int converted FROM f JOIN read_parquet('{P('leads')}') l USING(lead_id) WHERE l.lead_stage IN ('Won','Lost')""").df()
fr_["band"] = pd.cut(fr_.mins, [-1, 60, 240, 1440, 1e12], labels=["Within 1 hour","1 to 4 hours","4 to 24 hours","Over 24 hours"])
fr_.groupby("band", observed=True).converted.agg(conversion="mean", leads="size").reset_index().to_csv(DEST/"reply_bands.csv", index=False)
fr_.groupby("seniority").agg(conversion=("converted","mean"), median_first_reply_min=("mins","median"), leads=("mins","size")).reset_index().to_csv(DEST/"reply_seniority.csv", index=False)
print("app_data written:", sorted(p.name for p in DEST.iterdir()))
