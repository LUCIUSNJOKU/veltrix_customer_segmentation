"""Veltrix | Customer & Lead Analysis - Streamlit dashboard.

Run from the VS Code terminal (project folder):
    streamlit run app.py
Data is read from data/raw/veltrix_medium (Parquet) through DuckDB.
Override the location with the sidebar box or the VELTRIX_DATA_DIR env variable.
"""
import itertools
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

TEAL, TEAL_MID, TEAL_LIGHT, GREY = "#006D6A", "#2A9AA3", "#7FC4C2", "#6B7780"
HIGH_INTENT_STAGES = ("Offer", "Viewing")          # edit to change the 'High Intent Buyers' definition
TABLES = ["agents", "campaigns", "campaign_responses", "customers", "interactions",
          "leads", "locations", "payments", "properties", "site_visits", "transactions"]

LEAD_SOURCE_MAP = {
    "rightmove": "Rightmove", "right move": "Rightmove", "rightmove.co.uk": "Rightmove",
    "zoopla": "Zoopla", "zoopla.co.uk": "Zoopla",
    "onthemarket": "OnTheMarket", "on the market": "OnTheMarket", "otm": "OnTheMarket",
    "veltrix website": "Veltrix website", "veltrix.co.uk": "Veltrix website", "website": "Veltrix website",
    "google ads": "Google Ads", "google": "Google Ads",
    "facebook": "Facebook", "fb": "Facebook", "instagram": "Instagram", "insta": "Instagram",
    "tiktok": "TikTok", "wechat": "WeChat", "referral": "Referral", "email newsletter": "Email newsletter",
    "branch walk-in": "Walk-in", "walk-in": "Walk-in",
    "overseas agent partner": "Overseas agent partner", "relocation partner": "Relocation partner",
}
COUNTRY_MAP = {
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "england": "United Kingdom",
    "great britain": "United Kingdom", "u.k.": "United Kingdom", "gb": "United Kingdom",
    "hong kong": "Hong Kong SAR", "hong kong sar": "Hong Kong SAR", "hk": "Hong Kong SAR",
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates", "u.a.e.": "United Arab Emirates",
    "united states": "United States", "usa": "United States", "us": "United States",
    "china": "China", "mainland china": "China", "prc": "China",
    "singapore": "Singapore", "sg": "Singapore", "saudi arabia": "Saudi Arabia", "ksa": "Saudi Arabia",
    "qatar": "Qatar", "india": "India", "nigeria": "Nigeria", "canada": "Canada", "malaysia": "Malaysia",
}
PERSONAS = pd.DataFrame({
    "Persona": ["Highly Engaged Property Explorers", "High-Intent Buyers", "Slow-Progress Leads", "Early-Stage Browsers",
                "Overseas Property Buyers", "Rental-Ready Customers", "Repeat Buyers", "Investment-Focused Customers"],
    "Customers": [61351, 53597, 46045, 39119, 9596, 5859, 1535, 0]})
PERSONAS["Share %"] = (PERSONAS["Customers"] / 102000 * 100).round(2)

st.set_page_config(page_title="Veltrix | Customer & Lead Analysis", layout="wide", page_icon="🏠")
st.markdown(f"""<style>
.block-container {{padding-top: 1.2rem;}}
.banner {{background:{TEAL}; color:white; padding:0.9rem 1.4rem; border-radius:18px; margin-bottom:1rem;}}
.banner h1 {{color:white; font-size:1.7rem; margin:0; padding:0;}}
div[data-testid="stMetric"] {{background:#E6F2F1; border-radius:12px; padding:0.6rem 0.9rem;}}
div[data-testid="stMetricValue"] {{color:{TEAL};}}
</style>""", unsafe_allow_html=True)


_chart_ids = itertools.count()


# ----------------------------------------------------------------- data layer
def find_data_dir() -> str:
    env = os.environ.get("VELTRIX_DATA_DIR")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "data" / "raw" / "veltrix_medium"
        if cand.exists():
            return str(cand)
    return str(here / "data" / "raw" / "veltrix_medium")


@st.cache_resource(show_spinner="Connecting to data...")
def get_con(data_dir: str):
    base = Path(data_dir)
    con = duckdb.connect()
    for t in TABLES:
        f = base / f"{t}.parquet"
        if f.exists():
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    con.register("src_map", pd.DataFrame({"raw_key": LEAD_SOURCE_MAP.keys(), "source_std": LEAD_SOURCE_MAP.values()}))
    con.register("country_map", pd.DataFrame({"raw_key": COUNTRY_MAP.keys(), "country_std": COUNTRY_MAP.values()}))
    return con


@st.cache_data(show_spinner=False)
def run(sql: str, data_dir: str) -> pd.DataFrame:
    return get_con(data_dir).execute(sql).df()


def gbp(v, unit=""):
    return "n/a" if pd.isna(v) else f"£{v:,.0f}{unit}"


def big(v):
    for lim, s in ((1e9, "bn"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= lim:
            return f"{v / lim:,.2f}{s}"
    return f"{v:,.0f}"


def bar(df, x, y, title, horizontal=True, fmt=None, color=TEAL_MID, height=380):
    d = df.copy()
    fig = px.bar(d, x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v",
                 title=title, color_discrete_sequence=[color], text_auto=fmt or True)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None, xaxis_title=None)
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, key=f"chart_{next(_chart_ids)}")


def line(df, x, y, title, height=380):
    fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=[TEAL_MID])
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, key=f"chart_{next(_chart_ids)}")


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Veltrix")
data_dir = st.sidebar.text_input("Data folder (Parquet)", find_data_dir())
if not (Path(data_dir) / "leads.parquet").exists():
    st.error(f"Could not find leads.parquet in: {data_dir}\n\nSet the correct folder in the sidebar "
             "or set the VELTRIX_DATA_DIR environment variable.")
    st.stop()

rng = run("SELECT CAST(min(created_date) AS DATE) d0, CAST(max(created_date) AS DATE) d1 FROM leads", data_dir).iloc[0]
dates = st.sidebar.date_input("Lead / contract date range", (rng.d0, rng.d1), min_value=rng.d0, max_value=rng.d1)
d0, d1 = (dates if isinstance(dates, (list, tuple)) and len(dates) == 2 else (rng.d0, rng.d1))
LW = f"CAST(l.created_date AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}'"
TW = f"CAST(t.contract_date AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}'"
LEADS = "leads l LEFT JOIN src_map m ON lower(trim(l.lead_source)) = m.raw_key"
SRC = "coalesce(m.source_std, 'Other')"
st.sidebar.caption(f"Data to {rng.d1:%d %b %Y}")

st.markdown('<div class="banner"><h1>VELTRIX | CUSTOMER & LEAD ANALYSIS</h1></div>', unsafe_allow_html=True)
tabs = st.tabs(["Business Overview", "Marketing Performance", "Sales Performance", "Customer & Lead Analysis", "Segmentation & Personas"])

# ----------------------------------------------------------------- shared numbers
lead_k = run(f"SELECT count(*) n, sum(CAST(converted AS INT)) conv FROM {LEADS} WHERE {LW}", data_dir).iloc[0]
sale_k = run(f"""SELECT count(*) deals, sum(agreed_price_gbp) rev, avg(agreed_price_gbp) avg_deal,
                 sum(agreed_rent_pcm_gbp) rent FROM transactions t WHERE {TW}""", data_dir).iloc[0]
monthly = run(f"""SELECT date_trunc('month', CAST(contract_date AS DATE)) AS month, sum(agreed_price_gbp) AS revenue
                  FROM transactions t WHERE {TW} GROUP BY 1 ORDER BY 1""", data_dir)
conv_rate = lead_k.conv / lead_k.n if lead_k.n else 0
growth = None
if len(monthly) >= 24:
    last12, prev12 = monthly.revenue.tail(12).sum(), monthly.revenue.iloc[-24:-12].sum()
    growth = (last12 / prev12 - 1) if prev12 else None
by_branch = run(f"""SELECT a.branch, sum(t.agreed_price_gbp) AS revenue FROM transactions t
                    JOIN leads l ON t.lead_id = l.lead_id JOIN agents a ON l.agent_id = a.agent_id
                    WHERE {TW} AND t.agreed_price_gbp IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""", data_dir)
by_type = run(f"""SELECT transaction_type, count(*) AS deals, coalesce(sum(agreed_price_gbp), 0) AS revenue
                  FROM transactions t WHERE {TW} GROUP BY 1 ORDER BY 2 DESC""", data_dir)
conv_src = run(f"""SELECT {SRC} AS source, count(*) AS leads, avg(CAST(converted AS INT)) * 100 AS conversion_pct
                   FROM {LEADS} WHERE {LW} GROUP BY 1 ORDER BY 3 DESC""", data_dir)

# ----------------------------------------------------------------- 1. business overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Total leads", big(lead_k.n))
    c[1].metric("Lead conversion rate", f"{conv_rate:.1%}")
    c[2].metric("Average deal", gbp(sale_k.avg_deal))
    c[3].metric("Revenue growth (last 12m vs prior 12m)", "n/a" if growth is None else f"{growth:.2%}")
    c[4].metric("Total deals closed", big(sale_k.deals))
    c[5].metric("Sum of agreed price", "£" + big(sale_k.rev or 0))
    a, b = st.columns(2)
    with a:
        line(monthly, "month", "revenue", "Monthly sales revenue trend")
    with b:
        bar(conv_src, "source", "conversion_pct", "Lead conversion by marketing channel (%)", fmt=".1f")
    a, b, c3 = st.columns(3)
    with a:
        bar(by_branch.head(15), "branch", "revenue", "Sales revenue by branch (top 15)", fmt=".2s")
    with b:
        bar(by_type, "transaction_type", "deals", "Deals closed by transaction type", horizontal=False)
    with c3:
        bar(by_type[by_type.revenue > 0], "transaction_type", "revenue", "Sales revenue by transaction type", horizontal=False, fmt=".2s")

# ----------------------------------------------------------------- 2. marketing
with tabs[1]:
    resp = run(f"""SELECT count(*) n FROM campaign_responses r
                   WHERE CAST(r.response_time AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}'""", data_dir).iloc[0].n
    n_camp = run("SELECT count(*) n FROM campaigns", data_dir).iloc[0].n
    c = st.columns(4)
    c[0].metric("Total leads", big(lead_k.n))
    c[1].metric("Lead conversion rate", f"{conv_rate:.1%}")
    c[2].metric("Total campaign responses", big(resp))
    c[3].metric("Total campaigns", f"{n_camp:,}")
    trend = run(f"""SELECT date_trunc('month', CAST(response_time AS DATE)) AS month, count(*) AS responses
                    FROM campaign_responses WHERE CAST(response_time AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}'
                    GROUP BY 1 ORDER BY 1""", data_dir)
    by_ch = run(f"""SELECT c.channel, count(*) AS responses FROM campaign_responses r JOIN campaigns c USING (campaign_id)
                    WHERE CAST(r.response_time AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}' GROUP BY 1 ORDER BY 2 DESC""", data_dir)
    leads_src = run(f"SELECT {SRC} AS source, count(*) AS leads FROM {LEADS} WHERE {LW} GROUP BY 1 ORDER BY 2 DESC", data_dir)
    theme = run("""SELECT c.theme, sum(c.spend_gbp) AS spend,
                   (SELECT count(*) FROM campaign_responses r JOIN campaigns c2 USING (campaign_id) WHERE c2.theme = c.theme) AS responses
                   FROM campaigns c GROUP BY c.theme""", data_dir)
    theme["responses_per_gbp"] = theme.responses / theme.spend
    a, b, c3 = st.columns(3)
    with a:
        line(trend, "month", "responses", "Monthly campaign response trend")
    with b:
        bar(by_ch, "channel", "responses", "Campaign responses by channel", fmt=".2s")
    with c3:
        bar(leads_src, "source", "leads", "Total leads by marketing channel", fmt=".2s")
    a, b, c3 = st.columns(3)
    with a:
        bar(theme.sort_values("spend", ascending=False), "theme", "spend", "Campaign spend by theme (£)", fmt=".2s")
    with b:
        bar(conv_src, "source", "conversion_pct", "Conversion rate by lead source (%)", fmt=".1f")
    with c3:
        bar(theme.sort_values("responses_per_gbp", ascending=False), "theme", "responses_per_gbp", "Responses per £ spent by theme", fmt=".2f")

# ----------------------------------------------------------------- 3. sales
with tabs[2]:
    c = st.columns(4)
    c[0].metric("Total deals", big(sale_k.deals))
    c[1].metric("Total sales revenue", "£" + big(sale_k.rev or 0))
    c[2].metric("Average deal value", gbp(sale_k.avg_deal))
    c[3].metric("Total rental income (sum of monthly rents)", "£" + big(sale_k.rent or 0))
    ptype = run(f"""SELECT p.property_type, count(*) AS deals FROM transactions t JOIN properties p USING (property_id)
                    WHERE {TW} GROUP BY 1 ORDER BY 2 DESC""", data_dir)
    agents = run(f"""SELECT l.agent_id, count(*) AS deals, avg(t.agreed_price_gbp) AS avg_deal_value
                     FROM transactions t JOIN leads l ON t.lead_id = l.lead_id WHERE {TW} GROUP BY 1""", data_dir)
    a, b = st.columns(2)
    with a:
        bar(by_branch.head(15), "branch", "revenue", "Sales revenue by branch", fmt=".2s")
    with b:
        bar(by_type, "transaction_type", "deals", "Deals closed by transaction type", horizontal=False)
    a, b = st.columns(2)
    with a:
        bar(agents.nlargest(10, "deals"), "agent_id", "deals", "Top 10 agents by deals closed")
    with b:
        bar(agents[agents.deals >= 5].nlargest(10, "avg_deal_value"), "agent_id", "avg_deal_value",
            "Top 10 agents by average deal value (5+ deals)", fmt=".2s")
    a, b = st.columns(2)
    with a:
        bar(ptype, "property_type", "deals", "Total deals by property type")
    with b:
        bar(by_type[by_type.revenue > 0], "transaction_type", "revenue", "Sales revenue by transaction type", horizontal=False, fmt=".2s")

# ----------------------------------------------------------------- 4. customer & lead analysis
with tabs[3]:
    opts = lambda col: ["All"] + sorted(run(f"SELECT DISTINCT {col} v FROM customers WHERE {col} IS NOT NULL", data_dir).v.astype(str))
    f1, f2, f3 = st.columns(3)
    age = f1.selectbox("Age", opts("age_band"))
    region = f2.selectbox("UK region", opts("uk_region_of_residence"))
    countries = ["All"] + sorted(set(COUNTRY_MAP.values()))
    country = f3.selectbox("Country", countries)
    cw = ["1=1"]
    if age != "All":
        cw.append(f"c.age_band = '{age}'")
    if region != "All":
        cw.append(f"c.uk_region_of_residence = '{region}'")
    if country != "All":
        cw.append(f"coalesce(cm.country_std, c.country_of_residence) = '{country}'")
    CW = " AND ".join(cw)
    base = f"""FROM customers c LEFT JOIN country_map cm ON lower(trim(c.country_of_residence)) = cm.raw_key
               JOIN leads l ON l.customer_id = c.customer_id WHERE {LW} AND {CW}"""
    stages = "', '".join(HIGH_INTENT_STAGES)
    k = run(f"""SELECT count(DISTINCT c.customer_id) customers, sum(CAST(l.converted AS INT)) converted, count(*) leads,
                count(DISTINCT CASE WHEN l.lead_stage IN ('{stages}') THEN c.customer_id END) high_intent {base}""", data_dir).iloc[0]
    resp1h = run(f"""SELECT avg(CASE WHEN i.response_time_min <= 60 THEN 1.0 ELSE 0 END) p
                     FROM interactions i JOIN leads l ON i.lead_id = l.lead_id JOIN customers c ON l.customer_id = c.customer_id
                     LEFT JOIN country_map cm ON lower(trim(c.country_of_residence)) = cm.raw_key
                     WHERE i.response_time_min IS NOT NULL AND {LW} AND {CW}""", data_dir).iloc[0].p
    resp1h = 0 if pd.isna(resp1h) else resp1h
    c = st.columns(5)
    c[0].metric("Total customers", f"{int(k.customers or 0):,}")
    c[1].metric("Converted leads", f"{int(k.converted or 0):,}")
    c[2].metric("Lead conversion rate", f"{(k.converted or 0) / k.leads:.1%}" if k.leads else "n/a")
    c[3].metric("High intent buyers", f"{int(k.high_intent or 0):,}")
    c[4].metric("Response rate gap", f"{1 - resp1h:.2%}", help="Share of inbound enquiries NOT answered within one hour")
    seg = run(f"SELECT coalesce(c.buyer_position, 'Unknown') AS segment, count(*) AS leads, avg(CAST(l.converted AS INT)) * 100 AS conversion_pct {base} GROUP BY 1 ORDER BY 3 DESC", data_dir)
    deals_pt = run(f"""SELECT p.property_type, count(DISTINCT t.transaction_id) AS deals FROM customers c
                       LEFT JOIN country_map cm ON lower(trim(c.country_of_residence)) = cm.raw_key
                       JOIN transactions t ON t.customer_id = c.customer_id JOIN properties p USING (property_id)
                       WHERE CAST(t.contract_date AS DATE) BETWEEN DATE '{d0}' AND DATE '{d1}' AND {CW} GROUP BY 1 ORDER BY 2 DESC""", data_dir)
    stage = run(f"SELECT l.lead_stage, count(*) AS leads {base} GROUP BY 1 ORDER BY 2 DESC", data_dir)
    lost = run(f"SELECT l.lost_reason, count(*) AS leads {base} AND l.lost_reason IS NOT NULL GROUP BY 1 ORDER BY 2 DESC", data_dir)
    agent_rt = run(f"""SELECT i.agent_id, avg(i.response_time_min) AS avg_minutes FROM interactions i
                       JOIN leads l ON i.lead_id = l.lead_id JOIN customers c ON l.customer_id = c.customer_id
                       LEFT JOIN country_map cm ON lower(trim(c.country_of_residence)) = cm.raw_key
                       WHERE i.response_time_min IS NOT NULL AND {LW} AND {CW} GROUP BY 1 ORDER BY 2 DESC LIMIT 10""", data_dir)
    a, b, c3 = st.columns(3)
    with a:
        bar(seg, "segment", "conversion_pct", "Lead conversion by buyer position (%)", fmt=".1f")
    with b:
        bar(deals_pt, "property_type", "deals", "Customers with completed deals by property type")
    with c3:
        bar(stage, "lead_stage", "leads", "Leads by lead stage")
    a, b, c3 = st.columns(3)
    with a:
        bar(lost, "lost_reason", "leads", "Lost leads by reason")
    with b:
        st.metric("Response within 1 hour", f"{resp1h:.2%}")
        st.progress(min(max(float(resp1h), 0.0), 1.0))
    with c3:
        bar(agent_rt, "agent_id", "avg_minutes", "Average agent response time (minutes, slowest 10)", fmt=".0f")

# ----------------------------------------------------------------- 5. segmentation
with tabs[4]:
    root = Path(data_dir).resolve().parents[2] if len(Path(data_dir).resolve().parents) > 2 else Path(data_dir)
    files = sorted((root / "Results" / "models").glob("kmeans_k*_assignments.csv"), key=lambda p: int(p.stem.split("_k")[1].split("_")[0])) if (root / "Results" / "models").exists() else []
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Behavioural personas")
        st.dataframe(PERSONAS, hide_index=True)
        bar(PERSONAS, "Persona", "Share %", "Share of customers by persona (%) - flags overlap", fmt=".1f")
    with b:
        res_f = root / "Results" / "reports" / "clustering_experiment_results.csv"
        if res_f.exists():
            res = pd.read_csv(res_f)
            fig = px.line(res, x="k", y="silhouette_score", color="algorithm", markers=True,
                          title="Silhouette score by k", color_discrete_sequence=[TEAL, GREY])
            fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, key="silhouette_chart")
        else:
            st.info("Clustering experiment results not found (Results/reports/clustering_experiment_results.csv).")
    st.subheader("Cluster profile")
    if not files:
        st.warning("No K-means assignment files found in Results/models. Run the segmentation notebook first.")
    else:
        labels = {int(p.stem.split("_k")[1].split("_")[0]): p for p in files}
        k_sel = st.selectbox("Number of clusters (k)", list(labels), index=list(labels).index(6) if 6 in labels else 0)
        asg = pd.read_csv(labels[k_sel])
        cust = run("""SELECT c.customer_id, c.stated_budget_gbp, CAST(c.is_overseas AS INT) AS overseas,
                      coalesce(l.n_leads, 0) AS n_leads, coalesce(l.n_conv, 0) AS n_conv, coalesce(t.rev, 0) AS revenue
                      FROM customers c
                      LEFT JOIN (SELECT customer_id, count(*) n_leads, sum(CAST(converted AS INT)) n_conv FROM leads GROUP BY 1) l USING (customer_id)
                      LEFT JOIN (SELECT customer_id, sum(agreed_price_gbp) rev FROM transactions GROUP BY 1) t USING (customer_id)""", data_dir)
        m = asg.merge(cust, on="customer_id", how="left")
        prof = m.groupby("cluster").agg(customers=("customer_id", "count"), median_budget=("stated_budget_gbp", "median"),
                                        overseas_share=("overseas", "mean"), leads=("n_leads", "sum"),
                                        converted=("n_conv", "sum"), revenue=("revenue", "sum")).reset_index()
        prof["share_%"] = (prof.customers / prof.customers.sum() * 100).round(1)
        prof["conversion_%"] = (prof.converted / prof.leads * 100).round(1)
        prof["overseas_share"] = (prof.overseas_share * 100).round(1)
        prof["cluster"] = "Segment " + (prof.cluster + 1).astype(str)
        st.dataframe(prof[["cluster", "customers", "share_%", "median_budget", "overseas_share", "leads", "conversion_%", "revenue"]],
                     hide_index=True)
        a, b = st.columns(2)
        with a:
            bar(prof, "cluster", "customers", "Customers per segment", horizontal=False, fmt=".2s")
        with b:
            bar(prof, "cluster", "conversion_%", "Lead conversion by segment (%)", horizontal=False, fmt=".1f")
