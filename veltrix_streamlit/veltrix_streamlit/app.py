"""Veltrix customer & buyer segmentation: Streamlit app.
Run locally:  streamlit run app.py
Reads only the small files in ./app_data (built by prepare_app_data.py)."""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

D = Path(__file__).parent / "app_data"
st.set_page_config(page_title="Veltrix segmentation", page_icon="🏠", layout="wide")
TEAL, DEEP, MINT, AMBER, CORAL, SLATE = "#0F8B8D", "#0B4F4F", "#7FC8BE", "#F2A541", "#E4572E", "#5E8C8A"
SEG_COLORS = {"Mortgage buyers": DEEP, "Dormant buyers": MINT, "Active renters": TEAL, "Buy-to-let investors": AMBER,
              "Dormant renters": "#B9DAD5", "Downsizers": SLATE, "Overseas investors": CORAL}
st.markdown("""<style>
h1,h2,h3{font-family:Georgia,serif;color:#0B4F4F}
div[data-testid="stMetric"]{background:#D5EAE6;padding:12px 16px;border-radius:8px}
section[data-testid="stSidebar"]{background:#0B4F4F} section[data-testid="stSidebar"] *{color:#F3F9F7 !important}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ data loaders
@st.cache_data
def csv(name): return pd.read_csv(D / name)
@st.cache_data
def customers(): return pd.read_parquet(D / "customers.parquet")
@st.cache_data
def validation(): return json.load(open(D / "validation.json"))
@st.cache_resource
def model_bundle():
    try:
        return pickle.load(open(D / "model.pkl", "rb")), None
    except Exception as e:  # kmodes / scikit-learn missing or version mismatch
        return None, str(e)

SEG = csv("segments.csv").sort_values("share", ascending=False).reset_index(drop=True)
SEG_ORDER = SEG.segment.tolist()
pct = lambda x, d=1: "n/a" if pd.isna(x) else f"{x*100:.{d}f}%"
gbp = lambda x: "n/a" if pd.isna(x) else f"£{x:,.0f}"
def style(fig, h=420, legend=False):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), showlegend=legend, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#12302F"), title_font=dict(family="Georgia", size=18, color=DEEP))
    fig.update_xaxes(gridcolor="#C9DEDA"); fig.update_yaxes(gridcolor="#C9DEDA"); return fig

# ------------------------------------------------------------------ shared analytics
def monthly(intent="All"):
    L = csv("monthly_leads.csv"); C = csv("monthly_completions.csv")
    if intent != "All": L = L[L.intent == intent]
    L = L.groupby("month_start").n.sum().rename("Leads")
    S = C[C.kind == "Sale completions"].set_index("month_start").n.rename("Sale completions")
    M = C[C.kind == "Let move-ins"].set_index("month_start").n.rename("Let move-ins")
    df = pd.concat([L, S, M], axis=1).fillna(0); df.index = pd.to_datetime(df.index); return df[df.index <= "2026-09-01"]

def season_fig(df, series):
    fig = go.Figure()
    colors = {"Leads": TEAL, "Sale completions": CORAL, "Let move-ins": AMBER}
    for s_ in series: fig.add_trace(go.Scatter(x=df.index, y=df[s_], name=s_, mode="lines", line=dict(color=colors[s_], width=3)))
    for y in range(df.index.min().year, df.index.max().year + 1):
        for m, col in ((8, "rgba(242,165,65,.28)"), (12, "rgba(15,139,141,.20)")):
            t0 = pd.Timestamp(y, m, 1)
            if df.index.min() <= t0 <= df.index.max(): fig.add_vrect(x0=t0 - pd.Timedelta(days=15), x1=t0 + pd.Timedelta(days=15), fillcolor=col, line_width=0)
    t = pd.Timestamp("2025-04-01")
    if df.index.min() <= t <= df.index.max(): fig.add_vline(x=t, line_dash="dot", line_color=SLATE, annotation_text="1 Apr 2025 stamp duty change", annotation_position="top left")
    fig.update_layout(legend=dict(orientation="h", y=1.1)); return style(fig, 450, True)

def stamp_stats():
    df = monthly(); f, m, a = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")
    return dict(lead_jump=df.Leads[m] / df.Leads[f] - 1, comp_x=df["Sale completions"][m] / df["Sale completions"][f],
                apr_drop=df["Sale completions"][a] / df["Sale completions"][m] - 1)

def seasonal_index(s_):
    s_ = s_[(s_.index.year >= 2024) & (s_.index.year <= 2025)]
    return (s_ / s_.groupby(s_.index.year).transform("median")).groupby(s_.index.month).mean()

# ------------------------------------------------------------------ pages
def page_overview():
    st.title("Veltrix customer & buyer segmentation")
    st.caption("Synthetic UK estate agency data, Jan 2023 to Sep 2026. Findings test the method, not real customers.")
    V = validation(); C = customers()
    a, b, c, d = st.columns(4)
    a.metric("Unique customers", f"{C.master_id.nunique():,}", f"{len(C) - C.master_id.nunique():,} duplicates merged", delta_color="off")
    b.metric("Segments", V["K"]); c.metric("Bootstrap stability", pct(V["stability"]), "target 80%", delta_color="off")
    dormant = SEG[SEG.segment.str.startswith("Dormant")].share.sum(); d.metric("Never enquired", pct(dormant, 0), "dormant segments", delta_color="off")
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(SEG, x="share", y="segment", orientation="h", color="segment", color_discrete_map=SEG_COLORS, text=SEG.share.map(lambda v: f"{v*100:.1f}%"),
                     title="Share of customers by segment"); fig.update_yaxes(categoryorder="array", categoryarray=SEG_ORDER[::-1], title=""); fig.update_xaxes(tickformat=".0%", title="")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        st.subheader("Headlines")
        ss = stamp_stats(); rb = csv("reply_bands.csv"); top3 = SEG.sort_values("share_of_deal_value", ascending=False).head(3)
        st.markdown(f"- **{top3.share_of_deal_value.sum()*100:.0f}% of deal value** comes from {', '.join(top3.segment)} ({top3.share.sum()*100:.0f}% of customers).\n"
                    f"- A reply **within 1 hour converts at {pct(rb.conversion.iloc[0])}**, against {pct(rb.conversion.iloc[-1])} after 24 hours.\n"
                    f"- **March 2025:** leads {ss['lead_jump']*100:+.0f}% on February; sale completions {ss['comp_x']:.1f}x, then {ss['apr_drop']*100:+.0f}% in April.\n"
                    f"- **{dormant*100:.0f}% of customers have no lead history**: the biggest reactivation pool.")
        st.info("Stability is below the 80% target and the real ARI against hidden personas has not been measured. See **Quality & validation**.")

def page_segments():
    st.title("Segment explorer")
    name = st.selectbox("Segment", SEG_ORDER); r = SEG[SEG.segment == name].iloc[0]
    c = st.columns(5)
    c[0].metric("Customers", f"{int(r.customers):,}", pct(r.share) + " of base", delta_color="off")
    c[1].metric("Lead conversion", pct(r.lead_conversion) if r.lead_conversion > 0 else "no leads"); c[2].metric("Avg deal value / customer", gbp(r.avg_deal_value_gbp))
    c[3].metric("Share of deal value", pct(r.share_of_deal_value)); c[4].metric("Late payments", pct(r.late_payment_rate))
    l, rr = st.columns(2)
    with l:
        st.subheader("Profile")
        prof = pd.DataFrame({"Attribute": ["Typical finance", "Region", "Main lead source", "Preferred property", "Top contact channel", "Top topics", "Median budget", "Median rent (pcm)", "Median first reply (min)"],
            "Value": [r.top_finance, r.top_region, r.top_lead_source, r.top_property_type, r.top_channel, r.top_topics, gbp(r.median_budget_gbp), gbp(r.median_rent_pcm_gbp), f"{r.median_first_reply_min:,.0f}"]})
        st.dataframe(prof, hide_index=True, width="stretch")
    with rr:
        st.subheader("How to treat them")
        st.success(f"**Tier {int(r.priority_tier)}** | {r.desk} desk | {r.route_to}\n\nFirst reply: **{r.first_reply_sla}**")
        st.caption("Tiers come from held-out conversion and value, not from the clustering itself.")
    st.subheader("All segments compared")
    show = SEG[["segment", "customers", "share", "lead_conversion", "avg_deal_value_gbp", "share_of_deal_value", "late_payment_rate", "priority_tier"]].rename(columns={
        "segment": "Segment", "customers": "Customers", "share": "Share", "lead_conversion": "Conversion", "avg_deal_value_gbp": "Avg deal value", "share_of_deal_value": "Share of value", "late_payment_rate": "Late payments", "priority_tier": "Tier"})
    st.dataframe(show.style.format({"Customers": "{:,.0f}", "Share": "{:.1%}", "Conversion": "{:.1%}", "Avg deal value": "£{:,.0f}", "Share of value": "{:.1%}", "Late payments": "{:.1%}"}),
                 hide_index=True, width="stretch")
    fig = px.scatter(SEG[SEG.lead_conversion > 0], x="lead_conversion", y="avg_deal_value_gbp", size="customers", color="segment", color_discrete_map=SEG_COLORS, text="segment",
                     title="Conversion vs value per customer (bubble = customers)", labels={"lead_conversion": "Lead conversion", "avg_deal_value_gbp": "Avg deal value (£)"})
    fig.update_traces(textposition="top center"); fig.update_xaxes(tickformat=".0%"); st.plotly_chart(style(fig, 450), width="stretch")

QUESTIONS = ["Q1. Who are our customers, and how many groups are there?", "Q2. Which segments convert best and create the most value?",
    "Q3. Which lead sources bring leads that convert?", "Q4. Does reply speed change whether we win the lead?",
    "Q5. How did the stamp duty change and the seasons shape demand?", "Q6. What does each segment want to buy or rent, and where?",
    "Q7. Which customers carry payment risk?", "Q8. How far can we trust the segments?"]
def page_qa():
    st.title("Eight business questions")
    q = st.selectbox("Question", QUESTIONS); i = QUESTIONS.index(q)
    if i == 0:
        st.markdown("**Seven groups. Two of them (dormant buyers and dormant renters) have never enquired.** The five active groups are mortgage buyers, active renters, buy-to-let investors, downsizers and overseas investors.")
        fig = px.pie(SEG, names="segment", values="customers", color="segment", color_discrete_map=SEG_COLORS, hole=.5); st.plotly_chart(style(fig, 400, True), width="stretch")
    elif i == 1:
        top = SEG.sort_values("share_of_deal_value", ascending=False).head(3)
        st.markdown(f"**{top.share_of_deal_value.sum()*100:.0f}% of deal value comes from {', '.join(top.segment)}.** Per customer, downsizers are worth the most and convert best.")
        m = SEG.melt(id_vars="segment", value_vars=["share", "share_of_deal_value"], var_name="Measure", value_name="v").replace({"share": "% of customers", "share_of_deal_value": "% of deal value"})
        fig = px.bar(m, x="segment", y="v", color="Measure", barmode="group", color_discrete_sequence=[MINT, TEAL], title="Customers vs deal value (hold-out customers)")
        fig.update_yaxes(tickformat=".0%", title=""); fig.update_xaxes(title="", categoryorder="array", categoryarray=SEG_ORDER); st.plotly_chart(style(fig, 430, True), width="stretch")
    elif i == 2:
        intent = st.radio("Lead intent", ["All", "Buy", "Rent"], horizontal=True); S = csv("source_conversion.csv")
        if intent != "All": S = S[S.intent == intent]
        S = S.groupby("source")[["closed_leads", "won"]].sum().reset_index(); S["conversion"] = S.won / S.closed_leads; S = S.sort_values("conversion", ascending=False)
        avg = S.won.sum() / S.closed_leads.sum()
        st.markdown(f"**Average conversion is {pct(avg)}.** Partners, walk-ins and referrals beat it; social and overseas channels trail it.")
        fig = px.bar(S, x="conversion", y="source", orientation="h", text=S.conversion.map(pct), hover_data={"closed_leads": ":,"}, color_discrete_sequence=[TEAL], title="Conversion by lead source (closed leads)")
        fig.add_vline(x=avg, line_dash="dot", line_color=CORAL); fig.update_yaxes(categoryorder="array", categoryarray=S.source.tolist()[::-1], title=""); fig.update_xaxes(tickformat=".0%", title="")
        st.plotly_chart(style(fig, 520), width="stretch")
    elif i == 3:
        b = csv("reply_bands.csv"); s = csv("reply_seniority.csv")
        st.markdown(f"**Yes.** A reply within one hour converts at {pct(b.conversion.iloc[0])}, four times the {pct(b.conversion.iloc[-1])} seen after 24 hours.")
        fig = px.bar(b, x="band", y="conversion", text=b.conversion.map(pct), color_discrete_sequence=[TEAL], title="Conversion by first-reply time"); fig.update_yaxes(tickformat=".0%", title=""); fig.update_xaxes(title="")
        st.plotly_chart(style(fig, 380), width="stretch")
        st.dataframe(s.rename(columns={"seniority": "Agent seniority", "conversion": "Conversion", "median_first_reply_min": "Median first reply (min)", "leads": "Closed leads"}).style.format({"Conversion": "{:.1%}", "Median first reply (min)": "{:,.0f}", "Closed leads": "{:,.0f}"}), hide_index=True)
        st.caption("Seniority may also reflect lead quality: this shows association, not proof of cause.")
    elif i == 4:
        ss = stamp_stats()
        st.markdown(f"**Buyers rushed in March 2025, then demand fell.** Leads {ss['lead_jump']*100:+.0f}% on February; sale completions {ss['comp_x']:.1f}x February, then {ss['apr_drop']*100:+.0f}% in April.")
        st.plotly_chart(season_fig(monthly(), ["Leads", "Sale completions"]), width="stretch"); st.caption("Orange = August, teal = December. See the Seasonality page for filters.")
    elif i == 5:
        st.markdown("**Each active segment has a distinct pattern.** Use the Demand explorer page for the full breakdown.")
        R = csv("demand_region.csv").set_index("persona"); P = csv("demand_price_band.csv").set_index("persona")
        rows = [{"Segment": s_, "Top regions": ", ".join(f"{k} {v*100:.0f}%" for k, v in R.loc[s_].sort_values(ascending=False).head(3).items()),
                 "Top price bands": ", ".join(f"{k} {v*100:.0f}%" for k, v in P.loc[s_].sort_values(ascending=False).head(2).items())} for s_ in R.index]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif i == 6:
        L = SEG[SEG.late_payment_rate.notna() & (SEG.lead_conversion > 0)].sort_values("late_payment_rate", ascending=False)
        st.markdown(f"**{L.segment.iloc[0]} ({pct(L.late_payment_rate.iloc[0])}) and {L.segment.iloc[1]} ({pct(L.late_payment_rate.iloc[1])}) pay late far more often** than {', '.join(L.segment.iloc[2:])}.")
        fig = px.bar(L, x="segment", y="late_payment_rate", text=L.late_payment_rate.map(pct), color_discrete_sequence=[CORAL], title="Share of settled payments made late"); fig.update_yaxes(tickformat=".0%", title=""); fig.update_xaxes(title="")
        st.plotly_chart(style(fig, 400), width="stretch")
        st.caption("Dormant segments have no payments. Suggested actions: standing orders for renters, clear transfer guidance for overseas buyers.")
    else: page_quality(embedded=True)

def page_seasonality():
    st.title("Seasonality and the stamp duty spike")
    c1, c2 = st.columns([1, 2]); intent = c1.radio("Lead intent", ["All", "Buy", "Rent"], horizontal=True)
    full = monthly(intent); lo, hi = full.index.min().to_pydatetime(), full.index.max().to_pydatetime()
    rng = c2.slider("Date range", lo, hi, (lo, hi), format="MMM YYYY"); df = full[(full.index >= rng[0]) & (full.index <= rng[1])]
    series = st.multiselect("Series", ["Leads", "Sale completions", "Let move-ins"], default=["Leads", "Sale completions"])
    if series: st.plotly_chart(season_fig(df, series), width="stretch")
    ss = stamp_stats(); a, b, c = st.columns(3)
    a.metric("Leads, Mar-25 vs Feb-25", f"{ss['lead_jump']*100:+.0f}%"); b.metric("Sale completions, Mar-25 vs Feb-25", f"{ss['comp_x']:.2f}x"); c.metric("Completions, Apr-25 vs Mar-25", f"{ss['apr_drop']*100:+.0f}%")
    st.subheader("August and December (month ÷ the year's median month, 2024 to 2025)")
    idx = pd.DataFrame({n: seasonal_index(monthly(intent)[n]).loc[[8, 12]] for n in ["Leads", "Sale completions", "Let move-ins"]}).rename(index={8: "August", 12: "December"})
    st.dataframe(idx.style.format("{:.2f}x"), width="stretch")
    st.caption("Leads dip in August and December; completions do not. 2023 completions ramp up from zero because the data starts in January 2023, so only full years are used.")

def page_demand():
    st.title("Demand explorer")
    dims = {"Region": "demand_region.csv", "Price band": "demand_price_band.csv", "Property type": "demand_property_type.csv", "Payment structure": "demand_payment_structure.csv"}
    dim = st.radio("Dimension", list(dims), horizontal=True); T = csv(dims[dim]).set_index("persona")
    st.caption("Share of each segment's viewings (transactions for payment structure). Dormant segments have no viewings.")
    fig = px.imshow(T.values, x=T.columns, y=T.index, color_continuous_scale=["#F3F9F7", TEAL, DEEP], aspect="auto", text_auto=".0%"); fig.update_xaxes(tickangle=-40)
    st.plotly_chart(style(fig, 420), width="stretch")
    seg = st.selectbox("Zoom into a segment", T.index.tolist()); s_ = T.loc[seg].sort_values(ascending=False).head(10)
    f2 = px.bar(x=s_.values, y=s_.index, orientation="h", color_discrete_sequence=[TEAL], text=[f"{v*100:.0f}%" for v in s_.values], title=f"{seg}: top {dim.lower()} choices")
    f2.update_yaxes(categoryorder="total ascending", title=""); f2.update_xaxes(tickformat=".0%", title=""); st.plotly_chart(style(f2, 380), width="stretch")
    st.subheader("Listing popularity (15.2M property views)")
    PV = csv("property_views.csv"); by = PV.groupby("region")[["views", "listings"]].sum(); by["Views per listing"] = by.views / by.listings
    f3 = px.bar(by.sort_values("Views per listing"), x="Views per listing", y=by.sort_values("Views per listing").index, orientation="h", color_discrete_sequence=[AMBER]); f3.update_yaxes(title="")
    st.plotly_chart(style(f3, 400), width="stretch"); st.caption("Property views cannot be linked to customers (no customer ID), so this shows listings, not segments.")

def page_customers():
    st.title("Customers")
    t1, t2, t3 = st.tabs(["Look up a customer", "Filter and export", "Score new customers"]); C = customers()
    with t1:
        cid = st.text_input("Customer ID", placeholder=C.customer_id.iloc[0]); st.caption(f"Try: {', '.join(C.customer_id.sample(3, random_state=1))}")
        if cid:
            row = C[C.customer_id == cid.strip()]
            if row.empty: st.warning("No such customer ID.")
            else:
                r = row.iloc[0]; a, b, c = st.columns(3); a.metric("Segment", r.segment); b.metric("Priority tier", int(r.priority_tier)); c.metric("Desk", r.desk)
                st.success(f"Route to **{r.route_to}** | first reply **{r.first_reply_sla}**")
                if r.customer_id != r.master_id: st.info(f"This record is a merged duplicate of master customer {r.master_id}.")
                st.dataframe(row.drop(columns=["segment", "priority_tier", "desk", "route_to", "first_reply_sla"]).T.rename(columns={row.index[0]: "Value"}), width="stretch")
    with t2:
        a, b, c = st.columns(3); sg = a.multiselect("Segment", SEG_ORDER); tr = b.multiselect("Tier", [1, 2, 3]); rg = c.multiselect("Region", sorted(C.region_group.dropna().unique()))
        F = C
        if sg: F = F[F.segment.isin(sg)]
        if tr: F = F[F.priority_tier.isin(tr)]
        if rg: F = F[F.region_group.isin(rg)]
        st.write(f"**{len(F):,}** records match"); st.dataframe(F.head(1000), hide_index=True, width="stretch")
        st.download_button("Download matching records (CSV)", F.to_csv(index=False).encode(), "veltrix_customers_filtered.csv", "text/csv")
    with t3:
        bundle, err = model_bundle()
        if bundle is None: st.error(f"Model could not be loaded ({err}). Install kmodes==0.12.2 and scikit-learn==1.8.0.")
        else:
            st.write("Upload a CSV with the engineered feature columns (same schema as the template) to assign segments.")
            st.download_button("Download template CSV", (D / "scoring_template.csv").read_bytes(), "scoring_template.csv", "text/csv")
            up = st.file_uploader("Feature CSV", type="csv")
            if up:
                df = pd.read_csv(up, keep_default_na=False, na_values=[""]); need = bundle["NUM_COLS"] + bundle["CAT_COLS"]; miss = [c for c in need if c not in df.columns]
                if miss: st.error(f"Missing columns: {', '.join(miss[:10])}{'...' if len(miss) > 10 else ''}")
                elif len(df) > 20000: st.error("Please upload at most 20,000 rows at a time.")
                elif df[bundle["NUM_COLS"]].isna().any().any(): st.error("Numeric feature columns contain blanks. Fill or remove those rows first.")
                else:
                    with st.spinner("Scoring..."):
                        cats = df[bundle["CAT_COLS"]].fillna("None").astype(str)   # the text value "None" is a real category, not a blank
                        X = np.hstack([bundle["scaler"].transform(df[bundle["NUM_COLS"]]), cats.to_numpy()]).astype(object)
                        lab = bundle["model"].predict(X, categorical=list(range(len(bundle["NUM_COLS"]), len(need))))
                    out = df.copy(); out.insert(0, "segment", pd.Series(lab).map(bundle["names"]).values)
                    st.success(f"Scored {len(out):,} customers"); st.bar_chart(out.segment.value_counts())
                    st.dataframe(out.head(200), width="stretch"); st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")

def page_quality(embedded=False):
    if not embedded: st.title("Quality & validation")
    V = validation(); st.subheader("Scorecard against the project targets")
    sc = pd.DataFrame([["Active leads resolved to one customer ID", "≥ 95%", "100% (largely by construction)", "Pass"], ["Number of segments", "6 to 8", str(V["K"]), "Pass"],
        ["Stability across 20 bootstrap refits", "≥ 80%", f"{pct(V['stability'])} (range {pct(V['stability_min'], 0)} to {pct(V['stability_max'], 0)})", "Below target"],
        ["ARI against hidden personas", "≥ 0.5", "Not measured (answer key unavailable to the run)", "Pending"],
        ["Segments differ on held-out outcomes", "significant", "Conversion, deal value, late payment: p < 0.001", "Pass"]], columns=["Measure", "Target", "Result", "Status"])
    st.dataframe(sc, hide_index=True, width="stretch")
    st.caption(f"Proxy check: k-prototypes agrees with the buyer-position field at ARI 0.315 vs {V['kmeans_proxy_ari']:.3f} for k-means. This is a proxy, not the project ARI.")
    stab = csv("stability.csv"); fig = px.bar(stab, x="bootstrap", y="matched_agreement", color_discrete_sequence=[TEAL], title="Stability of each bootstrap refit")
    fig.add_hline(y=0.8, line_dash="dot", line_color=CORAL, annotation_text="80% target"); fig.update_yaxes(tickformat=".0%", range=[0, 1], title=""); st.plotly_chart(style(fig, 360), width="stretch")
    if embedded: return
    k = csv("k_curve.csv"); f2 = go.Figure(); f2.add_trace(go.Scatter(x=k.k, y=k.gower_silhouette, name="k-prototypes", line=dict(color=TEAL, width=3)))
    f2.add_trace(go.Scatter(x=k.k, y=k.hier_silhouette, name="hierarchical", line=dict(color=AMBER, dash="dash"))); f2.add_vrect(x0=5.5, x1=8.5, fillcolor="rgba(15,139,141,.10)", line_width=0)
    f2.update_layout(title="Gower silhouette by k (shaded = permitted 6 to 8)", xaxis_title="k"); st.plotly_chart(style(f2, 360, True), width="stretch")
    st.subheader("Data-quality issue log"); st.dataframe(csv("data_quality_issue_log.csv"), hide_index=True, width="stretch")
    st.subheader("Limitations")
    st.markdown("- Synthetic data. Conversion and deal value are association, not cause.\n- About 40% of customers have no lead history, so the two dormant segments are blunt.\n"
                "- The web-session table was not available to this run, so first-time buyers and home movers are merged into *Mortgage buyers*.\n- Segment names are rule-based suggestions for sign-off.")

PAGES = {"Overview": page_overview, "Segment explorer": page_segments, "Business questions": page_qa, "Seasonality": page_seasonality,
         "Demand explorer": page_demand, "Customers": page_customers, "Quality & validation": page_quality}
with st.sidebar:
    st.markdown("## 🏠 Veltrix"); choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed"); st.caption("Customer & buyer segmentation")
PAGES[choice]()
