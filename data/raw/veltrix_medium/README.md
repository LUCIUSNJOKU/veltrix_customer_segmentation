# Veltrix synthetic dataset

Synthetic data for **Veltrix Customer and Buyer Segmentation: A Case Study**. Veltrix is a made-up UK estate agency and new-homes developer, and none of the data describes real people. Customers have pseudonymous IDs plus SHA-256 hashes of their email and phone (no raw contact details). Money is in GBP. The data covers 1 January 2023 to 30 September 2026.

## What's in the package

| File | Contents |
|---|---|
| `veltrix_medium_parquet.zip` | **Main project dataset** (Medium scale): 100,000 base customers, about 19.1M rows, Parquet (zstd) |
| `veltrix_small_csv.zip` | Small scale (10,000 customers, about 2M rows) as CSV, for testing the pipeline, Excel and Power BI |
| `veltrix_generate.py` | The generator. Use `--seed 42` (the default) to rebuild exactly the same data |
| `validate.py` | Checks keys, consistency and realism with DuckDB |
| `recoverability_check.py` | Clustering baseline scored against the answer key. **Run this only after Phase 4** |
| `*_validation.txt` | Validation output for each delivered scale |

Inside each zip, `_restricted/_answer_key_customer_persona` holds the hidden persona of every customer and links each duplicate to its original (`duplicate_of`). **Give this folder to the project lead only. It must never be used as a model input.**

## Row counts (seed 42)

| Table | Small | Medium |
|---|---|---|
| locations | 36 | 36 |
| agents | 75 | 300 |
| customers | 10,200 | 102,000 |
| campaigns | 270 | 270 |
| leads | 9,624 | 95,804 |
| interactions | 53,962 | 532,789 |
| properties | 4,000 | 16,667 |
| site_visits | 23,794 | 233,593 |
| transactions | 1,737 | 16,665 |
| payments | 11,628 | 106,199 |
| campaign_responses | 81,259 | 808,953 |
| web_sessions | 205,668 | 2,003,460 |
| property_views | 1,569,520 | 15,163,137 |
| **Total** | **~1.97M** | **~19.1M** |

`web_sessions` and `property_views` are folders of Parquet part files. In the small CSV set each of them is a single CSV. Note that small `property_views.csv` has 1.57M rows, which is more than Excel's limit of 1,048,576, so open it in Power BI, DuckDB or pandas.

## Quick start (DuckDB)

```python
import duckdb
con = duckdb.connect()
D = "veltrix_medium"
for t in ["customers","leads","agents","locations","properties","interactions","site_visits",
          "transactions","payments","campaigns","campaign_responses"]:
    con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{D}/{t}.parquet')")
con.execute(f"CREATE VIEW web_sessions AS SELECT * FROM read_parquet('{D}/web_sessions/*.parquet')")
con.execute(f"CREATE VIEW property_views AS SELECT * FROM read_parquet('{D}/property_views/*.parquet')")
con.sql("SELECT lead_source, count(*) FROM leads GROUP BY 1 ORDER BY 2 DESC").show()
```

Rebuild a scale yourself: `python veltrix_generate.py --scale medium --out data/medium` (takes about 30 s). The `large` scale (500k customers, 76M property views) takes about 3 minutes and needs about 4 GB of RAM. `xlarge` needs about 16 GB of RAM.

## Hidden structure (what a good model should find)

Eight personas generate every customer. They overlap on purpose:

| Persona | Share | Key signals built in |
|---|---|---|
| first_time_buyer | 22% | ~£245k, Midlands and North, flats and terraces, shared ownership and Lifetime ISAs, 80% mobile, TikTok and Instagram, slow decisions |
| home_mover_family | 18% | ~£465k, South East and East, semis and detached homes, chain-collapse losses, school-catchment topics |
| downsizer | 8% | Aged 55+, 72% cash or equity release, bungalows, South West and Wales, phone and branch visits, high conversion |
| btl_investor | 12% | ~£190k, high-yield North and North East, 60% limited-company SPVs, repeat enquiries, largest discounts (~5.8%) |
| overseas_investor | 7% | ~£640k, London new-build and off-plan (85% of views), HK, UAE, China, Nigeria and others, ~23% of sessions between midnight and 6am UK time, 20% virtual tours, international transfers |
| corporate_relocation | 4% | Company lets of about £3,200 pcm, relocation partners, fast decisions, punctual payment |
| young_renter | 14% | ~£1,450 pcm, 82% mobile, WhatsApp, quick to decide, highest rate of late rent (~14%) |
| window_shopper | 15% | Heavy browsing, rarely enquire, ~2% conversion, 70% missing budgets |

## Built-in data-quality problems (Phase 2 work)

- About 2% duplicate customers. Each duplicate carries its own ID, slightly different answers and one extra lead. About 55% share their original's `email_sha256` and about 50% share its `phone_sha256`; the rest can only be found from their profile.
- Missing budgets, from 20% to 70% depending on persona. About 0.4% of budgets have typos: an extra zero, a value entered in £k, or a stray leading 1. Some rent budgets are weekly or annual figures.
- 14 inconsistent lead-source spellings, such as `rightmove`, `Right Move`, `OTM`, `FB`, `Insta`, `Walk-in` and `veltrix.co.uk`.
- Mixed country labels, such as `United Kingdom`, `UK`, `England`, `Great Britain`, `U.K.`, `HK`, `UAE` and `KSA`.
- Missing values: 18% of `age_band`, 25% of `occupation_sector`, 4% of `size_sqm`, 7% of `epc_rating` and 2% of `response_time_min`.
- 25% of web sessions are anonymous, 2% of IP countries are VPN noise, and about 8% of overseas visitors browse from the UK.

## Business dynamics

- Seasonality includes a spring peak, an August dip for sales, a summer peak for lettings and a December slump. There is about 6% growth a year.
- The stamp duty change on 1 April 2025 is visible: in March 2025 leads are about 40% above February and sale completions are more than double, followed by a fall in April.
- Senior agents reply faster (median ~74 minutes, against ~575 for juniors). Faster replies raise conversion: about 31% when the reply comes within 1 hour, against about 10% after 24 hours.
- About 18% of closed leads convert. Open leads (about 4.5%) still sit in pipeline stages at the end of the data.

## Validation results (Medium)

- All 17 foreign-key relationships have 0 orphan keys.
- No sale property is sold twice. No lead comes before its customer's first contact. No viewing comes before its lead or before the property was marketed. No agent handles a lead before their hire date.
- Persona shares are within ±0.3 percentage points of the design.
- Recoverability:
  - Naive k-means on raw features reaches an adjusted Rand index (ARI) of about 0.40–0.50.
  - Curated features with k-prototypes reach an ARI of about 0.63–0.72 (the worked solution: 0.72).
  - The project target of ARI ≥ 0.5 is therefore reachable, but only with real feature engineering.
  - Gower silhouettes are about 0.16–0.19, below the 0.25 target; this is normal for overlapping personas on mixed data.
