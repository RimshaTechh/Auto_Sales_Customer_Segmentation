import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Auto Sales — Customer Segmentation", layout="wide")

# ---------------------------------------------------------
# COLUMN REQUIREMENTS + ALIASES
# ---------------------------------------------------------
REQUIRED_COLUMNS = [
    "CUSTOMERNAME", "COUNTRY", "PRODUCTLINE",
    "DAYS_SINCE_LASTORDER", "ORDERNUMBER", "SALES"
]

COLUMN_ALIASES = {
    "COUNTRY": ["COUNTRY", "NATION", "COUNTRY_NAME", "COUNTRYNAME"],
    "CUSTOMERNAME": ["CUSTOMERNAME", "CUSTOMER_NAME", "CUSTOMER NAME", "CUSTOMER", "CLIENT", "CLIENTNAME"],
    "PRODUCTLINE": ["PRODUCTLINE", "PRODUCT_LINE", "PRODUCT LINE", "PRODUCT"],
    "SALES": ["SALES", "SALE_AMOUNT", "SALEAMOUNT", "REVENUE", "TOTAL_SALES", "TOTALSALES", "AMOUNT"],
    "ORDERNUMBER": ["ORDERNUMBER", "ORDER_NUMBER", "ORDER NUMBER", "ORDER_ID", "ORDERID"],
    "DAYS_SINCE_LASTORDER": [
        "DAYS_SINCE_LASTORDER", "DAYS_SINCE_LAST_ORDER", "DAYSSINCELASTORDER",
        "RECENCY_DAYS", "RECENCYDAYS", "RECENCY"
    ],
}


# ---------------------------------------------------------
# HELPER FUNCTIONS (all defined up top, before use)
# ---------------------------------------------------------
def map_columns(df):
    """Normalize column names and auto-map common naming variants to the
    standard names the app expects."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper().str.replace(r"\s+", "_", regex=True)

    rename_map = {}
    for standard, aliases in COLUMN_ALIASES.items():
        if standard in df.columns:
            continue
        for alias in aliases:
            alias_norm = alias.strip().upper().replace(" ", "_")
            if alias_norm in df.columns:
                rename_map[alias_norm] = standard
                break
    return df.rename(columns=rename_map)


def label_cluster(row, profile_df):
    """Generate a human-readable label for a cluster based on its
    relative Recency / Frequency / Monetary standing vs other clusters."""
    n = profile_df.shape[0]
    recency_rank = profile_df["Avg_Recency"].rank()[row.name]          # lower recency = better
    monetary_rank = profile_df["Avg_Monetary"].rank(ascending=False)[row.name]  # higher spend = better
    frequency_rank = profile_df["Avg_Frequency"].rank(ascending=False)[row.name]  # higher freq = better

    is_recent = recency_rank <= n / 2
    is_high_spender = monetary_rank <= n / 2
    is_frequent = frequency_rank <= n / 2

    if is_high_spender and is_frequent and is_recent:
        return "💎 Top / VIP Customers"
    elif is_high_spender and not is_recent:
        return "⚠️ High-Value but At Risk"
    elif is_frequent and is_recent and not is_high_spender:
        return "🔁 Loyal Regular Buyers"
    elif not is_recent and not is_frequent:
        return "😴 Inactive / Churned"
    else:
        return "🌱 Occasional / New Customers"


def dedupe_labels(profile_df):
    """If two clusters share a label, append their avg spend to tell them apart."""
    profile_df = profile_df.copy()
    counts = profile_df["Segment_Label"].value_counts()
    dupes = counts[counts > 1].index.tolist()

    for label in dupes:
        mask = profile_df["Segment_Label"] == label
        subset = profile_df[mask].sort_values("Avg_Monetary", ascending=False)
        for idx in subset.index:
            avg_val = profile_df.loc[idx, "Avg_Monetary"]
            profile_df.loc[idx, "Segment_Label"] = f"{label} (avg ${avg_val:,.0f})"

    return profile_df


@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df


@st.cache_resource
def load_pretrained_model():
    if os.path.exists("kmeans_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("kmeans_model.pkl")
        scl = joblib.load("scaler.pkl")
        return model, scl
    return None, None


# ---------------------------------------------------------
# SIDEBAR — DATA UPLOAD
# ---------------------------------------------------------
st.sidebar.header("Data")

SAMPLE_CSV_PATH = "Auto Sales data.csv"  # the file the model was trained on

if os.path.exists(SAMPLE_CSV_PATH):
    with open(SAMPLE_CSV_PATH, "rb") as f:
        st.sidebar.download_button(
            "Download sample CSV (training data)",
            data=f,
            file_name="Auto_Sales_sample.csv",
            mime="text/csv",
            help="This is the exact file structure the app expects — use it as a reference."
        )
else:
    st.sidebar.caption("Sample CSV not found in app directory.")

uploaded_file = st.sidebar.file_uploader("Upload Auto Sales CSV", type="csv")

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
else:
    st.info("👆 Please upload a CSV file from the sidebar to see the analysis.")
    st.stop()

# ---------------------------------------------------------
# COLUMN VALIDATION / AUTO-MAPPING
# ---------------------------------------------------------
df = map_columns(df_raw)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing_cols:
    st.error(
        "Some required columns could not be automatically detected in your CSV: "
        f"**{', '.join(missing_cols)}**"
    )
    st.info(
        "You can either re-upload a CSV that includes these columns, "
        "or manually map your existing columns to the required ones below."
    )

    with st.expander("Manually map columns", expanded=True):
        available_cols = ["-- None --"] + df_raw.columns.tolist()
        manual_map = {}
        for col in missing_cols:
            choice = st.selectbox(f"Which column corresponds to **{col}**?", available_cols, key=f"map_{col}")
            if choice != "-- None --":
                manual_map[col] = choice

        if manual_map:
            for standard, source_col in manual_map.items():
                df[standard] = df_raw[source_col]

    still_missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if still_missing:
        st.error(f"Still missing required column(s): {', '.join(still_missing)}. Cannot proceed.")
        st.stop()
    else:
        st.success("All required columns mapped. Continuing below.")

# Coerce numeric columns in case they came in as text / had stray characters
df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")
df["DAYS_SINCE_LASTORDER"] = pd.to_numeric(df["DAYS_SINCE_LASTORDER"], errors="coerce")

dropped = df["SALES"].isna().sum() + df["DAYS_SINCE_LASTORDER"].isna().sum()
if dropped > 0:
    st.warning(f"{dropped} row(s) had non-numeric SALES / DAYS_SINCE_LASTORDER values and were dropped.")
df = df.dropna(subset=["SALES", "DAYS_SINCE_LASTORDER"])

if df.empty:
    st.error("No valid rows remain after cleaning. Please check your CSV's data quality.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

countries = sorted(df["COUNTRY"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Country", countries, default=countries)

product_lines = sorted(df["PRODUCTLINE"].dropna().unique().tolist())
selected_products = st.sidebar.multiselect("Product line", product_lines, default=product_lines)

df_filtered = df[
    df["COUNTRY"].isin(selected_countries) &
    df["PRODUCTLINE"].isin(selected_products)
]

st.sidebar.header("Segmentation")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=9, value=3, step=1)

# ---------------------------------------------------------
# HEADER + METRICS
# ---------------------------------------------------------
st.title("Auto Sales — Customer Segmentation")
st.caption("RFM analysis + KMeans clustering, based on the exploratory notebook.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Orders", f"{df_filtered.shape[0]:,}")
col2.metric("Customers", f"{df_filtered['CUSTOMERNAME'].nunique():,}")
col3.metric("Total sales", f"${df_filtered['SALES'].sum():,.0f}")
col4.metric("Avg order value", f"${df_filtered['SALES'].mean():,.0f}")

st.divider()

# ---------------------------------------------------------
# EDA CHARTS
# ---------------------------------------------------------
st.subheader("Overview")

c1, c2 = st.columns(2)

with c1:
    fig_sales = px.histogram(df_filtered, x="SALES", nbins=40, title="Distribution of order sales")
    st.plotly_chart(fig_sales, use_container_width=True)

with c2:
    top_countries = df_filtered["COUNTRY"].value_counts().head(10).reset_index()
    top_countries.columns = ["COUNTRY", "ORDERS"]
    fig_country = px.bar(top_countries, x="ORDERS", y="COUNTRY", orientation="h",
                          title="Top 10 countries by order count")
    fig_country.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_country, use_container_width=True)

fig_product = px.bar(
    df_filtered["PRODUCTLINE"].value_counts().reset_index(),
    x="count", y="PRODUCTLINE", orientation="h",
    title="Orders by product line"
)
fig_product.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_product, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# RFM COMPUTATION (live, driven by sidebar filters + k)
# ---------------------------------------------------------
st.subheader("Customer segments")

if df_filtered["CUSTOMERNAME"].nunique() < k:
    st.error("Not enough customers in the current filter selection for this many clusters. "
              "Reduce k or widen your filters.")
    st.stop()

rfm = df_filtered.groupby("CUSTOMERNAME").agg(
    Recency=("DAYS_SINCE_LASTORDER", "min"),
    Frequency=("ORDERNUMBER", "nunique"),
    Monetary=("SALES", "sum")
).reset_index()

rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

features = rfm[["Recency", "Frequency", "Monetary_log"]]
live_scaler = StandardScaler()
X_scaled = live_scaler.fit_transform(features)

live_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = live_kmeans.fit_predict(X_scaled).astype(str)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
rfm["PCA1"] = coords[:, 0]
rfm["PCA2"] = coords[:, 1]

# Cluster profile (numeric summary)
profile = rfm.groupby("Cluster").agg(
    Customers=("CUSTOMERNAME", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Monetary=("Monetary", "sum")
).round(1)

# Human-readable, de-duplicated label per cluster
profile["Segment_Label"] = profile.apply(lambda r: label_cluster(r, profile), axis=1)
profile = dedupe_labels(profile)

# Map the label back onto individual customers for charts/tables
rfm["Segment_Label"] = rfm["Cluster"].map(profile["Segment_Label"])

# PCA scatter — colored by human-readable segment instead of raw cluster number
fig_pca = px.scatter(
    rfm, x="PCA1", y="PCA2", color="Segment_Label",
    hover_data=["CUSTOMERNAME", "Recency", "Frequency", "Monetary"],
    title=f"Customer segments (k={k}) — PCA projection"
)
st.plotly_chart(fig_pca, use_container_width=True)

# Cluster profile tabs
cluster_ids = sorted(rfm["Cluster"].unique(), key=int)
tabs = st.tabs([f"{profile.loc[c, 'Segment_Label']}" for c in cluster_ids])

for tab, cid in zip(tabs, cluster_ids):
    with tab:
        row = profile.loc[cid]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers", int(row["Customers"]))
        m2.metric("Avg recency (days)", f"{row['Avg_Recency']:.0f}")
        m3.metric("Avg frequency", f"{row['Avg_Frequency']:.1f}")
        m4.metric("Avg spend", f"${row['Avg_Monetary']:,.0f}")
        st.dataframe(
            rfm[rfm["Cluster"] == cid][["CUSTOMERNAME", "Recency", "Frequency", "Monetary"]]
            .sort_values("Monetary", ascending=False),
            use_container_width=True, hide_index=True
        )

st.divider()

# ---------------------------------------------------------
# CUSTOMER LOOKUP
# ---------------------------------------------------------
st.subheader("Customer lookup")

customer = st.selectbox("Choose a customer", sorted(rfm["CUSTOMERNAME"].unique()))
lookup_row = rfm[rfm["CUSTOMERNAME"] == customer].iloc[0]

lc1, lc2, lc3, lc4 = st.columns(4)
lc1.metric("Segment", lookup_row["Segment_Label"])
lc2.metric("Recency (days)", f"{lookup_row['Recency']:.0f}")
lc3.metric("Frequency", f"{lookup_row['Frequency']:.0f}")
lc4.metric("Monetary", f"${lookup_row['Monetary']:,.0f}")

st.divider()

# ---------------------------------------------------------
# PREDICT CLUSTER FOR A NEW / HYPOTHETICAL CUSTOMER
# ---------------------------------------------------------
st.subheader("Predict a customer's segment")
st.caption("Enter RFM values for any customer (existing or hypothetical) to see which segment they'd fall into.")

pretrained_model, pretrained_scaler = load_pretrained_model()

use_pretrained = pretrained_model is not None
if use_pretrained:
    st.info(f"Using saved model (kmeans_model.pkl, trained with k={pretrained_model.n_clusters}).")
else:
    st.info(f"No saved model found — using the live model above (k={k}).")

r_min, r_max = int(rfm["Recency"].min()), int(rfm["Recency"].max())
f_min, f_max = int(rfm["Frequency"].min()), int(rfm["Frequency"].max())
m_min, m_max = float(rfm["Monetary"].min()), float(rfm["Monetary"].max())

p1, p2, p3 = st.columns(3)
with p1:
    input_recency = st.slider("Recency (days since last order)", r_min, r_max, int(rfm["Recency"].median()))
with p2:
    input_frequency = st.slider("Frequency (number of orders)", f_min, f_max, int(rfm["Frequency"].median()))
with p3:
    input_monetary = st.slider("Monetary (total spend, $)", m_min, m_max, float(rfm["Monetary"].median()))

if st.button("Predict cluster"):
    input_monetary_log = np.log1p(input_monetary)
    input_features = pd.DataFrame(
        [[input_recency, input_frequency, input_monetary_log]],
        columns=["Recency", "Frequency", "Monetary_log"]
    )

    if use_pretrained:
        input_scaled = pretrained_scaler.transform(input_features)
        predicted_cluster = pretrained_model.predict(input_scaled)[0]
    else:
        input_scaled = live_scaler.transform(input_features)
        predicted_cluster = live_kmeans.predict(input_scaled)[0]

    match_profile = profile[profile.index == str(predicted_cluster)]
    segment_name = (
        match_profile["Segment_Label"].values[0]
        if not match_profile.empty else f"Cluster {predicted_cluster}"
    )

    st.success(f"Predicted segment: **{segment_name}**  \n(Cluster ID: {predicted_cluster})")

    if not match_profile.empty:
        st.write("Typical profile of this segment:")
        st.dataframe(
            match_profile[["Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary", "Total_Monetary"]],
            use_container_width=True
        )
