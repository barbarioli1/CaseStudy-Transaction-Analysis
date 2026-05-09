import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Transaction Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Transaction Analytics Dashboard")

REQUIRED_COLUMNS = [
    "TransactionNo", "Date", "ProductNo", "ProductName",
    "Price", "Quantity", "CustomerNo", "Country"
]


def money(value):
    return f"${value:,.2f}"


def number(value):
    return f"{value:,.0f}"


def table(df):
    st.dataframe(df, width="stretch", hide_index=True)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def upper_iqr(series, multiplier=5):
    q1, q3 = series.quantile([0.25, 0.75])
    return q3 + multiplier * (q3 - q1)


def remove_outliers(df):
    groups = ["CustomerNo", "day", "Country"]
    counts = df.groupby(groups).size().rename("transactions_same_customer_day_country").reset_index()
    counts["is_transaction_count_outlier"] = counts["transactions_same_customer_day_country"] > upper_iqr(counts["transactions_same_customer_day_country"])

    df = df.merge(counts, on=groups, how="left")
    df["is_quantity_outlier"] = df["Quantity"] > upper_iqr(df["Quantity"])

    mask = df["is_transaction_count_outlier"] | df["is_quantity_outlier"]
    return df[~mask].copy(), df[mask].copy()


@st.cache_data
def clean_data(raw):
    df = raw[REQUIRED_COLUMNS].copy()
    start_rows = len(df)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Price", "Quantity", "CustomerNo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["TransactionNo", "ProductNo", "ProductName", "Country"]:
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna().copy()
    missing_removed = start_rows - len(df)

    positive = df["Price"].gt(0) & df["Quantity"].gt(0)
    non_positive_removed = int((~positive).sum())
    df = df[positive].copy()

    df["Quantity"] = df["Quantity"].astype("Int64")
    df["CustomerNo"] = df["CustomerNo"].astype("Int64")
    df["gross_revenue"] = df["Price"] * df["Quantity"]
    df["year"] = df["Date"].dt.year
    df["quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["week"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["day"] = df["Date"].dt.date
    df["weekday"] = df["Date"].dt.day_name()

    df, outliers = remove_outliers(df)

    return df, outliers, {
        "missing_removed": missing_removed,
        "non_positive_removed": non_positive_removed,
        "outliers_removed": len(outliers),
    }


def metric_row(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


def country_summary(df):
    out = df.groupby("Country", as_index=False).agg(
        total_value=("gross_revenue", "sum"),
        transaction_count=("gross_revenue", "count"),
        average_transaction=("gross_revenue", "mean"),
        median_transaction=("gross_revenue", "median"),
    ).sort_values("total_value", ascending=False)
    out["cumulative_revenue_pct"] = out["total_value"].cumsum() / out["total_value"].sum() * 100
    return out


def pareto_chart(df):
    fig = go.Figure()
    fig.add_bar(x=df["Country"], y=df["total_value"], name="Revenue")
    fig.add_scatter(x=df["Country"], y=df["cumulative_revenue_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers")
    fig.update_layout(
        title="Country Revenue Pareto Chart",
        xaxis_title="Country",
        yaxis_title="Revenue",
        yaxis2={"title": "Cumulative Revenue %", "overlaying": "y", "side": "right", "range": [0, 110], "ticksuffix": "%"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def quarter_share_heatmap(df):
    revenue = df.pivot_table(index="Country", columns="quarter", values="gross_revenue", aggfunc="sum", fill_value=0)
    share = revenue.div(revenue.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return px.imshow(
        share.sort_index(),
        aspect="auto",
        title="Quarter Share of Total Country Revenue",
        labels={"x": "Quarter", "y": "Country", "color": "% of country revenue"},
    )


def customer_metrics(df):
    customers = df.groupby(["CustomerNo", "Country"], as_index=False).agg(
        first_purchase=("Date", "min"),
        last_purchase=("Date", "max"),
        orders=("TransactionNo", "nunique"),
        revenue=("gross_revenue", "sum"),
        units=("Quantity", "sum"),
    )
    customers["buyer_type"] = np.where(customers["orders"].eq(1), "One-time buyer", "Repeat buyer")
    customers["lifetime_days"] = (customers["last_purchase"] - customers["first_purchase"]).dt.days
    customers["revenue_per_order"] = customers["revenue"] / customers["orders"]

    buyer_summary = customers.groupby("buyer_type", as_index=False).agg(
        customers=("CustomerNo", "nunique"),
        revenue=("revenue", "sum"),
        avg_revenue_per_customer=("revenue", "mean"),
        avg_orders_per_customer=("orders", "mean"),
    )
    buyer_summary["customer_share_pct"] = buyer_summary["customers"] / buyer_summary["customers"].sum() * 100

    lifetime_by_country = customers.groupby("Country", as_index=False).agg(
        customers=("CustomerNo", "nunique"),
        avg_lifetime_days=("lifetime_days", "mean"),
        median_lifetime_days=("lifetime_days", "median"),
        avg_orders_per_customer=("orders", "mean"),
        avg_customer_value=("revenue", "mean"),
    )

    return customers, buyer_summary.round(2), lifetime_by_country.round(2)


def monthly_customer_mix(df, history):
    first_month = history.groupby(["CustomerNo", "Country"], as_index=False)["month"].min().rename(columns={"month": "first_month"})
    monthly = df[["CustomerNo", "Country", "month"]].drop_duplicates().merge(first_month, on=["CustomerNo", "Country"])
    monthly["new_customers"] = monthly["month"].eq(monthly["first_month"])

    out = monthly.groupby(["month", "Country"], as_index=False).agg(
        total_customers=("CustomerNo", "nunique"),
        new_customers=("new_customers", "sum"),
    )
    out["returning_customers"] = out["total_customers"] - out["new_customers"]
    return out


def filtered(df, outliers):
    st.sidebar.subheader("Filters")
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    countries = sorted(df["Country"].dropna().unique())
    selected = st.sidebar.multiselect("Countries", countries, default=countries)

    if len(date_range) == 2:
        start, end = date_range
        df = df[df["Date"].dt.date.between(start, end)]
        outliers = outliers[outliers["Date"].dt.date.between(start, end)]

    return df[df["Country"].isin(selected)].copy(), outliers[outliers["Country"].isin(selected)].copy()


raw = load_data("Sales_Transactions_Interview_Dataset.csv")
df, outliers, cleaning_kpis = clean_data(raw)
full_df = df.copy()
df, outliers = filtered(df, outliers)

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

metric_row([
    ("Total Value", money(df["gross_revenue"].sum())),
    ("Transactions", number(len(df))),
    ("Avg Transaction", money(df["gross_revenue"].mean())),
    ("Median Transaction", money(df["gross_revenue"].median())),
    ("Countries", number(df["Country"].nunique())),
])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Country Analysis", "Time Trends", "Outliers", "Raw Data", "Customer Analysis"])

with tab1:
    st.subheader("Overview")
    monthly = df.groupby("month", as_index=False).agg(total_value=("gross_revenue", "sum"), transaction_count=("gross_revenue", "count"))
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.line(monthly, x="month", y="total_value", markers=True, title="Monthly Transaction Value"), width="stretch")
    col2.plotly_chart(px.bar(monthly, x="month", y="transaction_count", title="Monthly Transaction Count"), width="stretch")
    st.plotly_chart(px.histogram(df, x="gross_revenue", nbins=50, title="Distribution of Transaction Amounts"), width="stretch")
    st.plotly_chart(quarter_share_heatmap(df), width="stretch")

with tab2:
    st.subheader("Country Analysis")
    summary = country_summary(df)
    col1, col2 = st.columns(2)
    col1.plotly_chart(pareto_chart(summary), width="stretch")
    col2.plotly_chart(px.bar(summary.head(20), x="Country", y="transaction_count", title="Top Countries by Transaction Count"), width="stretch")
    table(summary)

with tab3:
    st.subheader("Time Trends")
    grain = st.radio("Select time grouping", ["day", "week", "month"], horizontal=True)
    metric = st.selectbox("Metric", ["total_value", "transaction_count"])
    trend = df.groupby([grain, "Country"], as_index=False).agg(total_value=("gross_revenue", "sum"), transaction_count=("gross_revenue", "count"))
    st.plotly_chart(px.line(trend, x=grain, y=metric, color="Country", markers=True, title=f"{metric.replace('_', ' ').title()} by {grain.title()} and Country"), width="stretch")
    heatmap = df.groupby(["Country", "month"], as_index=False).agg(total_value=("gross_revenue", "sum"))
    st.plotly_chart(px.density_heatmap(heatmap, x="month", y="Country", z="total_value", title="Transaction Value Heatmap by Country and Month"), width="stretch")

with tab4:
    st.subheader("Outlier Analysis")
    outliers = outliers.copy()
    both = outliers["is_transaction_count_outlier"] & outliers["is_quantity_outlier"]
    outliers["outlier_reason"] = np.select(
        [both, outliers["is_transaction_count_outlier"], outliers["is_quantity_outlier"]],
        ["Transaction Count + Quantity", "Transaction Count", "Quantity"],
        default="Other",
    )
    baseline_count = len(df) + len(outliers)
    metric_row([
        ("Detected Outliers", number(len(outliers))),
        ("Outlier Rate", f"{len(outliers) / baseline_count * 100 if baseline_count else 0:.1f}%"),
        ("Outlier Value", money(outliers["gross_revenue"].sum())),
        ("Largest Outlier", money(outliers["gross_revenue"].max() if len(outliers) else 0)),
    ])

    outlier_country = outliers.groupby("Country", as_index=False).agg(outlier_count=("gross_revenue", "count"), outlier_value=("gross_revenue", "sum"), avg_outlier_value=("gross_revenue", "mean")).sort_values("outlier_value", ascending=False)
    outlier_reason = outliers.groupby("outlier_reason", as_index=False).agg(outlier_count=("gross_revenue", "count"), outlier_value=("gross_revenue", "sum")).sort_values("outlier_count", ascending=False)
    distribution = pd.concat([df.assign(transaction_type="Retained"), outliers.assign(transaction_type="Removed Outlier")], ignore_index=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.scatter(outliers.sort_values("Date"), x="Date", y="gross_revenue", color="outlier_reason", hover_data=["TransactionNo", "Country", "ProductName", "Quantity"], title="Removed Outliers Over Time"), width="stretch")
    col2.plotly_chart(px.histogram(distribution, x="gross_revenue", color="transaction_type", barmode="overlay", nbins=40, opacity=0.7, title="Gross Revenue Distribution: Typical vs Outlier"), width="stretch")

    col3, col4 = st.columns(2)
    col3.plotly_chart(px.bar(outlier_country.head(15), x="Country", y="outlier_count", hover_data=["outlier_value", "avg_outlier_value"], title="Countries With The Most Removed Outliers"), width="stretch")
    col4.plotly_chart(px.bar(outlier_reason, x="outlier_reason", y="outlier_count", hover_data=["outlier_value"], title="Removed Outliers by Detection Rule"), width="stretch")

    st.plotly_chart(px.box(distribution, x="Country", y="gross_revenue", color="transaction_type", points="outliers", title="Retained vs Removed Transaction Amounts by Country"), width="stretch")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Outlier Summary by Country")
        table(outlier_country)
    with col6:
        st.subheader("Flagged Transactions")
        table(outliers.sort_values("gross_revenue", ascending=False))

with tab5:
    st.subheader("Data Cleaning KPIs")
    metric_row([
        ("Rows Removed for Missing Values", number(cleaning_kpis["missing_removed"])),
        ("Rows Removed for Non-Positive Values", number(cleaning_kpis["non_positive_removed"])),
        ("Rows Removed as Outliers", number(cleaning_kpis["outliers_removed"])),
    ])
    st.subheader("Raw Filtered Data")
    table(df)

with tab6:
    st.subheader("Customer Analysis")
    customers, buyer_summary, lifetime_by_country = customer_metrics(df)
    repeat_rate = (customers["orders"].gt(1).mean() * 100) if len(customers) else 0
    metric_row([
        ("One-time Buyers", number((customers["orders"] == 1).sum())),
        ("Repeat Buyers", number((customers["orders"] > 1).sum())),
        ("Repeat Buyer Rate", f"{repeat_rate:.2f}%"),
        ("Avg Observed Lifetime", f"{customers['lifetime_days'].mean():,.2f} days"),
    ])

    monthly_mix = monthly_customer_mix(df, full_df)
    totals = monthly_mix.groupby("month", as_index=False)[["new_customers", "returning_customers"]].sum()
    st.plotly_chart(px.line(totals, x="month", y=["new_customers", "returning_customers"], markers=True, title="Monthly New vs Returning Customers"), width="stretch")

    countries = sorted(monthly_mix["Country"].dropna().unique())
    selected = st.multiselect("Countries", countries, default=countries[:1], key="customer_country_filter")
    country_mix = monthly_mix[monthly_mix["Country"].isin(selected)].melt(
        id_vars=["month", "Country"],
        value_vars=["new_customers", "returning_customers"],
        var_name="customer_type",
        value_name="customers",
    )
    st.plotly_chart(px.line(country_mix, x="month", y="customers", color="customer_type", line_dash="Country", markers=True, title="Monthly New vs Returning Customers by Country"), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Buyer Summary")
        table(buyer_summary)
    with col2:
        st.subheader("Customer Lifetime by Country")
        table(lifetime_by_country)
