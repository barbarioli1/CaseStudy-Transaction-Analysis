import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Transaction Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Transaction Analytics Dashboard")

import pandas as pd
import numpy as np



def plot_country_quarter_revenue_heatmap(df, width=1200, base_height=500):
    """
    Plot each quarter's share of total country revenue as a Plotly heat map.
    """
    data = df.copy()

    # Ensure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # If gross_revenue is missing, create it
    if "gross_revenue" not in data.columns:
        data["gross_revenue"] = data["Price"] * data["Quantity"]

    # Quarter period
    data["quarter_period"] = data["Date"].dt.to_period("Q")

    # Aggregate quarterly revenue by country
    quarter_rev = (
        data.groupby(["Country", "quarter_period"], as_index=False)["gross_revenue"]
        .sum()
    )

    if quarter_rev.empty:
        raise ValueError("No data available to plot after aggregation.")

    # Full quarter range across the dataset
    all_quarters = pd.period_range(
        start=quarter_rev["quarter_period"].min(),
        end=quarter_rev["quarter_period"].max(),
        freq="Q"
    )

    quarter_labels = [f"Q{q.quarter}-{q.year}" for q in all_quarters]
    # Countries
    countries = sorted(quarter_rev["Country"].dropna().unique())

    # Pivot revenue table
    revenue_pivot = (
        quarter_rev.pivot(index="Country", columns="quarter_period", values="gross_revenue")
        .reindex(index=countries, columns=all_quarters)
    )

    country_total_revenue = revenue_pivot.sum(axis=1, skipna=True)
    share_pivot = revenue_pivot.div(country_total_revenue.replace(0, np.nan), axis=0) * 100

    display_countries = list(reversed(countries))
    z_values = share_pivot.reindex(index=display_countries, columns=all_quarters).to_numpy(dtype=float)
    quarter_revenue_values = revenue_pivot.reindex(index=display_countries, columns=all_quarters).to_numpy(dtype=float)
    country_total_values = country_total_revenue.reindex(display_countries).to_numpy(dtype=float)
    country_total_grid = np.repeat(country_total_values[:, None], len(all_quarters), axis=1)
    max_share = float(np.nanmax(z_values)) if np.isfinite(z_values).any() else 0.0

    quarter_revenue_labels = np.where(
        np.isnan(quarter_revenue_values),
        "No revenue",
        np.vectorize(lambda value: f"${value:,.2f}")(quarter_revenue_values),
    )
    country_total_labels = np.where(
        np.isnan(country_total_grid),
        "No revenue",
        np.vectorize(lambda value: f"${value:,.2f}")(country_total_grid),
    )
    share_labels = np.where(
        np.isnan(z_values),
        "No revenue",
        np.vectorize(lambda value: f"{value:.1f}%")(z_values),
    )
    hover_data = np.stack(
        [quarter_revenue_labels, country_total_labels, share_labels],
        axis=-1,
    )

    row_height = 12
    plot_height = max(base_height, row_height * len(countries) + 120)

    fig = go.Figure(
        data=go.Heatmap(
            x=quarter_labels,
            y=display_countries,
            z=z_values,
            customdata=hover_data,
            colorscale=[
                [0.0, "#f6f9fd"],
                [0.25, "#c1dcf7"],
                [0.75, "#41a1d8"],
                [0.90, "#1269b6"],
                [1.0, "#08306b"],
            ],
            colorbar={"title": "% of Country Revenue"},
            hoverongaps=False,
            hovertemplate=(
                "Country: %{y}<br>"
                "Quarter: %{x}<br>"
                "Quarter revenue: %{customdata[0]}<br>"
                "Country total: %{customdata[1]}<br>"
                "Share of country revenue: %{customdata[2]}"
                "<extra></extra>"
            ),
            ygap=1,
            zmin=0,
            zmax=max_share if max_share > 0 else 1,
        )
    )

    fig.update_layout(
        title="Quarter Share of Total Country Revenue",
        width=width,
        height=plot_height,
        xaxis_title="Quarter",
        yaxis_title="Country",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    fig.update_xaxes(
        side="bottom",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        gridwidth=1,
    )

    return fig
# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)
def remove_transaction_count_outliers(
    df,
    country_col,
    customer_id_col,
    quantity_col="Quantity",
    method="iqr",
    iqr_multiplier=1.5,
    percentile_cutoff=0.99,
):
    df = df.copy()

    group_cols = [customer_id_col, "day", country_col]

    # Count transactions for same customer, same day, same country
    transaction_counts = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="transactions_same_customer_day_country")
    )

    def calc_upper_bound(series):
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            return q3 + iqr_multiplier * iqr
        elif method == "percentile":
            return series.quantile(percentile_cutoff)
        else:
            raise ValueError("method must be either 'iqr' or 'percentile'")

    transaction_upper_bound = calc_upper_bound(
        transaction_counts["transactions_same_customer_day_country"]
    )
    quantity_upper_bound = calc_upper_bound(df[quantity_col].dropna())

    transaction_counts["is_transaction_count_outlier"] = (
        transaction_counts["transactions_same_customer_day_country"] > transaction_upper_bound
    )

    # Add the count back to original transaction rows
    df_with_counts = df.merge(transaction_counts, on=group_cols, how="left")

    # Quantity outlier at the row level
    df_with_counts["is_quantity_outlier"] = df_with_counts[quantity_col] > quantity_upper_bound

    outlier_mask = (
        df_with_counts["is_transaction_count_outlier"]
        | df_with_counts["is_quantity_outlier"]
    )

    outlier_rows = df_with_counts[outlier_mask].copy()
    cleaned_df = df_with_counts[~outlier_mask].copy()

    return cleaned_df, outlier_rows, {
        "transaction_count_upper_bound": transaction_upper_bound,
        "quantity_upper_bound": quantity_upper_bound,
    }


@st.cache_data
def clean_n_enrich_data(df):
    column_types = {
    "TransactionNo": "string",
    "Date": "string",
    "ProductNo": "string",
    "ProductName": "string",
    "Price": "float64",
    "Quantity": "Int64",
    "CustomerNo": "Int64",
    "Country": "string"
    }
    date_columns = list(["Date"])
    original_df=df.copy()
    df = df.dropna()

    for col, dtype in column_types.items():
        if dtype  in ["Int64", "int64", "float64"]: 
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype(dtype)
    df = df.dropna()

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    na_kpi_count=original_df.shape[0]-df.shape[0]
    for col in df.columns:
        if df[col].dtype == "object" and col not in ["Date", "Country",]:
            df[col] = df[col].str.strip()
    mask = df[["Price", "Quantity"]].gt(0).all(axis=1)
    df = df[mask].copy()
    below_or_0_kpi_count = df[~mask].shape[0]

    df["year"] = df["Date"].dt.year
    df["month_number"] = df["Date"].dt.month
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    df["week"] = (df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    df["day"] = df["Date"].dt.date
    df["weekday"] = df["Date"].dt.day_name()
    df['gross_revenue']=df['Price']*df['Quantity']

    cleaned_df, outlier_rows, _ = remove_transaction_count_outliers(
    df=df,
    country_col="Country",
    customer_id_col="CustomerNo",
    method="iqr",
    iqr_multiplier=5
    )
    df=cleaned_df.copy()
    outliers_removed_kpi_count = outlier_rows.shape[0]
    df["quarter"] = "Q" + (((df["month_number"] - 1) // 3) + 1).astype(str)
    df["year_quarter"] = df["year"].astype(str) + df["quarter"]
    kpis = {
        "na_kpi_count": na_kpi_count,
        "below_or_0_kpi_count": below_or_0_kpi_count,
        "outliers_removed_kpi_count": outliers_removed_kpi_count
    }
    return df, outlier_rows, kpis


def format_number(value):
    return f"{value:,.0f}"


def format_money(value):
    return f"${value:,.2f}"


@st.cache_data
def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def render_table(df, selectable=False, page_size=100, pagination_threshold=2000):
    if len(df) <= pagination_threshold:
        st.dataframe(df, width="stretch")
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
    )

    if selectable:
        gb.configure_selection("multiple", use_checkbox=True)

    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=page_size)
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
    )


def build_customer_analysis_tables(source_df):
    customer_df = source_df.copy()
    customer_df["Date"] = pd.to_datetime(customer_df["Date"], errors="coerce")
    customer_df = customer_df.dropna(subset=["CustomerNo", "TransactionNo", "Country", "Date"]).copy()
    customer_df["order_date"] = customer_df["Date"].dt.normalize()

    if "gross_revenue" not in customer_df.columns:
        customer_df["gross_revenue"] = customer_df["Price"] * customer_df["Quantity"]

    customer_kpis = (
        customer_df.groupby(["CustomerNo", "Country"], as_index=False)
        .agg(
            first_purchase=("order_date", "min"),
            last_purchase=("order_date", "max"),
            order_count=("TransactionNo", "nunique"),
            total_revenue=("gross_revenue", "sum"),
            total_units=("Quantity", "sum"),
        )
    )

    customer_kpis["buyer_type"] = np.where(
        customer_kpis["order_count"] == 1,
        "One-time buyer",
        "Repeat buyer",
    )

    buyer_summary = (
        customer_kpis.groupby("buyer_type", as_index=False)
        .agg(
            customers=("CustomerNo", "nunique"),
            total_revenue=("total_revenue", "sum"),
            avg_revenue_per_customer=("total_revenue", "mean"),
            avg_orders_per_customer=("order_count", "mean"),
        )
    )

    buyer_summary["customer_share_pct"] = (
        buyer_summary["customers"] / buyer_summary["customers"].sum() * 100
    ).round(2)
    buyer_summary["total_revenue"] = buyer_summary["total_revenue"].round(2)
    buyer_summary["avg_revenue_per_customer"] = buyer_summary["avg_revenue_per_customer"].round(2)
    buyer_summary["avg_orders_per_customer"] = buyer_summary["avg_orders_per_customer"].round(2)

    buyer_summary_by_country = (
        customer_kpis.groupby(["Country", "buyer_type"], as_index=False)
        .agg(
            customers=("CustomerNo", "nunique"),
            total_revenue=("total_revenue", "sum"),
            avg_revenue_per_customer=("total_revenue", "mean"),
        )
    )

    country_totals = buyer_summary_by_country.groupby("Country")["customers"].transform("sum")
    buyer_summary_by_country["customer_share_pct"] = (
        buyer_summary_by_country["customers"] / country_totals * 100
    ).round(2)
    buyer_summary_by_country["total_revenue"] = buyer_summary_by_country["total_revenue"].round(2)
    buyer_summary_by_country["avg_revenue_per_customer"] = buyer_summary_by_country["avg_revenue_per_customer"].round(2)

    customer_kpis["customer_lifetime_days"] = (
        customer_kpis["last_purchase"] - customer_kpis["first_purchase"]
    ).dt.days
    customer_kpis["avg_revenue_per_order"] = (
        customer_kpis["total_revenue"] / customer_kpis["order_count"]
    ).round(2)

    customer_lifetime_summary = pd.DataFrame(
        {
            "metric": [
                "Customers",
                "Average observed lifetime (days)",
                "Median observed lifetime (days)",
                "Average orders per customer",
                "Average customer lifetime value",
                "Median customer lifetime value",
                "Average revenue per order",
            ],
            "value": [
                customer_kpis["CustomerNo"].nunique(),
                round(customer_kpis["customer_lifetime_days"].mean(), 2),
                round(customer_kpis["customer_lifetime_days"].median(), 2),
                round(customer_kpis["order_count"].mean(), 2),
                round(customer_kpis["total_revenue"].mean(), 2),
                round(customer_kpis["total_revenue"].median(), 2),
                round(customer_kpis["avg_revenue_per_order"].mean(), 2),
            ],
        }
    )

    customer_lifetime_by_country = (
        customer_kpis.groupby("Country", as_index=False)
        .agg(
            customers=("CustomerNo", "nunique"),
            avg_lifetime_days=("customer_lifetime_days", "mean"),
            median_lifetime_days=("customer_lifetime_days", "median"),
            avg_orders_per_customer=("order_count", "mean"),
            avg_customer_lifetime_value=("total_revenue", "mean"),
        )
    )

    for column in [
        "avg_lifetime_days",
        "median_lifetime_days",
        "avg_orders_per_customer",
        "avg_customer_lifetime_value",
    ]:
        customer_lifetime_by_country[column] = customer_lifetime_by_country[column].round(2)

    repeat_buyer_rate = round((customer_kpis["order_count"] > 1).mean() * 100, 2)

    return {
        "customer_kpis": customer_kpis,
        "buyer_summary": buyer_summary,
        "buyer_summary_by_country": buyer_summary_by_country,
        "customer_lifetime_summary": customer_lifetime_summary,
        "customer_lifetime_by_country": customer_lifetime_by_country,
        "repeat_buyer_rate": repeat_buyer_rate,
    }


# -----------------------------
# File upload
# -----------------------------
df_raw = load_data("Sales_Transactions_Interview_Dataset.csv")


df, outlier_rows, kpis = clean_n_enrich_data(df_raw)
full_df = df.copy()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.subheader("Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

countries = sorted(df["Country"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Countries",
    countries,
    default=countries
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[
        (df["Date"].dt.date >= start_date) &
        (df["Date"].dt.date <= end_date)
    ]
    outlier_rows = outlier_rows[
        (outlier_rows["Date"].dt.date >= start_date) &
        (outlier_rows["Date"].dt.date <= end_date)
    ]

df = df[df["Country"].isin(selected_countries)]
outlier_rows = outlier_rows[outlier_rows["Country"].isin(selected_countries)]

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# -----------------------------
# KPI section
# -----------------------------
total_value = df['gross_revenue'].sum()
transaction_count = len(df)
avg_transaction_value = df['gross_revenue'].mean()
median_transaction_value = df['gross_revenue'].median()
country_count = df["Country"].nunique()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Value", format_money(total_value))
col2.metric("Transactions", format_number(transaction_count))
col3.metric("Avg Transaction", format_money(avg_transaction_value))
col4.metric("Median Transaction", format_money(median_transaction_value))
col5.metric("Countries", format_number(country_count))


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Country Analysis",
    "Time Trends",
    "Outliers",
    "Raw Data",
    "Customer Analysis"
])


# -----------------------------
# Overview
# -----------------------------
with tab1:
    quarterly_revenue = (
        df.groupby(["year", "quarter"], as_index=False)
        .agg(
            total_revenue=("gross_revenue", "sum"),
            avg_revenue=("gross_revenue", "mean"),
            transaction_count=("gross_revenue", "count")
        )
        .sort_values(["year", "quarter"])
    )
    st.subheader("Overview")

    col1, col2 = st.columns(2)

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            total_value=('gross_revenue', "sum"),
            transaction_count=('gross_revenue', "count")
        )
    )

    country_pareto = (
        df.groupby("Country", as_index=False)
        .agg(country_revenue=("gross_revenue", "sum"))
        .sort_values("country_revenue", ascending=False)
    )
    country_pareto["cumulative_revenue_pct"] = (
        country_pareto["country_revenue"].cumsum() / country_pareto["country_revenue"].sum() * 100
    )

    with col1:
        fig = px.line(
            monthly,
            x="month",
            y="total_value",
            markers=True,
            title="Monthly Transaction Value"
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            monthly,
            x="month",
            y="transaction_count",
            title="Monthly Transaction Count"
        )
        st.plotly_chart(fig, width="stretch")


    st.subheader("Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x='gross_revenue',
        nbins=50,
        title="Distribution of Transaction Amounts"
    )
    st.plotly_chart(fig, width="stretch")
    fig = plot_country_quarter_revenue_heatmap(df)
    st.subheader("Revenue Quarter Profile")
    st.plotly_chart(fig, width="stretch")




# -----------------------------
# Country analysis
# -----------------------------
with tab2:
    st.subheader("Country Analysis")

    country_summary = (
        df.groupby("Country", as_index=False)
        .agg(
            total_value=('gross_revenue', "sum"),
            transaction_count=('gross_revenue', "count"),
            average_transaction=('gross_revenue', "mean"),
            median_transaction=('gross_revenue', "median")
        )
        .sort_values("total_value", ascending=False)
    )
    country_summary["cumulative_revenue_pct"] = (
        country_summary["total_value"].cumsum() / country_summary["total_value"].sum() * 100
    )
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_bar(
            x=country_summary["Country"],
            y=country_summary["total_value"],
            name="Total Value"
        )
        fig.add_scatter(
        x=country_summary["Country"],
        y=country_summary["cumulative_revenue_pct"],
        name="Cumulative %",
        mode="lines+markers",
        yaxis="y2",
        line={"color": "#08306b", "width": 3},
        marker={"size": 7},
        hovertemplate="Country: %{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
        )
        fig.update_layout(
            title="Country Revenue Pareto Chart",
            xaxis_title="Country",
            yaxis_title="Revenue",
            yaxis2={
                "title": "Cumulative Revenue %",
                "overlaying": "y",
                "side": "right",
                "range": [0, 110],
                "ticksuffix": "%",
            },
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width="stretch")



    with col2:
        fig = px.bar(
            country_summary.head(20),
            x="Country",
            y="transaction_count",
            title="Top Countries by Transaction Count"
        )
        st.plotly_chart(fig, width="stretch")

    render_table(country_summary)


# -----------------------------
# Time trends
# -----------------------------
with tab3:
    st.subheader("Time Trends")

    time_grain = st.radio(
        "Select time grouping",
        ["day", "week", "month"],
        horizontal=True
    )

    trend = (
        df.groupby([time_grain, "Country"], as_index=False)
        .agg(
            total_value=('gross_revenue', "sum"),
            transaction_count=('gross_revenue', "count")
        )
        .sort_values(time_grain)
    )

    metric = st.selectbox(
        "Metric",
        ["total_value", "transaction_count"]
    )

    fig = px.line(
        trend,
        x=time_grain,
        y=metric,
        color="Country",
        markers=True,
        title=f"{metric.replace('_', ' ').title()} by {time_grain.title()} and Country"
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Country x Month Heatmap")

    heatmap_data = (
        df.groupby(["Country", "month"], as_index=False)
        .agg(total_value=('gross_revenue', "sum"))
    )

    fig = px.density_heatmap(
        heatmap_data,
        x="month",
        y="Country",
        z="total_value",
        title="Transaction Value Heatmap by Country and Month"
    )
    st.plotly_chart(fig, width="stretch")


# -----------------------------
# Outlier analysis
# -----------------------------
with tab4:
    st.subheader("Outlier Analysis")

    outliers = outlier_rows.copy()
    baseline_df = pd.concat([df, outliers], ignore_index=True)

    outliers["outlier_reason"] = outliers.apply(
        lambda row: ", ".join(
            reason for reason, is_flagged in {
                "Transaction Count": row.get("is_transaction_count_outlier", False),
                "Quantity": row.get("is_quantity_outlier", False),
            }.items() if is_flagged
        ) or "Other",
        axis=1,
    )

    outlier_country_summary = (
        outliers.groupby("Country", as_index=False)
        .agg(
            outlier_count=("gross_revenue", "count"),
            outlier_value=("gross_revenue", "sum"),
            avg_outlier_value=("gross_revenue", "mean")
        )
        .sort_values("outlier_value", ascending=False)
    )

    reason_summary = (
        outliers.groupby("outlier_reason", as_index=False)
        .agg(
            outlier_count=("gross_revenue", "count"),
            outlier_value=("gross_revenue", "sum")
        )
        .sort_values("outlier_count", ascending=False)
    )

    total_outlier_value = outliers["gross_revenue"].sum()
    outlier_share = (len(outliers) / len(baseline_df) * 100) if len(baseline_df) else 0
    avg_outlier_value = outliers["gross_revenue"].mean() if not outliers.empty else 0
    max_outlier_value = outliers["gross_revenue"].max() if not outliers.empty else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Detected Outliers", format_number(len(outliers)))
    kpi2.metric("Outlier Rate", f"{outlier_share:.1f}%")
    kpi3.metric("Outlier Value", format_money(total_outlier_value))
    kpi4.metric("Largest Outlier", format_money(max_outlier_value))

    st.caption(
        "Removed rows are sourced from transaction-count and quantity outlier detection in data cleaning. "
        f"Average outlier transaction: {format_money(avg_outlier_value)}"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.scatter(
            outliers.sort_values("Date"),
            x="Date",
            y="gross_revenue",
            color="outlier_reason",
            hover_data=["TransactionNo", "Country", "ProductName", "Quantity"],
            title="Removed Outliers Over Time"
        )
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        distribution_df = pd.concat([
            df.assign(transaction_type="Retained"),
            outliers.assign(transaction_type="Removed Outlier")
        ], ignore_index=True)
        fig = px.histogram(
            distribution_df,
            x="gross_revenue",
            color="transaction_type",
            barmode="overlay",
            nbins=40,
            opacity=0.7,
            title="Gross Revenue Distribution: Typical vs Outlier"
        )
        st.plotly_chart(fig, width="stretch")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig = px.bar(
            outlier_country_summary.head(15),
            x="Country",
            y="outlier_count",
            hover_data=["outlier_value", "avg_outlier_value"],
            title="Countries With The Most Removed Outliers"
        )
        st.plotly_chart(fig, width="stretch")

    with chart_col4:
        fig = px.bar(
            reason_summary,
            x="outlier_reason",
            y="outlier_count",
            hover_data=["outlier_value"],
            title="Removed Outliers by Detection Rule"
        )
        st.plotly_chart(fig, width="stretch")

    fig = px.box(
        distribution_df,
        x="Country",
        y="gross_revenue",
        color="transaction_type",
        points="outliers",
        title="Retained vs Removed Transaction Amounts by Country"
    )
    st.plotly_chart(fig, width="stretch")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.subheader("Outlier Summary by Country")
        render_table(outlier_country_summary)

    with summary_col2:
        st.subheader("Flagged Transactions")
        render_table(outliers.sort_values("gross_revenue", ascending=False))

with tab6:
    st.subheader("Customer Analysis")

    customer_metrics = build_customer_analysis_tables(df)
    buyer_summary = customer_metrics["buyer_summary"]
    buyer_summary_by_country = customer_metrics["buyer_summary_by_country"]
    customer_lifetime_summary = customer_metrics["customer_lifetime_summary"]
    customer_lifetime_by_country = customer_metrics["customer_lifetime_by_country"]
    repeat_buyer_rate = customer_metrics["repeat_buyer_rate"]

    one_time_buyers = int(
        buyer_summary.loc[
            buyer_summary["buyer_type"] == "One-time buyer",
            "customers",
        ].sum()
    )
    repeat_buyers = int(
        buyer_summary.loc[
            buyer_summary["buyer_type"] == "Repeat buyer",
            "customers",
        ].sum()
    )
    avg_observed_lifetime_days = float(
        customer_lifetime_summary.loc[
            customer_lifetime_summary["metric"] == "Average observed lifetime (days)",
            "value",
        ].iloc[0]
    )

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("One-time Buyers", format_number(one_time_buyers))
    kpi_col2.metric("Repeat Buyers", format_number(repeat_buyers))
    kpi_col3.metric("Repeat Buyer Rate", f"{repeat_buyer_rate:.2f}%")
    kpi_col4.metric("Avg Observed Lifetime", f"{avg_observed_lifetime_days:,.2f} days")

    customer_df = df.copy()
    customer_df['Date'] = pd.to_datetime(customer_df['Date'], errors='coerce')
    customer_df = customer_df.dropna(subset=['Date'])
    customer_df['month_period'] = customer_df['Date'].dt.to_period('M')

    customer_history_df = full_df.copy()
    customer_history_df['Date'] = pd.to_datetime(customer_history_df['Date'], errors='coerce')
    customer_history_df = customer_history_df.dropna(subset=['Date'])

    first_month = (
        customer_history_df.groupby(['CustomerNo', 'Country'], as_index=False)['Date']
        .min()
    )
    first_month['first_month'] = pd.to_datetime(first_month['Date']).dt.to_period('M')
    first_month = first_month.drop(columns=['Date'])

    customer_df = customer_df.merge(first_month, on=['CustomerNo', 'Country'], how='left')

    # Keep one row per customer-month-country so each customer counts once per month
    monthly_customers = customer_df.drop_duplicates(['CustomerNo', 'Country', 'month_period']).copy()

    # Flag whether the customer's first purchase month equals the current month (new customer)
    monthly_customers['is_new'] = monthly_customers['first_month'] == monthly_customers['month_period']

    # Aggregate counts per month + country
    monthly_summary = (
        monthly_customers.groupby(['month_period', 'Country'], as_index=False)
        .agg(
            total_customers=('CustomerNo', 'nunique'),
            new_customers=('is_new', 'sum')
        )
    )

    monthly_summary['old_customers'] = monthly_summary['total_customers'] - monthly_summary['new_customers']


    monthly_summary = monthly_summary.sort_values(['month_period', 'Country'])
    

    monthly_summary['month_start'] = monthly_summary['month_period'].dt.to_timestamp()
    agg_month = (
        monthly_summary.groupby('month_start', as_index=False)[['new_customers', 'old_customers']]
        .sum()
    )


    fig_customers = go.Figure()
    fig_customers.add_trace(
        go.Scatter(
            x=agg_month['month_start'],
            y=agg_month['new_customers'],
            mode='lines+markers',
            name='New Customers',
            line={'color': '#1f77b4'}
        )
    )
    fig_customers.add_trace(
        go.Scatter(
            x=agg_month['month_start'],
            y=agg_month['old_customers'],
            mode='lines+markers',
            name='Returning Customers',
            line={'color': '#ff7f0e'}
        )
    )
    fig_customers.update_layout(
        title='Monthly New vs Returning Customers (aggregated)',
        xaxis_title='Month',
        yaxis_title='Number of Customers',
        xaxis=dict(tickformat='%Y-%m')
    )

    st.plotly_chart(fig_customers, width="stretch")
    countries=sorted(monthly_summary['Country'].dropna().unique())
    selected_countries_ = st.multiselect(
    "Countries",
    countries,
    default=countries[0],
    key="customer_country_filter"
    )
    
    fig_customers_country = go.Figure()
    for country in selected_countries_:
        country_data = monthly_summary[monthly_summary['Country'] == country]
        fig_customers_country.add_trace(
            go.Scatter(
                x=country_data['month_start'],
                y=country_data['new_customers'],
                mode='lines+markers',
                name=f'New Customers - {country}',
                line={'color': '#1f77b4'}
            
        )
    )
    fig_customers_country.add_trace(
        go.Scatter(
            x=country_data['month_start'],
            y=country_data['old_customers'],
            mode='lines+markers',
            name=f'Returning Customers - {country}',
            line={'color': '#ff7f0e'}
        )
    )
    fig_customers_country.update_layout(
        title='Monthly New vs Returning Customers (aggregated)',
        xaxis_title='Month',
        yaxis_title='Number of Customers',
        xaxis=dict(tickformat='%Y-%m')
    )
    st.plotly_chart(fig_customers_country, width="stretch")
    
with tab5:
    st.subheader("Data Cleaning KPIs")

    raw_kpi_col1, raw_kpi_col2, raw_kpi_col3 = st.columns(3)
    raw_kpi_col1.metric("Rows Removed for Missing Values", format_number(kpis["na_kpi_count"]))
    raw_kpi_col2.metric("Rows Removed for Non-Positive Values", format_number(kpis["below_or_0_kpi_count"]))
    raw_kpi_col3.metric("Rows Removed as Outliers", format_number(kpis["outliers_removed_kpi_count"]))

    st.subheader("Raw Filtered Data")

    render_table(df, selectable=True)

