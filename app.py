# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Load & Preprocess Data
# ------------------------------
@st.cache_data
def load_data(path="ecommerce_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["order_date"])
    df.columns = df.columns.str.strip().str.lower()
    
    # Ensure numeric
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0)
    
    # Derived columns
    df["sales"] = df["quantity"] * df["price"] * (1 - df["discount"])
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["order_day"] = df["order_date"].dt.date
    df["order_week"] = df["order_date"].dt.to_period("W").dt.start_time
    return df

df = load_data("ecommerce_dataset.csv")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")
st.title("📊 E-commerce Data Analysis Dashboard")

st.markdown("""
This dashboard provides a **full exploratory data analysis (EDA)** on e-commerce sales data.  
Use the **sidebar filters** to explore the dataset.
""")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")
categories = st.sidebar.multiselect("Filter by Category", df["category"].unique(), default=df["category"].unique())
regions = st.sidebar.multiselect("Filter by Region", df["region"].unique(), default=df["region"].unique())
payments = st.sidebar.multiselect("Filter by Payment Method", df["payment_method"].unique(), default=df["payment_method"].unique())
date_range = st.sidebar.date_input("Filter by Date Range", [df["order_date"].min().date(), df["order_date"].max().date()])

analysis_section = st.sidebar.radio(
    "Choose Analysis Section",
    [
        "Descriptive Analysis",
        "KPIs",
        "Revenue Heatmap",
        "Time Series Analysis",
        "Top Customers",
        "Category Summary",
        "Payment Methods",
        "Customer Retention (Cohorts)",
        "Recommendations"
    ]
)

# Apply filters
df_filtered = df[
    (df["category"].isin(categories)) &
    (df["region"].isin(regions)) &
    (df["payment_method"].isin(payments)) &
    (df["order_date"].dt.date >= date_range[0]) &
    (df["order_date"].dt.date <= date_range[1])
]

# ------------------------------
# Descriptive Analysis
# ------------------------------
if analysis_section == "Descriptive Analysis":
    st.subheader("📑 Descriptive Statistics & Overview")
    st.write("**Dataset Shape:**", df_filtered.shape)
    st.write("**Columns:**", list(df_filtered.columns))
    st.write("**Missing Values:**")
    st.dataframe(df_filtered.isna().sum())
    
    st.write("**Summary Statistics (numeric features):**")
    st.dataframe(df_filtered.describe().T)

    st.write("**Sample Data (first 10 rows):**")
    st.dataframe(df_filtered.head(10))

# ------------------------------
# KPIs
# ------------------------------
elif analysis_section == "KPIs":
    st.subheader("📌 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_orders = df_filtered["order_id"].nunique()
    total_revenue = df_filtered["sales"].sum()
    aov = df_filtered.groupby("order_id")["sales"].sum().mean()
    unique_customers = df_filtered["customer_id"].nunique()
    repeat_rate = (df_filtered.groupby("customer_id")["order_id"].nunique() > 1).sum() / max(unique_customers,1)

    col1.metric("Total Orders", total_orders)
    col2.metric("Total Revenue", f"${total_revenue:,.2f}")
    col3.metric("Average Order Value", f"${aov:,.2f}")
    col4.metric("Repeat Customer Rate", f"{repeat_rate:.2%}")

# ------------------------------
# Revenue Heatmap
# ------------------------------
elif analysis_section == "Revenue Heatmap":
    st.subheader("🔥 Revenue Heatmap — Region × Category")
    pivot = df_filtered.pivot_table(values="sales", index="region", columns="category", aggfunc="sum", fill_value=0)
    fig_heatmap = px.imshow(pivot, labels=dict(x="Category", y="Region", color="Revenue"), text_auto=True)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------------------
# Time Series Analysis
# ------------------------------
elif analysis_section == "Time Series Analysis":
    st.subheader("📈 Time Series Analysis")
    
    # Daily trend
    daily = df_filtered.groupby("order_day")["sales"].sum()
    fig_daily = px.line(daily, x=daily.index, y=daily.values, title="Daily Revenue Trend")
    st.plotly_chart(fig_daily, use_container_width=True)

    # Weekly trend
    weekly = df_filtered.groupby("order_week")["sales"].sum()
    fig_weekly = px.line(weekly, x=weekly.index, y=weekly.values, title="Weekly Revenue Trend")
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Monthly trend + rolling avg
    monthly = df_filtered.set_index("order_date").resample("M")["sales"].sum()
    monthly_rolling = monthly.rolling(3, min_periods=1).mean()
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(x=monthly.index, y=monthly.values, name="Monthly Revenue"))
    fig_monthly.add_trace(go.Line(x=monthly.index, y=monthly_rolling, name="3-Month Rolling Avg", line=dict(width=3)))
    fig_monthly.update_layout(title="Monthly Revenue with Rolling Average")
    st.plotly_chart(fig_monthly, use_container_width=True)

# ------------------------------
# Top Customers
# ------------------------------
elif analysis_section == "Top Customers":
    st.subheader("🧾 Top Customers by Revenue")
    cust_rev = df_filtered.groupby("customer_id")["sales"].sum().sort_values(ascending=False).head(20)
    fig_cust = px.bar(cust_rev, x=cust_rev.index.astype(str), y=cust_rev.values, labels={"x":"Customer ID","y":"Revenue"})
    st.plotly_chart(fig_cust, use_container_width=True)

# ------------------------------
# Category Summary
# ------------------------------
elif analysis_section == "Category Summary":
    st.subheader("📦 Category Performance")
    cat_summary = df_filtered.groupby("category").agg(
        total_revenue=("sales","sum"),
        avg_price=("price","mean"),
        avg_discount=("discount","mean"),
        total_qty=("quantity","sum"),
        orders=("order_id","nunique")
    ).sort_values("total_revenue", ascending=False).reset_index()
    st.dataframe(cat_summary)

# ------------------------------
# Payment Methods
# ------------------------------
elif analysis_section == "Payment Methods":
    st.subheader("💳 Payment Method Analysis")
    fig_payment = px.pie(df_filtered, names="payment_method", values="sales", title="Revenue Share by Payment Method")
    st.plotly_chart(fig_payment, use_container_width=True)

# ------------------------------
# Customer Retention (Cohorts)
# ------------------------------
elif analysis_section == "Customer Retention (Cohorts)":
    st.subheader("🔁 Cohort Retention Analysis")
    first_purchase = df_filtered.groupby("customer_id")["order_date"].min().dt.to_period("M").dt.to_timestamp().rename("first_order_month")
    cust = df_filtered[["customer_id","order_id","order_month"]].drop_duplicates().merge(first_purchase, left_on="customer_id", right_index=True)
    cust["months_since_first"] = ((cust["order_month"] - cust["first_order_month"]) / np.timedelta64(1, "M")).round().astype(int)
    cohort = cust.pivot_table(index="first_order_month", columns="months_since_first", values="customer_id", aggfunc="nunique").fillna(0).astype(int)
    st.dataframe(cohort)

# ------------------------------
# Recommendations
# ------------------------------
elif analysis_section == "Recommendations":
    st.subheader("✅ Recommendations")
    st.write("- Optimize marketing in regions where certain categories underperform (see heatmap).")
    st.write("- Address steep cohort drop-offs with loyalty campaigns.")
    st.write("- Focus on categories with high quantity but low revenue (pricing issues).")
    st.write("- Promote profitable payment methods that yield higher AOV.")
