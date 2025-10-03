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

st.markdown("### Using the sidebar to filter data and explore key insights.")

# ------------------------------
# Sidebar (short version)
# ------------------------------
st.sidebar.header("🔍 Filters")
categories = st.sidebar.multiselect("Category", df["category"].unique(), default=df["category"].unique())
regions = st.sidebar.multiselect("Region", df["region"].unique(), default=df["region"].unique())
payments = st.sidebar.multiselect("Payment Method", df["payment_method"].unique(), default=df["payment_method"].unique())
date_range = st.sidebar.date_input("Date Range", [df["order_date"].min().date(), df["order_date"].max().date()])

# Apply filters
df_filtered = df[
    (df["category"].isin(categories)) &
    (df["region"].isin(regions)) &
    (df["payment_method"].isin(payments)) &
    (df["order_date"].dt.date >= date_range[0]) &
    (df["order_date"].dt.date <= date_range[1])
]

# ------------------------------
# 1) Descriptive Analysis
# ------------------------------
st.header("📑 Descriptive Analysis")
st.write("**Dataset Shape:**", df_filtered.shape)
st.write("**Missing Values:**")
st.dataframe(df_filtered.isna().sum())
st.write("**Summary Statistics:**")
st.dataframe(df_filtered.describe().T)

# ------------------------------
# 2) KPIs
# ------------------------------
st.header("📌 Key KPIs")
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
# 3) Time Series Analysis
# ------------------------------
st.header("📈 Time Series Analysis")

# Daily trend
daily = df_filtered.groupby("order_day")["sales"].sum()
fig_daily = px.line(
    x=daily.index, y=daily.values, 
    title="Daily Revenue Trend",
    labels={"x":"Date","y":"Revenue"},
    color_discrete_sequence=["#1f77b4"]
)
st.plotly_chart(fig_daily, use_container_width=True)

# Weekly trend
weekly = df_filtered.groupby("order_week")["sales"].sum()
fig_weekly = px.line(
    x=weekly.index, y=weekly.values, 
    title="Weekly Revenue Trend",
    labels={"x":"Week","y":"Revenue"},
    color_discrete_sequence=["#ff7f0e"]
)
st.plotly_chart(fig_weekly, use_container_width=True)

# Monthly trend
monthly = df_filtered.set_index("order_date").resample("M")["sales"].sum()
monthly_rolling = monthly.rolling(3, min_periods=1).mean()
fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(x=monthly.index, y=monthly.values, name="Monthly Revenue", marker_color="#2ca02c"))
fig_monthly.add_trace(go.Scatter(x=monthly.index, y=monthly_rolling, name="3-Month Avg", line=dict(color="#d62728", width=3)))
fig_monthly.update_layout(title="Monthly Revenue with Rolling Average")
st.plotly_chart(fig_monthly, use_container_width=True)

# ------------------------------
# 4) Revenue Heatmap
# ------------------------------
st.header("🔥 Revenue Heatmap (Region × Category)")
pivot = df_filtered.pivot_table(values="sales", index="region", columns="category", aggfunc="sum", fill_value=0)
fig_heatmap = px.imshow(
    pivot, 
    labels=dict(x="Category", y="Region", color="Revenue"), 
    color_continuous_scale="Viridis", 
    text_auto=True
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------------------
# 5) Top Customers
# ------------------------------
st.header("🧾 Top 20 Customers by Revenue")
cust_rev = df_filtered.groupby("customer_id")["sales"].sum().sort_values(ascending=False).head(20)
fig_cust = px.bar(
    x=cust_rev.index.astype(str), 
    y=cust_rev.values, 
    labels={"x":"Customer ID","y":"Revenue"},
    color=cust_rev.values,
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_cust, use_container_width=True)

# ------------------------------
# 6) Category Summary
# ------------------------------
st.header("📦 Category Performance Summary")
cat_summary = df_filtered.groupby("category").agg(
    total_revenue=("sales","sum"),
    avg_price=("price","mean"),
    avg_discount=("discount","mean"),
    total_qty=("quantity","sum"),
    orders=("order_id","nunique")
).sort_values("total_revenue", ascending=False).reset_index()
st.dataframe(cat_summary)

# ------------------------------
# 7) Payment Methods
# ------------------------------
st.header("💳 Payment Method Analysis")
fig_payment = px.pie(
    df_filtered, 
    names="payment_method", 
    values="sales", 
    title="Revenue Share by Payment Method",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_payment, use_container_width=True)

# ------------------------------
# ------------------------------
# ------------------------------
# 8) Customer Retention (Cohorts)
# ------------------------------
st.header("🔁 Customer Retention (Cohort View)")

# First purchase month for each customer
first_purchase = (
    df_filtered.groupby("customer_id")["order_date"]
    .min()
    .dt.to_period("M")
    .dt.to_timestamp()
    .rename("first_order_month")
)

cust = (
    df_filtered[["customer_id", "order_id", "order_month"]]
    .drop_duplicates()
    .merge(first_purchase, left_on="customer_id", right_index=True)
)

# Calculate difference in months between first and subsequent orders
cust["months_since_first"] = (
    (cust["order_month"].dt.year - cust["first_order_month"].dt.year) * 12
    + (cust["order_month"].dt.month - cust["first_order_month"].dt.month)
)

# Cohort pivot table
cohort = (
    cust.pivot_table(
        index="first_order_month",
        columns="months_since_first",
        values="customer_id",
        aggfunc="nunique"
    )
    .fillna(0)
    .astype(int)
)

st.dataframe(cohort)



# ------------------------------
# 9) Recommendations
# ------------------------------
st.header("✅ Recommendations")
st.write("- Focus marketing on high-revenue categories in underperforming regions.")
st.write("- Strengthen retention programs where cohort drop-offs are steep.")
st.write("- Review categories with high volume but low revenue contribution.")
st.write("- Incentivize preferred payment methods to increase AOV.")
