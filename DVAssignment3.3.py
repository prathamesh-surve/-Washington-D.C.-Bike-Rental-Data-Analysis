import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Bike Rental Dashboard", layout="wide")

st.title("🚲 Bike Rental Data Analysis Dashboard")

# ---------------------------
# 1. Load Dataset (AUTO)
# ---------------------------

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

if not os.path.exists("train.csv"):
    st.error("❌ train.csv file not found in project folder!")
    st.stop()

data = load_data()

st.success("Dataset loaded successfully!")

st.write("### Dataset Preview")
st.dataframe(data.head())

# ---------------------------
# 2. Sidebar Filters
# ---------------------------

st.sidebar.header("🔧 Filters")

column_select = st.sidebar.selectbox("Select main filter column", data.columns)

filtered_data = data.copy()

if pd.api.types.is_numeric_dtype(data[column_select]):
    min_val = float(data[column_select].min())
    max_val = float(data[column_select].max())
    range_slider = st.sidebar.slider(
        f"Filter {column_select}", min_val, max_val, (min_val, max_val)
    )
    filtered_data = filtered_data[
        (filtered_data[column_select] >= range_slider[0]) &
        (filtered_data[column_select] <= range_slider[1])
    ]

elif "date" in column_select.lower():
    filtered_data[column_select] = pd.to_datetime(filtered_data[column_select], errors="coerce")
    min_date = filtered_data[column_select].min()
    max_date = filtered_data[column_select].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    filtered_data = filtered_data[
        (filtered_data[column_select] >= pd.to_datetime(date_range[0])) &
        (filtered_data[column_select] <= pd.to_datetime(date_range[1]))
    ]

else:
    categories = filtered_data[column_select].dropna().unique()
    selected = st.sidebar.multiselect("Select categories", categories, default=categories)
    filtered_data = filtered_data[filtered_data[column_select].isin(selected)]

# ---------------------------
# 3. KPIs
# ---------------------------

st.subheader("📊 Key Metrics")

numeric_cols = filtered_data.select_dtypes(include="number").columns.tolist()

c1, c2, c3 = st.columns(3)

c1.metric("Rows", len(filtered_data))

if numeric_cols:
    c2.metric(f"Avg {numeric_cols[0]}", round(filtered_data[numeric_cols[0]].mean(), 2))
    c3.metric(f"Max {numeric_cols[0]}", round(filtered_data[numeric_cols[0]].max(), 2))

# ---------------------------
# 4. Visualizations
# ---------------------------

st.subheader("📈 Visualizations")

colA, colB = st.columns(2)

# Histogram
if numeric_cols:
    with colA:
        col_hist = st.selectbox("Histogram Column", numeric_cols)
        fig = px.histogram(filtered_data, x=col_hist, nbins=30, title="Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Scatter
if len(numeric_cols) >= 2:
    with colB:
        x = st.selectbox("Scatter X", numeric_cols, index=0)
        y = st.selectbox("Scatter Y", numeric_cols, index=1)
        fig2 = px.scatter(filtered_data, x=x, y=y, title="Scatter Plot")
        st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
if len(numeric_cols) >= 2:
    st.subheader("🔥 Correlation Heatmap")
    fig3, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# Bar chart
cat_cols = filtered_data.select_dtypes(include="object").columns.tolist()
if cat_cols:
    cat = st.selectbox("Category for Bar Chart", cat_cols)
    fig4 = px.bar(filtered_data[cat].value_counts(), title="Category Counts")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# 5. Summary
# ---------------------------

st.subheader("📄 Summary Statistics")
st.dataframe(filtered_data.describe(include="all"))
