import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# 1. Load Dataset
# ---------------------------
st.title("Interactive Dashboard")
st.markdown("### Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Sample of the dataset")
    st.dataframe(data.head())

    # ---------------------------
    # 2. Select Column for Analysis
    # ---------------------------
    column_select = st.selectbox("Select column for main analysis", data.columns)

    # ---------------------------
    # 3. Filtering Based on Column Type
    # ---------------------------
    if pd.api.types.is_numeric_dtype(data[column_select]):
        min_val = float(data[column_select].min())
        max_val = float(data[column_select].max())
        range_slider = st.slider(f"Select range for {column_select}", min_val, max_val, (min_val, max_val))
        filtered_data = data[(data[column_select] >= range_slider[0]) &
                             (data[column_select] <= range_slider[1])]
    elif pd.api.types.is_datetime64_any_dtype(data[column_select]) or "date" in column_select.lower():
        data[column_select] = pd.to_datetime(data[column_select], errors='coerce')
        min_date = data[column_select].min()
        max_date = data[column_select].max()
        date_range = st.date_input("Select date range", [min_date, max_date])
        filtered_data = data[(data[column_select] >= pd.to_datetime(date_range[0])) &
                             (data[column_select] <= pd.to_datetime(date_range[1]))]
    else:
        categories = data[column_select].unique()
        selected_categories = st.multiselect("Select categories", categories, default=categories)
        filtered_data = data[data[column_select].isin(selected_categories)]

    # ---------------------------
    # 4. Additional Filtering
    # ---------------------------
    cat_columns = data.select_dtypes(include=['object']).columns.tolist()
    cat_columns = [col for col in cat_columns if col != column_select]
    if cat_columns:
        second_cat = st.selectbox("Filter another categorical column (optional)", ["None"] + cat_columns)
        if second_cat != "None":
            options = data[second_cat].unique()
            selected_options = st.multiselect(f"Select {second_cat}", options, default=options)
            filtered_data = filtered_data[filtered_data[second_cat].isin(selected_options)]

    # ---------------------------
    # 5. Plots
    # ---------------------------
    st.write("### Visualizations")

    numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()

    # Histogram (Seaborn)
    if numeric_cols:
        col1 = st.selectbox("Select numeric column for histogram", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(filtered_data[col1], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    # Scatter Plot (Plotly)
    if len(numeric_cols) >= 2:
        col_x = st.selectbox("X-axis for scatter plot", numeric_cols, index=0)
        col_y = st.selectbox("Y-axis for scatter plot", numeric_cols, index=1)
        fig2 = px.scatter(filtered_data, x=col_x, y=col_y, color=cat_cols[0] if cat_cols else None)
        st.plotly_chart(fig2)

    # Boxplot
    if numeric_cols and cat_cols:
        num_col = st.selectbox("Numeric column for boxplot", numeric_cols, key="box1")
        cat_col_box = st.selectbox("Category column for boxplot", cat_cols, key="box2")
        fig3, ax = plt.subplots()
        sns.boxplot(x=filtered_data[cat_col_box], y=filtered_data[num_col], ax=ax)
        st.pyplot(fig3)

    # Bar chart (Streamlit)
    if cat_cols:
        cat_col_bar = st.selectbox("Select categorical column for bar chart", cat_cols, key="bar1")
        st.bar_chart(filtered_data[cat_col_bar].value_counts())

    # Pie chart (Plotly)
    if cat_cols:
        pie_col = st.selectbox("Select categorical column for pie chart", cat_cols, key="pie1")
        pie_data = filtered_data[pie_col].value_counts()
        fig4 = px.pie(values=pie_data.values, names=pie_data.index, title=f"Pie chart of {pie_col}")
        st.plotly_chart(fig4)

    # Line chart for trends over time (if date column exists)
    date_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col]) or "date" in col.lower()]
    if date_cols and numeric_cols:
        date_col = st.selectbox("Select date column for trend analysis", date_cols)
        num_col_line = st.selectbox("Select numeric column for trend", numeric_cols, key="line1")
        trend_data = filtered_data.groupby(date_col)[num_col_line].mean().reset_index()
        fig5 = px.line(trend_data, x=date_col, y=num_col_line, title=f"{num_col_line} over Time")
        st.plotly_chart(fig5)

    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        fig6, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(filtered_data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.write("### Correlation Heatmap")
        st.pyplot(fig6)

    # Stacked bar chart for two categorical columns
    if len(cat_cols) >= 2:
        cat_x = st.selectbox("X-axis categorical for stacked bar", cat_cols, key="stack1")
        cat_hue = st.selectbox("Hue categorical for stacked bar", cat_cols, key="stack2")
        cross_tab = pd.crosstab(filtered_data[cat_x], filtered_data[cat_hue])
        fig7 = cross_tab.plot(kind='bar', stacked=True, figsize=(8,5))
        plt.ylabel("Count")
        plt.title(f"Stacked Bar Chart: {cat_x} vs {cat_hue}")
        st.pyplot(plt.gcf())

    # Map visualization (if latitude and longitude columns exist)
    lat_cols = [col for col in data.columns if "lat" in col.lower()]
    lon_cols = [col for col in data.columns if "lon" in col.lower() or "lng" in col.lower()]
    if lat_cols and lon_cols:
        st.write("### Map Visualization")
        st.map(filtered_data.rename(columns={lat_cols[0]: "lat", lon_cols[0]: "lon"}))

    # ---------------------------
    # 6. Summary Statistics
    # ---------------------------
    st.write("### Summary Statistics of Filtered Data")
    st.write(filtered_data.describe(include='all'))
else:
    st.info("Please upload a CSV file to start analysis.")
