import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------
# 1. Load Dataset
# ---------------------------
st.title("Interactive Dashboard")

# Replace with your dataset path
data = pd.read_csv("/home/adminuser/DV_Assignment/train.csv")

st.write("### Sample of the dataset")
st.dataframe(data.head())

# ---------------------------
# 2. Select Column for Analysis
# ---------------------------
column_select = st.selectbox("Select column for analysis", data.columns)

# ---------------------------
# 3. Handle Filtering Based on Column Type
# ---------------------------

if pd.api.types.is_numeric_dtype(data[column_select]):
    # Numeric column -> Slider
    min_val = float(data[column_select].min())
    max_val = float(data[column_select].max())
    range_slider = st.slider(
        f"Select range for {column_select}", min_val, max_val, (min_val, max_val)
    )
    filtered_data = data[(data[column_select] >= range_slider[0]) &
                         (data[column_select] <= range_slider[1])]
    
elif pd.api.types.is_datetime64_any_dtype(data[column_select]) or "date" in column_select.lower():
    # Date column -> Date Input
    data[column_select] = pd.to_datetime(data[column_select], errors='coerce')
    min_date = data[column_select].min()
    max_date = data[column_select].max()
    date_range = st.date_input("Select date range", [min_date, max_date])
    filtered_data = data[(data[column_select] >= pd.to_datetime(date_range[0])) &
                         (data[column_select] <= pd.to_datetime(date_range[1]))]
else:
    # Categorical column -> Multiselect
    categories = data[column_select].unique()
    selected_categories = st.multiselect("Select categories", categories, default=categories)
    filtered_data = data[data[column_select].isin(selected_categories)]

# ---------------------------
# 4. Additional Widgets
# ---------------------------
# Example: Filter by another categorical column if exists
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

st.write("### Plots")

# Plot 1: Histogram for numeric columns
numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
if numeric_cols:
    col1 = st.selectbox("Select numeric column for histogram", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(filtered_data[col1], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Plot 2: Scatter plot (numeric vs numeric)
if len(numeric_cols) >= 2:
    col_x = st.selectbox("X-axis for scatter plot", numeric_cols, index=0)
    col_y = st.selectbox("Y-axis for scatter plot", numeric_cols, index=1)
    fig2 = px.scatter(filtered_data, x=col_x, y=col_y)
    st.plotly_chart(fig2)

# Plot 3: Bar chart for a categorical column
cat_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    cat_col = st.selectbox("Select categorical column for bar chart", cat_cols)
    st.bar_chart(filtered_data[cat_col].value_counts())

# Plot 4: Boxplot for numeric by category
if numeric_cols and cat_cols:
    num_col = st.selectbox("Numeric column for boxplot", numeric_cols, key="box1")
    cat_col_box = st.selectbox("Category column for boxplot", cat_cols, key="box2")
    fig3, ax = plt.subplots()
    sns.boxplot(x=filtered_data[cat_col_box], y=filtered_data[num_col], ax=ax)
    st.pyplot(fig3)

# Plot 5: Optional pie chart for a categorical column
if cat_cols:
    pie_col = st.selectbox("Select categorical column for pie chart", cat_cols, key="pie1")
    pie_data = filtered_data[pie_col].value_counts()
    fig4 = px.pie(values=pie_data.values, names=pie_data.index, title=f"Pie chart of {pie_col}")
    st.plotly_chart(fig4)

# ---------------------------
# 6. Summary Statistics
# ---------------------------
st.write("### Summary Statistics of Filtered Data")
st.write(filtered_data.describe(include='all'))

