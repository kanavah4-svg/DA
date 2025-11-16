
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="ATELIER 8 Dashboard", layout="wide")

# --- Custom styling ---
st.markdown(
    """
    <style>
        body {
            background-color: #f5f0e6;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4 {
            color: #3e3e3e;
        }
        .stButton button {
            background-color: #e6ccb2;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# --- Team Info ---
with st.container():
    st.markdown(
        "<div style='text-align: right; font-size:16px;'>Kanav, Omkar, Jigyasa, Hardik, Harshal</div>",
        unsafe_allow_html=True
    )

# --- Tabs ---
tabs = st.tabs(["Overview & Filters", "Customer Segments (Clustering)", "Adoption Prediction (Classification)", "Pricing Insight (Regression)"])

# --- Tab 1 ---
with tabs[0]:
    st.header("Overview & Filters")

    with st.expander("Raw Data Preview (first 5 rows)"):
        st.dataframe(df.head())

    col = st.selectbox("Choose a numeric column to visualize", df.select_dtypes("number").columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=15, kde=True, ax=ax, color="#c6a27e")
    st.pyplot(fig)
    st.caption(f"Distribution of {col} shows general trends in respondent preferences or traits.")

# --- Tab 2 ---
with tabs[1]:
    st.header("Customer Segments (K-Means Clustering)")
    st.markdown("This clustering algorithm groups similar customers to identify patterns in their preferences.")

    X = df[["age", "income_level", "num_luxury_items", "sustainability_score"]]
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    df["cluster"] = model.labels_

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age", y="income_level", hue="cluster", palette="pastel", ax=ax)
    st.pyplot(fig)
    with st.expander("What is K-Means Clustering?"):
        st.write("K-Means is an unsupervised learning algorithm that groups data into clusters based on feature similarity.")

# --- Tab 3 ---
with tabs[2]:
    st.header("Adoption Prediction (Classification)")
    st.markdown("We predict adoption intent using Random Forest classifier.")

    Xc = df[["age", "income_level", "num_luxury_items", "wtp_restoration", "sustainability_score"]]
    y = df["adoption_intent"]

    model = RandomForestClassifier(random_state=42).fit(Xc, y)
    pred = model.predict(Xc)
    st.text(classification_report(y, pred))
    with st.expander("What is Classification?"):
        st.write("Classification predicts categories or labels. Here, we classify adoption intent levels.")

# --- Tab 4 ---
with tabs[3]:
    st.header("Pricing Insight (Regression)")
    st.markdown("Linear regression estimates the price customers are willing to pay.")

    Xr = df[["income_level", "num_luxury_items", "sustainability_score"]]
    yr = df["wtp_restoration"]

    reg = LinearRegression().fit(Xr, yr)
    yr_pred = reg.predict(Xr)

    rmse = mean_squared_error(yr, yr_pred, squared=False)
    r2 = r2_score(yr, yr_pred)

    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    with st.expander("What is Regression?"):
        st.write("Regression models relationships between variables to predict numeric outcomes like price.")
