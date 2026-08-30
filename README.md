# Auto Sales — Customer Segmentation

A customer segmentation tool for automotive sales data, built using **RFM analysis** (Recency, Frequency, Monetary) and **K-Means clustering**. Deployed as an interactive Streamlit app.

🔗 **Live app:** https://autosalescustomersegmentation.streamlit.app/

---

## 📌 Overview

This project segments customers based on their purchasing behavior to help identify high-value customers, at-risk accounts, loyal buyers, and inactive customers — enabling targeted retention and marketing strategies.

The approach is based on:
- **Recency** — days since a customer's last order
- **Frequency** — number of distinct orders placed
- **Monetary** — total amount spent

These three features are scaled and clustered using **K-Means**, with the optimal number of clusters chosen using the elbow method and silhouette score.

---

## 🧠 Methodology

1. **Data exploration** — distribution of sales, orders by product line, orders by country
2. **RFM feature engineering** — aggregated per customer from raw order-level data
3. **Log transformation** — applied to Monetary to reduce skew (`log1p`)
4. **Standardization** — features scaled using `StandardScaler`
5. **Clustering** — `KMeans` with `k` selected via elbow method, silhouette score, Davies-Bouldin index, and Calinski-Harabasz index
6. **Dimensionality reduction** — PCA used for 2D visualization of clusters
7. **Segment labeling** — clusters are automatically labeled in plain English (e.g., "Top / VIP Customers", "Inactive / Churned") based on their relative RFM standing

---

## 🗂️ Repository Structure

| File | Description |
|---|---|
| `app.py` | Streamlit web app — upload CSV, view segments, predict new customer segments |
| `Auto Sales data.csv` | Training dataset (also used as the downloadable sample/reference format) |
| `kmeans_model.pkl` | Pre-trained K-Means model (k=3) |
| `scaler.pkl` | Pre-trained `StandardScaler` used with the model |
| `requirements.txt` | Python dependencies |

---

## 🚀 Features

- **Upload any compatible CSV** — the app validates and auto-maps common column-naming variations (e.g., `Customer Name` → `CUSTOMERNAME`)
- **Manual column mapping fallback** — if a column can't be auto-detected, the user can map it manually via dropdowns
- **Interactive filters** — filter by country and product line
- **Adjustable clustering** — choose the number of clusters (k) live
- **Human-readable segment labels** — clusters are automatically named (VIP, At Risk, Loyal, Inactive, etc.) instead of raw numbers
- **Customer lookup** — search any customer to see their segment and RFM values
- **Predict segment for a new/hypothetical customer** — enter RFM values manually and get an instant segment prediction using the saved model

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Streamlit** — web app framework
- **Plotly** — interactive visualizations
- **joblib** — model persistence

---

## ▶️ Running Locally

```bash
git clone <repo-url>
cd auto_sales_customer_segmentation
pip install -r requirements.txt
streamlit run app.py
