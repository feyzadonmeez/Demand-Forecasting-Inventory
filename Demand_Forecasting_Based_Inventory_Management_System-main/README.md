> **Co-developed with @burak-basoglu**
   
> **Live Dashboard:** https://hypeproject.streamlit.app/


# 📦 Demand Forecasting-Based Inventory Management System

A comprehensive, data-driven decision support system designed to optimize inventory management through **demand forecasting**, **model comparison**, and **customer-centric insights**.

This project goes beyond traditional forecasting by integrating **business logic, analytics, and usability layers** into a single workflow.

---

## 🧩 Business Problem

Inventory management is a critical challenge in retail operations.

- Understocking leads to **lost sales and poor customer experience**
- Overstocking results in **increased holding costs and inefficient capital allocation**

To enable effective planning, businesses must answer:

- What will be the future demand for each product?
- Which products require immediate replenishment?
- How can forecasting outputs be translated into actionable decisions?
- Which customer segments should be targeted under stock constraints?

This project addresses these questions by building an **end-to-end forecasting and decision support system**.

---

## 🎯 Project Objectives

- Forecast product demand using historical sales data  
- Compare different forecasting approaches and evaluate performance  
- Support inventory planning with actionable insights  
- Aggregate demand across multiple time levels:
  - Weekly
  - Monthly
  - Quarterly  
- Integrate **RFM segmentation** to connect inventory decisions with customer value  
- Deliver outputs in a **dashboard-ready format**  

---

## 🧠 Solution Architecture

The system is structured as a multi-layer analytical pipeline:

### 1. Data Preparation
- Cleaning transactional data  
- Handling missing values  
- Removing anomalies and invalid entries  
- Aggregating data into time-based structures  

### 2. Exploratory Data Analysis (EDA)
- Understanding demand distribution and patterns  
- Identifying seasonality and trends  
- Extracting business-relevant insights  

### 3. Demand Forecasting
- Modeling future demand using historical data  
- Generating forecasts across different time granularities  
- Supporting operational planning  

### 4. Model Evaluation
- Comparing forecasting models  
- Storing results in `model_comparison.csv`  
- Selecting the most effective model  

### 5. Customer Segmentation (RFM)
- Identifying high-value and at-risk customers  
- Supporting marketing and inventory decisions  
- Linking demand insights with customer behavior  

### 6. Application Layer (Streamlit)
- Providing an interactive interface  
- Making results accessible to non-technical users  
- Enabling quick business interpretation  

---

## ⚙️ Methodology

### 📊 Demand Forecasting
- Time-based aggregation (weekly, monthly, quarterly)
- Forecast generation using historical demand patterns
- Model comparison and evaluation

### 👥 RFM Analysis
- Recency → How recently a customer purchased  
- Frequency → How often they purchase  
- Monetary → How much they spend  

### 📈 Model Evaluation
- Comparative performance tracking  
- Selection based on predictive effectiveness  

---

## 🧮 Data Assets

The project includes multiple processed datasets:

### Raw & Processed Data
- `df.csv`
- `df_merged.csv`

### Time-Based Aggregations
- `weekly_sales.csv`
- `monthly_sales.csv`
- `quarterly_sales.csv`

### Aggregated Outputs
- `aggregated_weekly_sales.csv`
- `aggregated_monthly_sales.csv`
- `aggregated_quarterly_sales.csv`

### Model Evaluation
- `model_comparison.csv`

### Dashboard Outputs
- `dashboard_summary.csv`
- `dashboard_summary_main.csv`

### Customer Segmentation
- `rfm_results.csv`

---

## 🛠️ Tech Stack

**Programming Language**
- Python

**Core Libraries**
- pandas
- numpy
- scikit-learn
- tensorflow

**Visualization & App Layer**
- streamlit
- altair

---

## 🚀 How to Run

### 1. Clone the repository

git clone "repository-url"  
cd Demand_Forecasting_Based_Inventory_Management_System

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run analysis

python main.py

### 4. Launch Streamlit app

streamlit run app.py

## 📊 Business Value

This system provides measurable impact in real-world scenarios:

### ✅ Reduced Stockout Risk
Forecast-driven planning helps maintain product availability.

### ✅ Lower Inventory Costs
Prevents overstocking and reduces capital lock-in.

### ✅ Smarter Decision-Making
Combines forecasting with segmentation for better strategy.

### ✅ Cross-Functional Value
Useful for:
- Data teams  
- Supply chain teams  
- Marketing teams  

---

## 💡 Example Use Cases

- Identifying products with upcoming high demand  
- Prioritizing inventory replenishment  
- Designing targeted campaigns for high-value customers  
- Monitoring performance through dashboards  

---

## 🔍 Key Strengths of the Project

This project demonstrates:

- Strong **problem framing**  
- Integration of **DL + business logic**  
- Focus on **decision support, not just prediction**  
- Multi-level data aggregation  
- Customer-centric thinking  

> This is not just a forecasting model —  
> it is a **decision support system for inventory optimization**.

---

## 🔮 Future Improvements

- Safety stock calculation  
- Reorder point optimization  
- Lead time integration  
- API deployment  
- Automated retraining pipeline  
- Real-time dashboard integration  
