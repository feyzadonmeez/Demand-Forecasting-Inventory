import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import streamlit as st
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import warnings
from sklearn.exceptions import ConvergenceWarning

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("data/df_merged.csv")
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    num_cols = dataframe.select_dtypes(include=["number"])
    print(num_cols.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

df[cat_cols].head()
df[num_cols].head()
df[cat_but_car].head()

id_cols = [col for col in df.columns if ("Id" or "ID") in col]

for col in id_cols:
  df[col] = df[col].astype("object")

df[id_cols].info()

cat_cols, cat_but_car, num_cols = grab_col_names(df)

df[cat_cols].head()
df[num_cols].head()
df[cat_but_car].head()

### Kategorik Değişken Analizi ###

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        # plt.figure(figsize=(15, 12))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, True)

### Sayısal Değişken Analizi ###

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8, 8))
        dataframe[numerical_col].hist()
        plt.xticks(rotation=25, ha='right', fontsize=15)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        # plt.xlabel(numerical_col, fontsize=12)
        plt.title(numerical_col, fontsize=18)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, True)

### Quantity ile Kategorik Değişken Analizi ###

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Quantity", col)

### Quantity ile Sayısal Değişken Analizi ###

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Quantity", col)

### Tarihe göre Quantity Analizi ###

df.columns
df_monthly_orders = df.groupby([df['OrderDate'].dt.to_period('M')]).agg({'Quantity': 'sum'})
df_monthly_orders.reset_index(inplace=True)
df_monthly_orders['OrderDate'] = df_monthly_orders['OrderDate'].dt.to_timestamp()
df_monthly_orders['Year'] = df_monthly_orders['OrderDate'].dt.year
unique_years = df_monthly_orders['Year'].nunique()
palette = sns.color_palette("viridis", unique_years)
year_to_color = {year: color for year, color in zip(df_monthly_orders['Year'].unique(), palette)}

plt.bar(df_monthly_orders['OrderDate'],
        df_monthly_orders['Quantity'],
        width=25, # Bar genişliği
        color=[year_to_color[year] for year in df_monthly_orders['Year']])
plt.title("Monthly Orders", fontsize=15)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show(block=True)

cat_cols, cat_but_car, num_cols = grab_col_names(df)

### Korelasyon Analizi ###

corr = df[num_cols].corr()
# plt.figure(figsize=(15, 8))
sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f")
plt.tight_layout()
plt.show()


