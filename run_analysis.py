import os
import pandas as pd
import numpy as np
import time
import warnings
import gc

from main import load_and_prepare_data
from forecasting import train_forecast_model
from rfm import rfm_analysis

warnings.filterwarnings("ignore")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Tensorflow çökmesin diye bunu yazmamız gerekiyormuş -Ahmet
# Yine de çöküyor. -Ahmet

def stock_status(remaining, forecast):
    if remaining <= 0:
        return "Stok Yok"
    elif remaining < forecast:
        return "Stok Yetersiz"
    elif remaining > forecast * 6:
        return "Aşırı Stok"
    else:
        return "Stok Yeterli"

# Progress takibi için printlerle güncelledim. -Burak
def main():
    print("⏳ Veriler yükleniyor...")
    monthly, weekly, quarterly, raw_df = load_and_prepare_data()
    print("Özet veriler kaydediliyor...")
    monthly.to_csv("data/aggregated_monthly_sales.csv", index=False)
    weekly.to_csv("data/aggregated_weekly_sales.csv", index=False)
    quarterly.to_csv("data/aggregated_quarterly_sales.csv", index=False)  # Yeni

    print("   -> data/aggregated_monthly_sales.csv (Oluşturuldu)")
    print("   -> data/aggregated_weekly_sales.csv (Oluşturuldu)")
    print("   -> data/aggregated_quarterly_sales.csv (Oluşturuldu)")

    print("RFM analizi yapılıyor...")
    rfm_df = rfm_analysis(raw_df)
    rfm_df.reset_index().to_csv("data/rfm_results.csv", index=False)

    print("Her kategoriden en çok satan 10 ürün seçiliyor...")

    product_sales = monthly.groupby(["CategoryId", "ProductId"])["Quantity"].sum().reset_index()
    product_sales = product_sales.sort_values(["CategoryId", "Quantity"], ascending=[True, False])
    top_products_df = product_sales.groupby("CategoryId").head(10)  # 20 ürünle modele gidince ram yetmiyo. Colab GPU da yetmedi. 10 ürüne düşürdüm. -Burak
    target_product_ids = top_products_df["ProductId"].unique()

    last_month_date = monthly["OrderDate"].max()
    last_df_ref = monthly[monthly["OrderDate"] == last_month_date]

    print(f"🚀 Toplam {len(target_product_ids)} ürün için analiz başlıyor...")

    forecast_results = []
    start_time = time.time() # başladığında sistem zamanını alıyormuş.

    for i, pid in enumerate(target_product_ids):
        if i % 10 == 0:
            elapsed = int(time.time() - start_time)
            print(f"   İşleniyor... {i + 1}/{len(target_product_ids)} (Geçen: {elapsed}sn)")
            gc.collect()

        product_info_row = last_df_ref[last_df_ref["ProductId"] == pid]

        if product_info_row.empty:
            stock_val = 0
            try:
                cat_name = monthly[monthly["ProductId"] == pid]["CategoryName"].iloc[-1]
                cat_id = monthly[monthly["ProductId"] == pid]["CategoryId"].iloc[-1]
            except:
                cat_name = "Bilinmiyor"
                cat_id = 0
        else:
            stock_val = product_info_row["StockQuantity"].iloc[0]
            cat_name = product_info_row["CategoryName"].iloc[0]
            cat_id = product_info_row["CategoryId"].iloc[0]

        series_monthly = monthly[monthly["ProductId"] == pid].set_index("OrderDate")["Quantity"]
        series_weekly = weekly[weekly["ProductId"] == pid].set_index("OrderDate")["Quantity"]
        series_quarterly = quarterly[quarterly["ProductId"] == pid].set_index("OrderDate")["Quantity"]

        configs = [
            (series_monthly, "Monthly", "GRU", 6, 6),
            (series_monthly, "Monthly", "LSTM", 6, 6),
            (series_weekly, "Weekly", "GRU", 12, 8),
            (series_weekly, "Weekly", "LSTM", 12, 8),
            (series_quarterly, "Quarterly", "GRU", 2, 2), # 3 aylıkta pencere ancak 2 oluyor. onda da yeterli veri yok dediği ürünler var. -Byrak
            (series_quarterly, "Quarterly", "LSTM", 2, 2)
        ]

        best_rmse = float('inf')
        selected_forecast_val = 0
        best_model_name = "None"

        for series_data, period, model_name, win, hor in configs:
            preds, ma, metrics, _ = train_forecast_model(series_data, model_type=model_name, window=win, horizon=hor)
            if preds is not None:
                if period == "Monthly":
                    next_pred = preds[0]

                elif period == "Weekly":
                    next_pred = np.sum(preds[:4])  # 4 haftalık toplam

                elif period == "Quarterly":
                    next_pred = preds[0] / 3

                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    selected_forecast_val = next_pred
                    best_model_name = f"{model_name} ({period})"

        if best_rmse == float('inf'):
            selected_forecast_val = series_monthly[-3:].mean() if len(series_monthly) > 0 else 0
            best_model_name = "Simple Average"

        remaining_final = stock_val
        status = stock_status(remaining_final, selected_forecast_val)

        forecast_results.append({
            "ProductId": pid,
            "RemainingStock": remaining_final,
            "NextMonthForecast": round(selected_forecast_val, 2),
            "BestModel": best_model_name,
            "Status": status,
            "CategoryName": cat_name,
            "CategoryId": cat_id
        })

    df_results = pd.DataFrame(forecast_results)
    df_results.to_csv("data/dashboard_summary.csv", index=False)
    print(" Analiz tamamlandı!")
    print(" OLUŞTURULAN DOSYALAR:")
    print("   1. data/dashboard_summary.csv")
    print("   2. data/rfm_results.csv")
    print("   3. data/aggregated_monthly_sales.csv")
    print("   4. data/aggregated_weekly_sales.csv")
    print("   5. data/aggregated_quarterly_sales.csv")


if __name__ == "__main__":
    main()