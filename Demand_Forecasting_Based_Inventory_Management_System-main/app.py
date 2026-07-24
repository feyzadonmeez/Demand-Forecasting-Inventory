import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from main import load_and_prepare_data
from forecasting import train_forecast_model

st.set_page_config(layout="wide", page_title="Stok Yönetim Sistemi")


@st.cache_data
def load_dashboard_data():
    summary_df = pd.read_csv("data/dashboard_summary.csv")
    rfm_df = pd.read_csv("data/rfm_results.csv")
    monthly = pd.read_csv('data/aggregated_monthly_sales.csv')
    weekly = pd.read_csv('data/aggregated_weekly_sales.csv')
    quarterly = pd.read_csv('data/aggregated_quarterly_sales.csv')  # Yeni
    raw_df = pd.read_csv('data/df.csv')
    # raw_df = load_and_prepare_data() model çalıştıktan sonra csv'ye kaydediyor, dashboard direkt csv'den okuyor. -Feyza

    return summary_df, rfm_df, monthly, weekly, quarterly, raw_df

#csv'ler oluştırulmamışsa uyarı versin: -Kübranur
try:
    df_all_status, rfm_df, monthly, weekly, quarterly, raw_df = load_dashboard_data()
except FileNotFoundError:
    st.error("Veri dosyaları bulunamadı! Lütfen önce 'run_analysis.py' dosyasını çalıştırın.")
    st.stop()


tabs = st.tabs(["📊 Genel Tablo", "📦 Overstock & RFM", "📈 Detaylı Simülasyon"])

# --------------------------------------  1. SEKME: GENEL TABLO  -------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([4, 3])

    with col1:
        st.markdown("### 🚨 Kritik Stok Listesi")
        st.caption("Aşağıdaki listede **Stok Yok** veya **Stok Yetersiz** durumundaki ürünler yer almaktadır.")

        critical_stock = df_all_status[
            df_all_status["Status"].isin(["Stok Yok", "Stok Yetersiz"])
        ][["ProductId", "CategoryName", "RemainingStock", "NextMonthForecast", "BestModel", "Status"]]

        st.dataframe(
            critical_stock,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ProductId": "Ürün Kodu",
                "CategoryName": "Kategori",
                "RemainingStock": "Mevcut Stok",
                "NextMonthForecast": "Tahmin",
                "BestModel": "Model",
                "Status": "Durum"
            }
        )

    with col2:
        st.markdown("### 📊 Genel Satış Özeti")

        last_month = monthly["OrderDate"].max()

        pie_data = monthly[monthly["OrderDate"] == last_month].groupby("CategoryName")["Quantity"].sum().reset_index()
        pie_data["Share"] = pie_data["Quantity"] / pie_data["Quantity"].sum()

        base = alt.Chart(pie_data).encode(
            theta=alt.Theta("Quantity:Q", stack=True)
        )

        pie = base.mark_arc(innerRadius=50).encode(
            color=alt.Color("CategoryName:N", legend=alt.Legend(title="Kategoriler")),
            tooltip=["CategoryName", "Quantity", alt.Tooltip("Share", format=".1%", title="Oran")]
        )

        text = base.mark_text(radius=120).encode(
            text=alt.Text("Share:Q", format=".1%"),
            order=alt.Order("CategoryName:N"),
            color=alt.value("black")
        )
        st.altair_chart(pie + text, use_container_width=True)

        bar = alt.Chart(monthly).mark_bar().encode(
            x=alt.X("yearmonth(OrderDate):T", title="Tarih"),
            y=alt.Y("sum(TotalSalesPrice):Q", title="Toplam Ciro"),
            tooltip=["yearmonth(OrderDate)", "sum(TotalSalesPrice)"]
        ).properties(title="Aylık Ciro Gelişimi")

        st.altair_chart(bar, use_container_width=True)

# ----------------------------------------   OVERSTOCK & RFM  -----------------------------------
with tabs[1]:
    col_os1, col_os2 = st.columns([1, 1])

    overstock_df = df_all_status[df_all_status["Status"] == "Aşırı Stok"]

    with col_os1:
        st.markdown("### 📦 Aşırı Stok Listesi")
        if not overstock_df.empty:
            st.caption("Detaylı RFM analizi için listeden bir ürün seçiniz.")
            event = st.dataframe(
                overstock_df[["ProductId", "RemainingStock", "NextMonthForecast", "CategoryName"]],
                selection_mode="single-row",
                on_select="rerun",
                hide_index=True,
                use_container_width=True,
                key="overstock_table"
            )
        else:
            st.success("Sistemde aşırı stoklu ürün bulunmamaktadır.")
            event = None

    with col_os2:
        if event and len(event.selection["rows"]) > 0:
            selected_index = event.selection["rows"][0]
            selected_row = overstock_df.iloc[selected_index]

            selected_pid = selected_row["ProductId"]
            selected_cat_name = selected_row["CategoryName"]
            selected_cat_id = df_all_status[df_all_status["ProductId"] == selected_pid]["CategoryId"].iloc[0]

            st.markdown(f"### 👥 Müşteri Analizi: {selected_cat_name}")
            st.info(f"Seçilen Ürün: **{selected_pid}**")
            st.write("Bu kategorideki satışların RFM segmentlerine göre dağılımı (Adet):")

            cat_sales_data = raw_df[raw_df["CategoryId"] == selected_cat_id]

            if not cat_sales_data.empty and "CustomerId" in rfm_df.columns:
                merged_data = pd.merge(cat_sales_data, rfm_df, on="CustomerId", how="inner")

                segment_stats = merged_data.groupby("Segment")["Quantity"].sum().reset_index()
                segment_stats.columns = ["Segment", "TotalQuantity"]

                rfm_chart = alt.Chart(segment_stats).mark_bar().encode(
                    x=alt.X("TotalQuantity:Q", title="Satılan Ürün Adeti"),
                    y=alt.Y("Segment:N", sort="-x", title="Segment"),
                    color="Segment:N",
                    tooltip=["Segment", "TotalQuantity"]
                ).properties(title=f"{selected_cat_name} Kategorisi RFM Satış Dağılımı")

                st.altair_chart(rfm_chart, use_container_width=True)
            else:
                st.warning("Bu kategori için müşteri/segment verisi eşleştirilemedi.")
        else:
            st.info("👈 Analizi görüntülemek için soldaki tablodan bir satıra tıklayın.")

#*-----------------------------------------------------3. SEKME: DETAYLI SİMÜLASYON----------------------------------------------------
with tabs[2]:
    st.markdown("### 🤖 Canlı Tahmin Simülasyonu")

    col_sel1, col_sel2, col_sel3 = st.columns(3)

    with col_sel1:
        product_list = monthly["ProductId"].unique()
        product = st.selectbox("Ürün Kodu Seçiniz", product_list)

    with col_sel2:
        model_choice = st.radio("Model Tipi", ["GRU", "LSTM"], horizontal=True)

    with col_sel3:
        # 3 Aylık Seçeneği Eklendi
        period_choice = st.radio("Periyot", ["Aylık (Monthly)", "Haftalık (Weekly)", "3 Aylık (Quarterly)"],
                                 horizontal=True)

    if st.button("Modeli Çalıştır"):
        with st.spinner(f"({model_choice}) modeli iki farklı strateji ile çalışıyor"):

            if "Haftalık" in period_choice:
                data_source = weekly
                win, hor = 12, 8
                period_name = "Weekly"
                freq = "W-MON"
            elif "3 Aylık" in period_choice:
                data_source = quarterly
                win, hor = 2, 2
                period_name = "Quarterly"
                freq = "QE"  # Quarter End
            else:
                data_source = monthly
                win, hor = 6, 6
                period_name = "Monthly"
                freq = "M"

            series = data_source[data_source["ProductId"] == product].set_index("OrderDate")["Quantity"]

            #  (MSE):

            preds_mse, _, metrics_mse, val_preds_mse = train_forecast_model(
                series, model_type=model_choice, window=win, horizon=hor, loss_type="mse"
            )


            # (PINBALL)

            preds_pinball, _, metrics_pinball, val_preds_pinball = train_forecast_model(
                series, model_type=model_choice, window=win, horizon=hor, loss_type="pinball", quantile=0.90
            )

            if preds_mse is not None:
                hist = series.reset_index()
                hist["Type"] = "Gerçekleşen"

                split_point = int(len(series) * 0.8)
                val_start_index = split_point + win
                val_df_list = []

                if len(val_preds_mse) > 0 and val_start_index < len(series):
                    val_dates = series.index[val_start_index:]
                    limit = min(len(val_dates), len(val_preds_mse))
                    val_df_list.append(pd.DataFrame({
                        "OrderDate": val_dates[:limit],
                        "Quantity": val_preds_mse[:limit],
                        "Type": f"Model Test (Std - {model_choice})"
                    }))

                    val_df_list.append(pd.DataFrame({
                        "OrderDate": val_dates[:limit],
                        "Quantity": val_preds_pinball[:limit],
                        "Type": f"Model Test (Güvenli - Pinball)"
                    }))

                val_df = pd.concat(val_df_list) if val_df_list else pd.DataFrame()

                future_dates = pd.date_range(series.index.max(), periods=hor + 1, freq=freq)[1:]

                pred_df_mse = pd.DataFrame({
                    "OrderDate": future_dates,
                    "Quantity": preds_mse,
                    "Type": f"Tahmin (Std - {model_choice})"
                })

                pred_df_pinball = pd.DataFrame({
                    "OrderDate": future_dates,
                    "Quantity": preds_pinball,
                    "Type": f"Tahmin (Güvenli - Pinball)"
                })

                plot = pd.concat([hist, val_df, pred_df_mse, pred_df_pinball])

                chart = alt.Chart(plot).mark_line(point=True).encode(
                    x="OrderDate:T",
                    y="Quantity:Q",
                    color=alt.Color("Type:N", legend=alt.Legend(orient="bottom", title="Veri Tipi")),
                    tooltip=[
                        alt.Tooltip("OrderDate:T", title="Tarih", format="%d %B %Y"),
                        alt.Tooltip("Quantity:Q", title="Adet", format=",.1f"),
                        alt.Tooltip("Type:N", title="Veri Tipi")
                    ]
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                m_col1, m_col2 = st.columns(2) # Başarılar yanyana gösterilsin

                with m_col1:
                    st.subheader("Standart Model (MSE)")
                    st.caption("Ortalamayı hedefler. Daha az hata yapar ama pikleri kaçırabilir.")
                    st.metric("Model Başarısı (RMSE)", metrics_mse['RMSE'])

                with m_col2:
                    st.subheader("Güvenli Model (Pinball Loss)")
                    st.caption("Stoksuz kalmamayı hedefler (%90). Tahminler daha yüksek çıkar.")
                    # Pinball Loss skorunu gösteriyoruz
                    st.metric("Risk Skoru (Pinball Loss)", metrics_pinball['PinballLoss'])

            else:
                st.error("Bu ürün için model oluşturmaya yetecek kadar geçmiş veri bulunmuyor.")