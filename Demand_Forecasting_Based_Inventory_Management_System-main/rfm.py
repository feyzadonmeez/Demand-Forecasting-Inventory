import pandas as pd
import datetime as dt


def rfm_analysis(df):
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    today_date = df["OrderDate"].max() + dt.timedelta(days=2)
    group_col = "CustomerId" if "CustomerId" in df.columns else "ProductId"

    rfm = df.groupby(group_col).agg({
        'OrderDate': lambda date: (today_date - date.max()).days,  # Recency
        'ProductId': lambda num: len(num),  # Frequency
        'TotalSalesPrice': lambda price: price.sum()  # Monetary
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm["RecencyScore"] = pd.qcut(rfm['Recency'].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
    rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str))

# Segmentler türkçeye çevrildi -Kübra
    seg_map = {
        r'[1-2][1-2]': 'Uykuda',
        r'[1-2][3-4]': 'Risk Altında',
        r'[1-2]5': 'Kaybedilmek Üzere',
        r'3[1-2]': 'Uykuya Yakın',
        r'33': 'İlgi Bekleyen',
        r'[3-4][4-5]': 'Sadık Müşteriler',
        r'41': 'Umut Veren',
        r'51': 'Yeni Müşteriler',
        r'[4-5][2-3]': 'Potansiyel Sadıklar',
        r'5[4-5]': 'Şampiyonlar'
    }

    rfm['Segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

    return rfm