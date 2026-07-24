import pandas as pd

category_map = {
    1: "Gıda",
    2: "İçecek",
    3: "Elektronik",
    4: "Giyim",
    5: "Mobilya & Dekorasyon",
    6: "Kişisel Bakım & Kozmetik",
    7: "Otomotiv & Araç Bakım",
    8: "Kamp & Outdoor",
    9: "Evcil Hayvan",
    10: "Kafe & Pastane",
    11: "Kırtasiye & Ofis",
    12: "Oyuncak & Hobi",
    13: "Bahçe & Dış Mekan"
}

def load_and_prepare_data():
    df = pd.read_csv("data/df_merged.csv")
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    df["TotalSalesPrice"] = df["Quantity"] * df["UnitPrice"]
    df["CategoryName"] = df["CategoryId"].map(category_map)

    # aylık grupla
    monthly = (
        df
        .set_index("OrderDate")
        .groupby("ProductId")
        .resample("M")
        .agg({
            "Quantity": "sum",
            "TotalSalesPrice": "sum",
            "StockQuantity": "last",
            "CategoryId": "last",
            "CategoryName": "last"
        })
        .reset_index()
    )
    # haftalık grupla
    weekly = (
        df
        .set_index("OrderDate")
        .groupby("ProductId")
        .resample("W-MON")
        .agg({
            "Quantity": "sum",
            "TotalSalesPrice": "sum",
            "StockQuantity": "last",
            "CategoryId": "last",
            "CategoryName": "last"
        })
        .reset_index()
    )

# Seyrek veri problemi için 3 aylık gruplama ekledim. Deneriz, iyi sonuç çıkmazsa sileriz 3 aylık gruplama için tarih aralığımız az olabilri. -Burak
    quarterly = (
        df
        .set_index("OrderDate")
        .groupby("ProductId")
        .resample("QE")
        .agg({
            "Quantity": "sum",
            "TotalSalesPrice": "sum",
            "StockQuantity": "last",
            "CategoryId": "last",
            "CategoryName": "last"
        })
        .reset_index()
    )

    monthly.to_csv("data/monthly_sales.csv", index=False)
    weekly.to_csv("data/weekly_sales.csv", index=False)
    quarterly.to_csv("data/quarterly_sales.csv", index=False)
    df.to_csv("data/df.csv", index=False)


    return monthly, weekly, quarterly, df