import pandas as pd

# Membaca dataset Mobile Legends
df = pd.read_csv("mobile_legends_skins_dataset_1500.csv")
df.columns = df.columns.str.strip()

# Menampilkan informasi umum dataset
df.info()

# 1. Memisahkan data kategorikal dan numerikal
kategorikal = df.select_dtypes(include=["object", "str"])
numerikal = df.select_dtypes(include=["number"])

print("DATA KATEGORIKAL")
print(kategorikal.columns.tolist())

print("\nDATA NUMERIKAL")
print(numerikal.columns.tolist())

# 2. Menampilkan tipe data setiap kolom
print("\nJENIS DATA")
print(df.dtypes)

# 3. Menghitung data duplikat
print("\nJUMLAH DUPLIKAT")
print(df.duplicated().sum())

# Menghitung total nilai null di seluruh data
print("\nJUMLAH NULL")
print(df.isnull().sum().sum())

# Menghitung angka nol per kolom numerikal
print("\nJUMLAH ZERO")
print(numerikal.eq(0).sum())


## kategorikal = 7
## numerikal = 15