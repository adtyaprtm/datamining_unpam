from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def input_atau_default(prompt, default, converter=None):
    try:
        value = input(prompt)
    except EOFError:
        print(f"{prompt} (menggunakan nilai default: {default})")
        return default

    value = value.strip()
    if value == "":
        return default

    if converter is not None:
        return converter(value)
    return value


# 1. Baca data skin, lalu hapus spasi berlebih pada nama kolom dan isi teks.
base_dir = Path(__file__).resolve().parent
candidates = [
    base_dir / "datasets" / "mobile_legends_skins_dataset_1500.csv",
    base_dir / "mobile_legends_skins_dataset_1500.csv",
]

for dataset_path in candidates:
    if dataset_path.exists():
        data = pd.read_csv(dataset_path)
        break
else:
    raise FileNotFoundError(
        "Dataset tidak ditemukan. Pastikan file 'mobile_legends_skins_dataset_1500.csv' ada di folder root atau folder 'datasets'."
    )

data.columns = data.columns.str.strip()
for kolom in data.select_dtypes(exclude="number").columns:
    data[kolom] = data[kolom].str.strip()

# 2. Pilih fitur untuk klasifikasi. Target yang diprediksi: purchase_recommendation.
fitur_angka = ["is_limited", "is_event", "is_collab", "price_idr",
               "popularity_score", "pick_rate_percent", "win_rate_percent"]
X = pd.get_dummies(data[fitur_angka + ["rarity"]], columns=["rarity"])  # rarity diubah jadi angka 0/1
y = data["purchase_recommendation"]

# 3. Sisihkan 20% data untuk menguji model; sisanya dipakai untuk latihan.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Latih model (kedalaman dibatasi agar tidak menghafal data) dan prediksi data uji.
model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
prediksi_uji = model.predict(X_test)
print(f"Akurasi data uji: {model.score(X_test, y_test) * 100:.2f}%")

# 5. Tampilkan jumlah prediksi data uji yang benar dan salah.
benar = (prediksi_uji == y_test).sum()
plt.bar(["Benar", "Salah"], [benar, len(y_test) - benar], color=["seagreen", "tomato"])
plt.ylabel("Jumlah skin pada data uji")
plt.title("Hasil pengujian klasifikasi rekomendasi pembelian skin")
plt.savefig("skin_testing.png", bbox_inches="tight")
plt.close()

# 6. Gambar pohon keputusan supaya alur keputusannya bisa dilihat.
plt.figure(figsize=(22, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, fontsize=8)
plt.savefig("skin_pohon_keputusan.png", bbox_inches="tight")
plt.close()

# 7. Minta data skin baru satu per satu. Jika stdin tidak tersedia (misalnya saat dijalankan dari IDE non-interaktif),
#    gunakan nilai default yang diambil dari dataset agar program tetap berjalan tanpa error.
pilihan_rarity = sorted(data["rarity"].unique())
default_rarity = pilihan_rarity[0]
rarity = input_atau_default(f"Rarity {pilihan_rarity}: ", default_rarity, lambda x: x.strip().title())
while rarity not in pilihan_rarity:
    rarity = input_atau_default(f"Rarity {pilihan_rarity}: ", default_rarity, lambda x: x.strip().title())

sample_row = data.iloc[0].to_dict()
baru = {
    "is_limited": int(input_atau_default("Skin limited? (1 = ya, 0 = tidak): ", int(sample_row.get("is_limited", 0)), int)),
    "is_event": int(input_atau_default("Skin event? (1 = ya, 0 = tidak): ", int(sample_row.get("is_event", 0)), int)),
    "is_collab": int(input_atau_default("Skin collab? (1 = ya, 0 = tidak): ", int(sample_row.get("is_collab", 0)), int)),
    "price_idr": float(input_atau_default("Harga (rupiah): ", float(sample_row.get("price_idr", 0)), float)),
    "popularity_score": float(input_atau_default("Popularity score: ", float(sample_row.get("popularity_score", 0)), float)),
    "pick_rate_percent": float(input_atau_default("Pick rate (%): ", float(sample_row.get("pick_rate_percent", 0)), float)),
    "win_rate_percent": float(input_atau_default("Win rate (%): ", float(sample_row.get("win_rate_percent", 0)), float)),
}

# 8. Samakan format data baru dengan data latih, lalu prediksi dan tampilkan rekomendasinya.
baru_df = pd.get_dummies(pd.DataFrame([baru]).assign(rarity=rarity), columns=["rarity"])
baru_df = baru_df.reindex(columns=X.columns, fill_value=0)
print("Rekomendasi pembelian:", model.predict(baru_df)[0])