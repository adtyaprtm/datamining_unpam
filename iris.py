from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

base_dir = Path(__file__).resolve().parent

# 1. Ambil dataset iris bawaan sklearn
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2. Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Latih model Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Prediksi dan hitung akurasi
prediksi = model.predict(X_test)
akurasi = model.score(X_test, y_test)
print(f"Akurasi data uji iris: {akurasi * 100:.2f}%")

# 5. Buat plot hasil benar-salah
benar = (prediksi == y_test).sum()
salah = len(y_test) - benar

plt.bar(["Benar", "Salah"], [benar, salah], color=["seagreen", "tomato"])
plt.ylabel("Jumlah data uji")
plt.title("Hasil Pengujian Klasifikasi Iris")
plt.savefig(base_dir / "iris_testing.png", bbox_inches="tight")
plt.close()

# 6. Buat plot pohon keputusan
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=feature_names, class_names=iris.target_names, filled=True, fontsize=9)
plt.title("Pohon Keputusan Iris")
plt.savefig(base_dir / "iris_decision_tree.png", bbox_inches="tight")
plt.close()

# 7. Cetak hasil prediksi satu contoh paling sederhana
contoh = X_test[0]
label_prediksi = model.predict([contoh])[0]
print(f"Contoh data uji pertama diprediksi sebagai: {iris.target_names[label_prediksi]}")
print("File gambar sudah dibuat: iris_testing.png dan iris_decision_tree.png")