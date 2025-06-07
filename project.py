!pip install Sastrawi
!pip install Sastrawi matplotlib
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
nltk.download('all')
print("Klasifikasi sentimen dari ulasan pelanggan produk Parfum Dior untuk memahami persepsi konsumen dan membantu strategi pemasaran perusahaan")
file_path = 'review_parfum.csv'
df = pd.read_csv(file_path)
print('contoh 5 baris data awal')
print(df.head())
print("-"*30)
#Pelabelan Sentimen Berdasarkan Kata Kunci (Rule-Based)
kata_kunci_positif = [
    "bagus", "suka", "cocok", "wangi", "puas", "mantap", "terbaik", "rekomendasi",
    "keren", "enak", "cepat", "bonus", "ori", "original", "asli", "ampuh",
    "efektif", "lembut", "halus", "tidak mengecewakan", "sesuai", "oke",
    "ok", "baik", "thank you", "thanks", "terima kasih", "ga nyesel", "ga nyesel",
    "recommended", "cepet", "bgt", "banget" "sangat wangi""tahan lama"
# 'bgt'/'banget' sering muncul dengan kata positif
]
kata_kunci_negatif = [
    "tidak suka", "kecewa", "jelek", "buruk", "masalah", "lama", "bau",
    "tidak cocok", "jangan beli", "rugi", "mahal", "palsu", "iritasi", "lengket",
    "susah", "pecah", "rusak", "bocor", "kurang", "gak", "enggak", "nggak",
    "bukan", "tdk", "ga cocok", "ga suka", "gk", "tdk cocok", "mengecewakan",
    "berantakan", "lambat","bikin break out"
]
label_sentimen_berdasarkan_aturan = []
for ulasan in df['Review']:
    ulasan_lower = str(ulasan).lower() 
# Ubah ke huruf kecil dan string

    ada_positif = any(kata in ulasan_lower for kata in kata_kunci_positif)
    ada_negatif = any(kata in ulasan_lower for kata in kata_kunci_negatif)
    if ada_positif and not ada_negatif:
            label_sentimen_berdasarkan_aturan.append('positif')
    elif ada_negatif:
            label_sentimen_berdasarkan_aturan.append('negatif')
    else:
            label_sentimen_berdasarkan_aturan.append('positif') 
# Default jika tidak ada keyword jelas

df['sentiment'] = label_sentimen_berdasarkan_aturan
print("\nDistribusi sentimen hasil aturan:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
# Visualisasi Distribusi Sentimen
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribusi Sentimen Ulasan (Positif vs Negatif)')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Ulasan')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()
 #Pra-pemrosesan Teks (Text Preprocessing)
print("Tahap Pra-pemrosesan Teks")

# --- PENGATURAN STOPWORDS MENGGUNAKAN SASTRAWI ---
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory_sw = StopWordRemoverFactory()
stop_words_list = factory_sw.get_stop_words()
print(f"Jumlah stopwords Bahasa Indonesia dari Sastrawi: {len(stop_words_list)}")
# --- PENGATURAN STEMMER MENGGUNAKAN SASTRAWI ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
def _internal_preprocess_text(text):
  text = str(text).lower()
  text = re.sub(r'[^a-z\s]', '', text)
 # Hanya mempertahankan huruf dan spasi
  tokens = word_tokenize(text)

  processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_list and word.strip() != '']
  return " ".join(processed_tokens)

df['processed_review'] = df['Review'].apply(_internal_preprocess_text)
print("\nContoh ulasan setelah pra-pemrosesan menggunakan Sastrawi:")
print(df[['Review', 'processed_review']].head())
 #Transformasi Teks: Term Document Matrix (TDM)
print("Tahap Tranformasi Teks : Term Document Matrix dengan TF-IDF")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_review'])
feature_names_out = vectorizer.get_feature_names_out()
print(f"Bentuk TDM (jumlah ulasan, jumlah fitur/term): {X.shape}")
df['sentiment_label'] = df['sentiment'].map({'positif': 1, 'negatif': 0})
y = df['sentiment_label']
# Cek apakah ada nilai NaN di y setelah mapping (jika ada sentimen yang tidak 'positif'/'negatif')
if y.isnull().any():
# Hapus baris dimana y adalah NaN, dan juga X yang sesuai
    nan_indices = y.index[y.isnull()]
    X = X[~y.isnull().values]
    y = y.dropna()
    df = df.loc[y.index]
    print(f"Jumlah sampel setelah menghapus NaN: {len(y)}")
#Pemilihan Fitur (Feature Selection)
print("Tahap Pemilihan Fitur")
print("Fitur dipilih berdasarkan skor TF-IDF dan parameter `max_features` pada TfidfVectorizer.")
print(f"Jumlah fitur yang dipilih: {X.shape[1]}")
#Pembagian Data (Data Splitting menjadi Train dan Test)
print("Tahap Pembagian Data (Train dan Test)")
# Pastikan y tidak kosong setelah potensi dropna
if len(y) == 0:
    print("Error: Tidak ada data label yang valid untuk melanjutkan. Proses dihentikan.")
    exit()
elif len(y) < 2 : 
 # atau jumlah lain yang terlalu kecil untuk dibagi
    print("Error: Jumlah data label terlalu sedikit untuk dibagi menjadi train dan test. Proses dihentikan.")
    exit()
# Cek apakah ada cukup sampel di setiap kelas untuk stratify
value_counts_y = y.value_counts()
if len(value_counts_y) < 2 or value_counts_y.min() < 2 : # Minimal 2 sampel per kelas untuk stratify train/test split
# Jika hanya satu kelas, stratify akan error. Jika >1 tapi salah satu kelas <2, bisa error/warning.
    if len(value_counts_y) < 2:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Jumlah data training: {X_train.shape[0]} sampel")
print(f"Jumlah data testing: {X_test.shape[0]} sampel")
print("-" * 30)
# 7. Pelatihan Model Decision Tree
print("Tahap Pelatihan Model Decision Tree")

model = DecisionTreeClassifier(random_state=42, max_depth=10)
 # Batasi kedalaman untuk visualisasi
model.fit(X_train, y_train)
#Membuat Prediksi dan Evaluasi Akurasi
print("Tahap Evaluasi Model")
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Akurasi pada data Training: {train_accuracy:.4f}")

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Akurasi pada data Testing: {test_accuracy:.4f}")
print("-" * 30)
 # 9. Visualisasi Decision Tree
print("Tahap Visualisasi Decision Tree")
plt.figure(figsize=(20,10))
# Nama kelas: 0 untuk 'negatif', 1 untuk 'positif'
class_names_tree = [str(cls) for cls in model.classes_]
 # Menggunakan pemetaan eksplisit jika nama kelas dari model adalah numerik
class_names_display = ['negatif' if i == 0 else 'positif' for i in model.classes_]
from sklearn.tree import DecisionTreeClassifier, plot_tree
plot_tree(model,
          feature_names=feature_names_out,
          class_names=class_names_display,
          filled=True,
          rounded=True,
          fontsize=7,
          max_depth=3)
plt.title(f"Visualisasi Decision Tree (max_depth={model.get_params()['max_depth']}, plotted_depth=3)")
plt.show()
print("Catatan: Visualisasi Decision Tree di atas mungkin hanya menampilkan sebagian kedalaman pohon.")
output_df = df[['Review', 'processed_review', 'sentiment', 'sentiment_label']]
output_filename = "analisis_review_scarlett.csv"
output_df.to_csv(output_filename, index=False)
print(f"\nData yang telah diproses disimpan ke '{output_filename}'")

#Sahrul Mahdi Muhammad
#20230040137
