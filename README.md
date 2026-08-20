# 🍎 Klasifikasi Gambar Buah Segar & Busuk dengan CNN

Model *Deep Learning* berbasis **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar buah **segar (fresh)** dan **busuk (rotten)** ke dalam 6 kelas: apel, pisang, dan jeruk — masing-masing dalam kondisi segar maupun busuk.

## 🎯 Kelas Klasifikasi

Model dilatih untuk mengenali **6 kelas** gambar buah:

| Kelas | Keterangan |
|---|---|
| `freshapples` | Apel segar |
| `rottenapples` | Apel busuk |
| `freshbanana` | Pisang segar |
| `rottenbanana` | Pisang busuk |
| `freshoranges` | Jeruk segar |
| `rottenoranges` | Jeruk busuk |

## 📊 Hasil & Performa Model

| Metrik | Akurasi |
|---|---|
| **Test Set** | **98.80%** |
| **Training Set** | **97.05%** |

Model dilatih selama maksimal 25 epoch dengan `EarlyStopping` dan `ReduceLROnPlateau`, sehingga pelatihan berhenti otomatis begitu performa validasi tidak lagi meningkat.

## 📊 Dataset

Dataset yang digunakan adalah **[Fresh and Stale Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)** dari Kaggle, diunduh otomatis melalui `kagglehub`. Data awal (train & test bawaan dataset) digabungkan kembali lalu di-split ulang secara *stratified* menjadi:

- **70%** — Data Training
- **20%** — Data Validation
- **10%** — Data Test

## 🧠 Arsitektur Model

Model CNN dibangun secara sequential dengan 3 blok Convolutional, sebagai berikut:

```
Input (150x150x3)
 ├── Conv2D(32) → MaxPooling2D → BatchNormalization
 ├── Conv2D(64) → MaxPooling2D → BatchNormalization
 ├── Conv2D(128) → MaxPooling2D → BatchNormalization
 ├── Flatten
 ├── Dense(256, relu) → Dropout(0.5)
 └── Dense(6, softmax)
```

- **Optimizer**: Adam (`learning_rate=0.0005`)
- **Loss Function**: Categorical Crossentropy
- **Augmentasi Data**: rotasi, pergeseran lebar/tinggi, zoom, dan flip horizontal (untuk data training) guna mengurangi overfitting

## 📂 Struktur Project

```
Klasifikasi-Gambar-Buah-CNN-/
├── Submission_Akhir_Fundamental_Deep_Learning.ipynb   # Notebook utama: data prep, training, evaluasi, konversi model
├── requirements.txt                                   # Daftar dependency Python
├── saved_model/                                        # Model dalam format TensorFlow SavedModel
├── tflite/
│   ├── model.tflite                                    # Model dalam format TensorFlow Lite (untuk mobile/edge)
│   └── label.txt                                        # Daftar label kelas
├── tfjs_model/                                          # Model dalam format TensorFlow.js (untuk web)
└── README.md
```

Model disediakan dalam **3 format berbeda** agar bisa digunakan di berbagai platform:
- **SavedModel** — untuk deployment di server/Python (TensorFlow Serving, dsb.)
- **TFLite** — untuk aplikasi mobile & perangkat edge
- **TensorFlow.js** — untuk dijalankan langsung di browser

## 🛠️ Tech Stack

- **Python** & **TensorFlow / Keras** — pembangunan & pelatihan model CNN
- **TensorFlow.js** (`tensorflowjs`) — konversi model ke format web
- **Scikit-learn** — *stratified split* dataset
- **Matplotlib** — visualisasi kurva akurasi & loss
- **KaggleHub** — pengunduhan dataset
- **NumPy**

## ▶️ Cara Menjalankan

1. **Clone repository**
   ```sh
   git clone https://github.com/alfdmsr/Klasifikasi-Gambar-Buah-CNN-.git
   cd Klasifikasi-Gambar-Buah-CNN-
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Jalankan notebook**
   ```sh
   jupyter notebook Submission_Akhir_Fundamental_Deep_Learning.ipynb
   ```
   Jalankan seluruh cell secara berurutan — dataset akan otomatis terunduh dari Kaggle melalui `kagglehub`, dilanjutkan dengan preprocessing, training, evaluasi, hingga konversi model ke 3 format (SavedModel, TFLite, TensorFlow.js).

4. **Menggunakan model yang sudah dilatih**

   Tidak perlu training ulang — model hasil pelatihan sudah tersedia langsung di repository ini pada folder `saved_model/`, `tflite/`, dan `tfjs_model/`, siap dipakai untuk inferensi.

   Contoh inferensi menggunakan model TFLite:
   ```python
   import tensorflow as tf

   interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
   interpreter.allocate_tensors()
   # ... siapkan gambar input berukuran 150x150x3, lalu jalankan interpreter.invoke()
   ```

## 📌 Catatan

Proyek ini dibuat untuk tujuan pembelajaran *Computer Vision* dengan CNN, sekaligus melatih proses *deployment-ready* dengan mengonversi model ke berbagai format agar siap dipakai di lingkungan produksi (web, mobile, maupun server).