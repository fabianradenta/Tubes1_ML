# Tubes1_ML

Repository ini berisi implementasi dan eksperimen untuk membangun dan menguji Feedforward Neural Network (FFNN) sederhana menggunakan Python. Proyek ini dibuat sebagai bagian dari tugas besar mata kuliah Machine Learning.

## Deskripsi
Proyek ini mencakup:
- Implementasi FFNN dari awal tanpa library machine learning.
- Eksperimen dengan berbagai parameter seperti jumlah layer, fungsi aktivasi, dan learning rate.
- Visualisasi struktur jaringan, distribusi bobot, dan gradien.
- Perbandingan performa dengan library scikit-learn.

## Cara Setup
1. Clone repository ini:
   ```bash
   git clone https://github.com/fabianradenta/Tubes1_ML
   cd Tubes1_ML
2. Buat virtual environment dan aktifkan:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    venv\Scripts\activate     # Untuk Windows
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Pastikan dataset MNIST tersedia atau unduh secara otomatis saat program dijalankan.

## Cara Menjalankan Program
1. Untuk menjalankan eksperimen dan visualisasi:
    ```bash
    run test_ffnn.ipynb

2. Untuk menjalankan visualisasi struktur jaringan:
    ```bash
    run test_ffnn_visualizations.ipynb


## Pembagian Tugas
13522001: Visualisasi (struktur jaringan, distribusi bobot, distribusi gradien). Dokumentasi dan perbandingan dengan scikit-learn. 
13522105: Implementasi FFNN (forward pass, backward pass, update weights). Eksperimen parameter (jumlah layer, fungsi aktivasi, learning rate).
