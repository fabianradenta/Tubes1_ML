# Tubes1_ML
> *Source code* ini dibuat untuk memenuhi Tugas Besar 1 Mata Kuliah IF3270 Pembelajaran Mesin.

## Deskripsi Singkat
Repository ini berisi implementasi dan eksperimen untuk membangun dan menguji Feed Forward Neural Network (FFNN) sederhana menggunakan bahasa pemrograman Python. Proyek ini mencakup:
- Implementasi FFNN dari awal tanpa library machine learning.
- Eksperimen dengan berbagai parameter seperti jumlah layer, fungsi aktivasi, dan learning rate.
- Visualisasi struktur jaringan, distribusi bobot, dan gradien.
- Perbandingan performa dengan library scikit-learn.

## Cara Setup dan Menjalankan Program
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
    pip install -r src/requirements.txt
4. Jalankan program dengan Run All pada notebook `test_ffnn.ipynb`

## Pembagian Tugas
| **No** | **NIM**     | **Tugas**                                                                 |
|--------|-------------|---------------------------------------------------------------------------|
| 1      | 13522001    | - Visualisasi (struktur jaringan, distribusi bobot, distribusi gradien).  |
|        |             | - Dokumentasi dan perbandingan dengan scikit-learn.                       |
| 2      | 13522105    | - Implementasi FFNN (forward pass, backward pass, update weights).        |
|        |             | - Eksperimen parameter (jumlah layer, fungsi aktivasi, learning rate).    |
