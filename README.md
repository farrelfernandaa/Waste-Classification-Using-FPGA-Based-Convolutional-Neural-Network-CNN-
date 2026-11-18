# Waste Classification Using FPGA-Based Convolutional Neural Network (CNN)
*(Klasifikasi Jenis Sampah Menggunakan Convolutional Neural Network Berbasis FPGA)*

![Platform](https://img.shields.io/badge/Hardware-FPGA_(Verilog_RTL)-blue)
![Framework](https://img.shields.io/badge/Training-Python_/_TensorFlow-orange)
![Quantization](https://img.shields.io/badge/Precision-8--bit_Fixed--Point_(INT8)-green)
![Dataset](https://img.shields.io/badge/Dataset-TrashNet_(3_Classes)-lightgrey)

> **Identitas Mahasiswa:**
> - **Nama:** Farrel Fernanda
> - **NIM:** *1102223030*
> - **Mata Kuliah:** *Desain SoC dan FPGA*

---

## 1. Deskripsi Proyek

Proyek ini merancang akselerator perangkat keras *Quantized Tiny-CNN* berbasis **Field-Programmable Gate Array (FPGA)** menggunakan bahasa deskripsi perangkat keras **Verilog RTL** untuk mengklasifikasikan tiga jenis sampah anorganik utama, yaitu **Plastik (*Plastic*), Kertas/Kardus (*Paper*), dan Logam (*Metal*)**. 

Sistem dibangun menggunakan pendekatan *hardware-software co-design*:
- **Sisi Perangkat Lunak (Python/TensorFlow):** Pelatihan model *Tiny-CNN* dilakukan secara *offline* menggunakan dataset [TrashNet (`garythung/trashnet`)](https://huggingface.co/datasets/garythung/trashnet) dengan resolusi masukan **32×32 piksel *grayscale***, kemudian seluruh bobot (*weights*) dan *bias* dikuantisasi menjadi bilangan bulat bertanda 8-bit (**INT8**).
- **Sisi Perangkat Keras (FPGA Verilog RTL):** Proses *inference* dieksekusi langsung di level gerbang logika menggunakan memori internal FPGA (*Block RAM*) untuk mencapai waktu respons yang deterministik, berlatensi rendah (*ultra-low latency*), dan hemat daya tanpa ketergantungan pada komputasi *cloud*.

---

## 2. Struktur Repositori

    ├── README.md                  # Dokumentasi utama proyek, arsitektur, dan hasil verifikasi
    ├── system_architecture.png    # Diagram blok arsitektur perangkat keras (Input-Process-Output)
    ├── train_and_quantize.py      # Skrip training CNN (TrashNet) & kuantisasi bobot ke INT8
    ├── weights_sample.mem         # Contoh file bobot heksadesimal untuk inisialisasi ROM Verilog
    ├── mac_unit.v                 # Modul Multiply-Accumulate (MAC) 3x3 8-bit signed
    ├── relu_activation.v          # Modul fungsi aktivasi ReLU 32-bit
    ├── maxpool_2x2.v              # Modul Max-Pooling 2x2 komparator
    └── tb_cnn_blocks.v            # Testbench simulasi untuk verifikasi fungsional blok RTL

---

## 3. Arsitektur Model & Perangkat Keras

### A. Diagram Blok Sistem (*Input – Process – Output*)

![Diagram Arsitektur Sistem](system_architecture.png)

1. **Blok Input (`input_interface`):** Menerima citra sampah 32×32 piksel *grayscale* (1.024 piksel) yang dikuantisasi ke format `signed [7:0]` (INT8) ke dalam *Input Buffer* BRAM.
2. **Blok Process (`cnn_accelerator_core`):**
   - **Weight & Bias ROM:** Menyimpan 1.203 parameter hasil *training* dalam format heksadesimal (`.mem`).
   - **Line Buffer & Sliding Window 3×3:** Membentuk jendela konvolusi 3×3 secara berurutan.
   - **MAC Array (`mac_unit.v`):** Menghitung perkalian paralel 9 piksel dengan 9 bobot kernel beserta *bias* menggunakan akumulator 32-bit (`signed [31:0]`).
   - **ReLU & Max-Pooling (`relu_activation.v`, `maxpool_2x2.v`):** Melakukan aktivasi non-linear dan reduksi dimensi fitur 2×2.
   - **Fully Connected & Control FSM:** Menghitung skor akhir ketiga kelas dan mengatur sinkronisasi perpindahan *layer*.
3. **Blok Output (`output_controller`):** Unit komparator **Argmax** menghasilkan kode keputusan 2-bit (`00`: Plastik, `01`: Kertas, `10`: Logam) untuk menggerakkan sinyal PWM motor servo pemilah serta indikator LED/7-Segment pada *board* FPGA.

### B. Spesifikasi Layer & Kebutuhan Memori Parameter

| Tahap / Layer | Dimensi Output | Ukuran Kernel | Jumlah Filter | Jumlah Parameter | Memori Bobot (INT8) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Image** | 32 × 32 × 1 | - | - | 0 | 1.024 Byte (BRAM) |
| **Conv1 + ReLU** | 30 × 30 × 4 | 3 × 3 | 4 | 40 | 40 Byte |
| **MaxPool 1** | 15 × 15 × 4 | 2 × 2 | - | 0 | - |
| **Conv2 + ReLU** | 13 × 13 × 8 | 3 × 3 | 8 | 296 | 296 Byte |
| **MaxPool 2** | 6 × 6 × 8 | 2 × 2 | - | 0 | - |
| **Flatten** | 288 | - | - | 0 | - |
| **Fully Connected** | 3 (Kelas) | - | - | 867 | 867 Byte |
| **TOTAL** | - | - | - | **1.203 Parameter** | **± 1,2 KB Bobot** |

---

## 4. Ringkasan Hasil Pengujian & Sintesis

Pada tahap saat ini, pengujian difokuskan pada **verifikasi fungsional level RTL (*Behavioral Simulation*)** menggunakan *testbench* `tb_cnn_blocks.v` serta analisis efisiensi sumber daya memori:

| Komponen / Modul Uji | Metode Pengujian | Hasil Verifikasi & Analisis Teknis | Status |
| :--- | :--- | :--- | :---: |
| **`mac_unit.v`** | Simulasi Testbench (`INT8 × INT8 + Bias`) | Operasi konvolusi 3×3 berjalan deterministik dalam **1 siklus *clock*** dengan akumulator 32-bit tanpa *overflow*. | **PASS** |
| **`relu_activation.v`** | Simulasi Testbench (Input Positif & Negatif) | Sinyal masukan bertanda negatif (`MSB = 1`) berhasil dipotong menjadi `0` secara akurat. | **PASS** |
| **`maxpool_2x2.v`** | Simulasi Testbench (4 Input Nilai Fitur) | Komparator berhasil memilih nilai maksimum dari jendela 2×2 secara tepat. | **PASS** |
| **Kapasitas Memori Model** | Kalkulasi Arsitektur Kuantisasi INT8 | Total bobot hanya **1.203 Byte (±1,2 KB)** + buffer input **1 KB**, seluruhnya muat di dalam **BRAM internal FPGA** tanpa memerlukan DRAM eksternal. | **VERIFIED** |
| **Sintesis Bitstream & Board** | Implementasi Top-Level (`cnn_top.v`) | *Dijadwalkan pada tahap implementasi penuh ke board FPGA (UAS).* | **PLANNED** |

---

## 5. Dokumentasi Pengujian

### A. Menjalankan *Training* & Kuantisasi Model (Python)
Untuk melatih ulang model pada dataset TrashNet dan mengekspor bobot terkuantisasi ke format `.mem`:

    pip install tensorflow datasets pillow numpy
    python train_and_quantize.py

### B. Menjalankan Simulasi Modul RTL (Icarus Verilog)
Untuk mereproduksi hasil verifikasi *testbench* blok `MAC`, `ReLU`, dan `Max-Pooling`:

    iverilog -o sim_cnn mac_unit.v relu_activation.v maxpool_2x2.v tb_cnn_blocks.v
    vvp sim_cnn
