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

```text
├── README.md                  # Dokumentasi utama proyek, arsitektur, dan hasil verifikasi
├── system_architecture.png    # Diagram blok arsitektur perangkat keras (Input-Process-Output)
├── train_and_quantize.py      # Skrip training CNN (TrashNet) & kuantisasi bobot ke INT8
├── weights_sample.mem         # Contoh file bobot heksadesimal untuk inisialisasi ROM Verilog
├── mac_unit.v                 # Modul Multiply-Accumulate (MAC) 3x3 8-bit signed
├── relu_activation.v          # Modul fungsi aktivasi ReLU 32-bit
├── maxpool_2x2.v              # Modul Max-Pooling 2x2 komparator
└── tb_cnn_blocks.v            # Testbench simulasi untuk verifikasi fungsional blok RTL
