# Klasifikasi Jenis Sampah Menggunakan Convolutional Neural Network (CNN) Berbasis FPGA

![Platform](https://img.shields.io/badge/Hardware-FPGA_(Verilog_RTL)-blue)
![Framework](https://img.shields.io/badge/Training-Python_/_TensorFlow-orange)
![Quantization](https://img.shields.io/badge/Precision-8--bit_Fixed--Point_(INT8)-green)
![Dataset](https://img.shields.io/badge/Dataset-TrashNet_(3_Classes)-lightgrey)

> **Identitas Mahasiswa (Isi sebelum dikumpulkan):**
> - **Nama:** Farrel Fernanda
> - **NIM:** 1102223030

---

## Abstrak

Pemilahan sampah anorganik secara otomatis di tingkat hulu membutuhkan sistem klasifikasi visual yang cepat, hemat daya, dan mampu beroperasi secara mandiri tanpa ketergantungan pada komputasi *cloud*. Implementasi *Convolutional Neural Network* (CNN) pada arsitektur prosesor konvensional seperti CPU atau GPU umumnya memerlukan konsumsi daya besar serta memiliki latensi tinggi ketika diterapkan pada sistem tertanam (*embedded*). Penelitian ini mengusulkan perancangan akselerator perangkat keras *Quantized Tiny-CNN* berbasis *Field-Programmable Gate Array* (FPGA) menggunakan bahasa deskripsi perangkat keras Verilog RTL untuk mengklasifikasikan tiga jenis sampah kemasan utama, yaitu plastik, kertas/kardus, dan logam. Sistem dirancang menggunakan pendekatan *hardware-software co-design*, di mana proses pelatihan model dilakukan secara *offline* menggunakan dataset TrashNet dengan masukan citra *grayscale* berukuran 32×32 piksel, kemudian seluruh bobot (*weights*) dan *bias* dikuantisasi ke dalam format bilangan bulat bertanda 8-bit (*INT8*). Arsitektur perangkat keras pada FPGA dibagi menjadi tiga blok utama yang terdiri atas blok *Input* (*Input Buffer* BRAM dan antarmuka data), blok *Process* (*Weight ROM*, *Line Buffer*, *Sliding Window* 3×3, *Multiply-Accumulate Array*, unit aktivasi ReLU, *Max-Pooling* 2×2, *Fully Connected Layer*, serta *Control Finite State Machine*), dan blok *Output* (unit pembanding *Argmax* serta pembangkit sinyal kontrol aktuator pemilah). Hasil perancangan arsitektur menunjukkan bahwa model jaringan hanya memerlukan 1.203 parameter dengan kebutuhan memori bobot sebesar ±1,2 KB sehingga dapat disimpan sepenuhnya pada memori internal FPGA dengan latensi *inference* yang deterministik dan rendah.

**Kata Kunci:** *Convolutional Neural Network* (CNN), FPGA, Klasifikasi Sampah, Kuantisasi 8-Bit, Verilog RTL.

---

## 1. Latar Belakang & Permasalahan
Pemilahan sampah di tingkat hulu merupakan langkah krusial untuk meningkatkan efektivitas daur ulang[cite: 2]. Sistem klasifikasi sampah berbasis kecerdasan buatan (*Artificial Intelligence*) umumnya diimplementasikan pada perangkat berbasis CPU/GPU (seperti PC, Raspberry Pi, atau *cloud server*)[cite: 2]. Namun, pendekatan tersebut memiliki keterbatasan pada skenario sistem tertanam (*embedded*) dan otomasi industri[cite: 2]:
- **Latensi Tinggi:** Ketergantungan pada komunikasi jaringan (*cloud*) atau eksekusi sekuensial pada mikrokontroler standar memperlambat respons pemilahan mekanis (misalnya pada jalur konveyor)[cite: 2].
- **Konsumsi Daya Besar:** Penggunaan GPU atau SBC (*Single-Board Computer*) bertenaga tinggi kurang efisien untuk perangkat tempat sampah pintar mandiri (*standalone*)[cite: 2].

Proyek ini merancang **akselerator perangkat keras *Quantized Tiny-CNN* berbasis FPGA** menggunakan **Verilog RTL** untuk melakukan klasifikasi jenis sampah secara paralel, deterministik, berlatensi sangat rendah (*ultra-low latency*), dan hemat daya[cite: 2].

---

## 2. Evaluasi Kriteria Ide (Rubrik UTS)

### A. Keunikan & Kebermanfaatan Ide
- **Kebermanfaatan Nyata:** Memberikan solusi pemilahan otomatis sampah anorganik bernilai ekonomi (**Plastik, Kertas/Kardus, dan Logam**) untuk tempat sampah pintar maupun jalur sortir skala kecil[cite: 2].
- **Keunikan Teknis:** Alih-alih sekadar membuat model *deep learning* di sisi *software*, proyek ini memindahkan proses *inference* CNN langsung ke level gerbang logika (*hardware RTL* Verilog) dengan teknik **kuantisasi 8-bit integer (INT8)** agar arsitektur jaringan saraf tiruan dapat berjalan mandiri tanpa ketergantungan sistem operasi maupun internet[cite: 2].

### B. Kematangan Ide (Status Realisasi & Kelayakan Teknis)
- **Kelayakan Implementasi (*Hardware-Software Co-Design*):**
  - Proses *training* dilakukan secara *offline* di PC menggunakan Python untuk menghindari kompleksitas komputasi *backpropagation* di perangkat keras[cite: 2].
  - Bobot (*weights*) dan *bias* hasil *training* dikuantisasi menjadi **8-bit signed integer (`signed [7:0]`)** sehingga total kebutuhan memori parameter hanya **± 1,2 KB (1.203 parameter)**—sangat ringan dan muat sepenuhnya di dalam *Block RAM* (BRAM) internal FPGA kelas pemula sekalipun[cite: 2].
- **Status Pengerjaan Saat Ini (Tahap UTS):**
  - [x] Perancangan arsitektur *Tiny-CNN* ramah perangkat keras (input citra 32×32 *grayscale*)[cite: 2].
  - [x] Penyiapan pipeline *dataset* (**TrashNet**) dan skrip kuantisasi INT8 ke format memori hex (`.mem`)[cite: 2].
  - [x] Perancangan diagram blok sistem (*Input – Process – Output*) dan arsitektur RTL[cite: 2].
  - [x] Implementasi modul dasar RTL Verilog (`mac_unit.v`, `relu_activation.v`, `maxpool_2x2.v`) beserta *testbench* verifikasi (`tb_cnn_blocks.v`)[cite: 2].
  - [ ] *(Tahap UAS)* Integrasi penuh *Finite State Machine* (`cnn_top.v`), simulasi *bit-accurate* menyeluruh, dan sintesis ke *board* FPGA[cite: 2].

---

## 3. Arsitektur Sistem & Diagram Blok (Input – Process – Output)

![Diagram Arsitektur Sistem](system_architecture.png)

### Penjelasan Rinci Setiap Blok

1. **Blok Input (`input_interface`)**[cite: 2]
   - **Sumber Data:** Citra sampah yang telah di-*resize* menjadi **32×32 piksel** dan dikonversi ke **1 kanal *grayscale*** (total 1.024 piksel) untuk menghemat kapasitas memori BRAM hingga 66% dibandingkan citra RGB[cite: 2].
   - **Representasi Data:** Nilai piksel dikuantisasi ke format bilangan bulat bertanda 8-bit (`signed [7:0]`) dan dimuat ke *Input Buffer* melalui antarmuka UART RX atau inisialisasi memori (`$readmemh`)[cite: 2].

2. **Blok Process (`cnn_accelerator_core`)**[cite: 2]
   - **Weight & Bias ROM:** Menyimpan seluruh parameter hasil *training* dalam format heksadesimal (`.mem`)[cite: 2].
   - **Line Buffer & Sliding Window 3×3:** Menggeser aliran piksel secara *streaming* untuk membentuk jendela matriks 3×3 tanpa harus membaca ulang memori utama secara berulang[cite: 2].
   - **MAC Array (*Multiply-Accumulate*):** Melakukan perkalian paralel antara 9 piksel jendela input dengan 9 bobot kernel konvolusi menggunakan akumulator 32-bit (`signed [31:0]`) untuk mencegah *overflow*[cite: 2].
   - **ReLU & Max-Pooling Unit:**[cite: 2]
     - *ReLU* diimplementasikan menggunakan gerbang logika sederhana yang memeriksa bit tanda MSB (`data[31] ? 0 : data`)[cite: 2].
     - *Max-Pooling 2×2* menggunakan komparator untuk mengambil nilai maksimum dari setiap blok 2×2 fitur guna mereduksi dimensi spasial[cite: 2].
   - **Fully Connected (FC) Layer:** Menghitung akumulasi skor akhir untuk masing-masing dari 3 kelas target[cite: 2].
   - **Control FSM (*Finite State Machine*):** Mengatur sinkronisasi alamat memori, sinyal *enable*, dan transisi antar-*layer* (`IDLE` → `CONV1` → `POOL1` → `CONV2` → `POOL2` → `FC` → `DONE`)[cite: 2].

3. **Blok Output (`output_controller`)**[cite: 2]
   - **Argmax Unit:** Membandingkan 3 skor keluaran dari *Fully Connected Layer* dan menghasilkan kode kelas 2-bit[cite: 2]:
     - `00` : **Plastik (*Plastic*)**[cite: 2]
     - `01` : **Kertas / Kardus (*Paper*)**[cite: 2]
     - `10` : **Logam (*Metal*)**[cite: 2]
   - **Aktuator & Antarmuka:** Menggerakkan motor servo pemilah menggunakan sinyal PWM sesuai kategori sampah, menyalakan indikator LED/7-Segment pada *board* FPGA, serta mengirimkan hasil klasifikasi kembali ke PC via UART TX[cite: 2, 5].

---

## 4. Arsitektur Model *Tiny-CNN* & Estimasi Sumber Daya FPGA

| Tahap / Layer | Dimensi Output | Ukuran Kernel | Jumlah Filter | Jumlah Parameter | Memori Bobot (INT8) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Image** | 32 × 32 × 1 | - | - | 0 | 1.024 Byte (BRAM)[cite: 2] |
| **Conv1 + ReLU** | 30 × 30 × 4 | 3 × 3 | 4 | 40 | 40 Byte[cite: 2] |
| **MaxPool 1** | 15 × 15 × 4 | 2 × 2 | - | 0 | -[cite: 2] |
| **Conv2 + ReLU** | 13 × 13 × 8 | 3 × 3 | 8 | 296 | 296 Byte[cite: 2] |
| **MaxPool 2** | 6 × 6 × 8 | 2 × 2 | - | 0 | -[cite: 2] |
| **Flatten** | 288 | - | - | 0 | -[cite: 2] |
| **Fully Connected** | 3 (Kelas) | - | - | 867 | 867 Byte[cite: 2] |
| **TOTAL** | - | - | - | **1.203 Parameter** | **± 1,2 KB Bobot**[cite: 2] |

> **Kesimpulan Kelayakan Hardware:** Dengan total hanya **1.203 parameter (±1,2 KB)**, seluruh bobot jaringan dapat disimpan sepenuhnya di dalam memori internal FPGA tanpa memerlukan DRAM eksternal[cite: 2].

---

## 5. Daftar File dalam Repositori

```text
├── README.md                  # Dokumentasi utama proposal & arsitektur sistem (UTS)
├── system_architecture.png    # Diagram blok arsitektur Input-Process-Output
├── train_and_quantize.py      # Skrip training CNN (TrashNet) & kuantisasi bobot ke INT8
├── weights_sample.mem         # Contoh file bobot heksadesimal untuk ROM Verilog
├── mac_unit.v                 # Modul Multiply-Accumulate (MAC) 3x3 8-bit signed
├── relu_activation.v          # Modul fungsi aktivasi ReLU 32-bit
├── maxpool_2x2.v              # Modul Max-Pooling 2x2 komparator
└── tb_cnn_blocks.v            # Testbench simulasi untuk verifikasi modul dasar
