# ⚙️ 2. Tools & Library yang Akan Dipakai
Kita akan pakai PyTorch karena fleksibel dan cocok buat eksplorasi dari nol. Berikut tools dasarnya:

## ✅ Library utama:

1. torch → neural network
2. numpy → operasi numerik
3. matplotlib → visualisasi (optional)
4. tqdm → progress bar
5. sentencepiece / tokenizers (kalau mau buat tokenizer sendiri)

## 📦 Tambahan (opsional untuk kemudahan):

1. hydra atau argparse → konfigurasi lebih fleksibel
2. tensorboard → visualisasi training
3. pytest → testing otomatis


# 📚 3. Dataset

Awalnya kita bisa pakai:
1. Dataset kecil dulu: contoh kalimat bahasa Inggris
2. Format: teks biasa atau file .csv/.txt

Contoh data awal (tiny):

```bash
Input: Saya makan nasi
Output: I eat rice
```

A. Dataset Dummy (buat latihan dulu)
Contoh:

```bash
# File: data/raw/sample.txt
saya makan nasi <sep> I eat rice
kamu minum air <sep> you drink water
```

Kita bisa pisahkan input (Bahasa Indonesia) dan target (Bahasa Inggris) berdasarkan pemisah <sep>.

B. Contoh DatasetLoader sederhana (di PyTorch)

```bash
# src/dataset.py
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, filepath, tokenizer_src, tokenizer_tgt, max_len):
        self.pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split("<sep>")
                self.pairs.append((src.strip(), tgt.strip()))
        
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # Tokenisasi + padding
        src_ids = self.tokenizer_src.encode(src, max_length=self.max_len, padding='max_length', truncation=True)
        tgt_ids = self.tokenizer_tgt.encode(tgt, max_length=self.max_len, padding='max_length', truncation=True)

        return {
            "input_ids": src_ids,
            "target_ids": tgt_ids
        }
```

## 🧠 Tujuan Step Dataset Ini:

* Mengubah teks asli jadi angka (token ID)
* Menjamin setiap input/output punya panjang yang sama (pakai padding)
* Supaya bisa dibaca oleh model Transformer (yang hanya bisa baca angka)

Di tahap coding nanti, kita bisa:
* Buat custom Dataset class (PyTorch)
* Tambah preprocessing: padding, masking, tokenisasi, batching


# 🛠️ 4. Konfigurasi (di config.py)

```bash
config = {
    "max_len": 32,                  # Panjang maksimum input/output sequence
    "vocab_size": 8000,             # Ukuran total vocabulary (jumlah kata unik setelah tokenisasi)
    "embedding_dim": 256,           # Ukuran vektor embedding untuk setiap token
    "num_heads": 8,                 # Jumlah 'kepala' attention dalam multi-head attention
    "ffn_dim": 512,                 # Ukuran hidden layer dalam Feed Forward Network
    "num_encoder_layers": 6,        # Jumlah layer encoder
    "num_decoder_layers": 6,        # Jumlah layer decoder
    "dropout": 0.1,                 # Persentase dropout untuk regularisasi
    "batch_size": 32,               # Ukuran mini-batch untuk training
    "learning_rate": 1e-4,          # Learning rate (kecepatan belajar model)
    "epochs": 10,                   # Jumlah epoch (berapa kali semua data dilatih)
    "device": "cuda"                # atau "cpu"
}
```