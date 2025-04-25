# 🧱 Ringkasan Lengkap Arsitektur Transformer (Dari A sampai Z)

## 🔹 1. Input: Kalimat Mentah

User memasukan input, misal: 

```bash
Kalimat asli: "Saya makan nasi"
```

## 🔹 2. Tokenisasi

🔤 Tokenisasi memecah kalimat jadi potongan (token):
* Bisa kata ("Saya", "makan", "nasi")
* Bisa subword (dengan BPE, WordPiece, dll.)

Hasil token:

```bash
["Saya", "makan", "nasi"] → [1012, 4421, 2113]  (token ID)
```

Ini tergantung vocab tokenizer yang dipakai (misal: Hugging Face Tokenizer, SentencePiece, dst.)

## 🔹 3. Penambahan Token Khusus (opsional)

Beberapa model pakai token tambahan seperti:
* <sos> (start of sentence)
* <eos> (end of sentence)
* <pad> (padding)
* <mask> (khusus BERT)

Contoh setelah penambahan:

```bash
["<sos>", "Saya", "makan", "nasi", "<eos>"]
```

## 🔹 4. Embedding Layer

Token ID diubah jadi vektor dengan dimensi tetap, misalnya 512 dimensi. Contoh: 
```bash
"makan" (token ID = 4421) → [0.12, -0.35, ..., 0.48] (512 dimensi)
```

## 🔹 5. Positional Encoding

Dikombinasikan dengan embedding: **Karena Transformer tidak punya urutan bawaan.** Contoh akhir vektor:

```bash
Embedding + Positional Encoding = [0.19, -0.22, ..., 0.61]
```

## 🔹 6. Masuk ke Encoder Stack

Vektor hasil tadi → masuk ke encoder:
* Berisi 6–12 layer (Multi-head attention → Feed Forward → Residual → LayerNorm)

**Output encoder adalah representasi semantik dari seluruh input.**

## 🔹 7. Decoder Stack

Decoder mengambil:
* Output sebelumnya (yang sudah dihasilkan)
* Output dari encoder (untuk fokus ke konteks input)

Lalu:
* Melalui Masked Self-Attention
* Melalui Encoder-Decoder Attention
* Melalui FFN
* Lalu di-normalisasi dan lanjut ke prediksi

## 🔹 8. Linear Layer + Softmax

Output akhir vektor dari decoder → masuk ke:
* Linear Layer: ubah dimensi ke jumlah vocab (misal 30.000 dimensi)
* Softmax: ubah ke probabilitas tiap kata dalam vocab

Contoh hasil softmax:

```bash
"I" = 0.65  
"eat" = 0.10  
"he" = 0.05  
dll...
```

## 🔹 9. Decoding

Prediksi dimasukkan kembali ke decoder (autoregressive):

```bash
Input: "<sos>" → Output: "I"  
Input: "<sos> I" → Output: "eat"  
Input: "<sos> I eat" → Output: "rice"
```

Bisa menggunakan juga:
* Greedy decoding (ambil tertinggi)
* Beam search
* Top-k / nucleus sampling (lebih cocok untuk generatif)

## ✅ Ringkasan Super-Lengkap (Tabel)

| Tahap | Proses                | Output                    |
|-------|-----------------------|---------------------------|
| 1     | Kalimat mentah        | "Saya makan nasi"         |
| 2     | Tokenisasi            | ["Saya", "makan", "nasi"] |
| 3     | Token ID              | [1012, 4421, 2113]        |
| 4     | Embedding             | Vektor 512 dimensi        |
| 5     | Positional Encoding   | + urutan vektor           |
| 6     | Encoder Stack         | Representasi konteks      |
| 7     | Decoder Stack         | Prediksi token            |
| 8     | Linear + Softmax      | Probabilitas kata         |
| 9     | Decoding              | "I eat rice"              |