import torch
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    """
    =====================
     initialize sections
    =====================
    ✅ nn.Module adalah class dasar semua layer PyTorch

    ✅ Ini self-attention (Q=K=V dari input yang sama).
        * embed_dim = dimensi vektor input
        * num_heads = jumlah kepala (multi-head attention)
        * batch_first=True → input model dalam bentuk (batch, seq_len, dim) (bukan (seq_len, batch, dim))

    ✅ Pada FFN terdapat dua layer linear (dense) dengan ReLU di tengah.
        Gunanya untuk meningkatkan kapasitas representasi token.

    ✅ Setiap step (attention dan ffn) akan diberi residual connection → hasilnya distabilkan oleh LayerNorm. 
        Dropout mencegah overfitting.

    =====================
       forward sections
    =====================
    🔄 Flow: 
        1. Input x masuk ke self-attention → hasil: `attn_output`
        2. Hasilnya ditambahkan ke x (residual) lalu dinormalkan → `x = norm(x + attn)`
        3. Dimasukkan ke FFN → ditambahkan lagi ke inputnya → dinormalkan lagi
    """
    

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout):
        super().__init__()

        # self atention
        self.self_attn = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, dropout = dropout, batch_first = True)

        # Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embedding_dim)
        )

        # LayerNorm 
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # x: [batch_size, seq_len, embedding_dim]

        # Multi-head Self-attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask = mask)
        x = self.norm1(x + self.dropout1(attn_output))  # Residual + Norm

        # Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))  # Residual + Norm

        return x



class Encoder(nn.Module):
    """
    =====================
     initialize sections
    =====================
    🧠 Komponen 1 — Token Embedding
        🔤 Fungsi ini mengubah token ID (angka) menjadi vektor berdimensi embedding_dim.

    🧠 Komponen 2 — Positional Encoding
        🧭 Posisi kata dalam kalimat itu penting (misal: "kamu makan" ≠ "makan kamu").
        Kita tambahkan sinyal posisi ke setiap token.

    🧠 Komponen 3 — Layer Stack
        🔁 Ini membuat sekumpulan encoder layer. Misal: 6 layer → ulangi proses attention + FFN sebanyak 6x.

    =====================
       forward sections
    =====================
    🔄 Proses:
        1. Ubah token ID jadi vector → token_embedding
        2. Tambahkan posisi → + positional_encoding
        3. Dropout
        4. Masukkan ke setiap EncoderLayer
        5. Kembalikan output

    🔍 Positional Encoding (dalam _generate_positional_encoding)
        📐 Sinus dan Cosinus memberikan nilai unik berdasarkan posisi token, supaya model bisa tahu: "token ini di posisi ke-5" misalnya.
    """


    def __init__(self, vocab_size, max_len, embedding_dim, num_layers, num_heads, ffn_dim, dropout):
        super().__init__()
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional Encoding
        self.positional_encoding = self._generate_positional_encoding(max_len, embedding_dim)

        # Layer Stack
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # x: [batch_size, seq_len]
        
        seq_len = x.size(1)
        x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x  # Output: [batch_size, seq_len, embedding_dim]

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        return pe
