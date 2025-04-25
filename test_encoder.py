from src.model.encoder import Encoder
from src.config import config

import torch
import math
import re


def test_encoder_random(): 
    encoder = Encoder(
        vocab_size = config['vocab_size'],
        max_len = config['max_len'],
        embedding_dim = config['embedding_dim'],
        num_layers = config['num_encoder_layers'],
        num_heads = config['num_heads'],
        ffn_dim = config['ffn_dim'],
        dropout = config['dropout']
    )

    # random input
    sample_input = torch.randint(0, config['vocab_size'], (config['batch_size'], config['max_len']))
    
    # Forward pass: hasilnya harus berbentuk [batch_size, seq_len, embedding_dim]
    output = encoder(sample_input)

    # Verifikasi ukuran output sesuai harapan
    print(output.shape)

    assert output.shape == (config['batch_size'], config['seq_len'], config['embedding_dim']), f"Output shape mismatch: {output.shape}"
    print("✅ Encoder unit test passed.")


# ====== Tokenizer: bersihkan simbol dan lowercase ======
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


# # Fungsi untuk membangun vocabulary dari teks input user
# # Vocabulary bersifat dinamis (dibuat dari input user langsung)
# def build_vocab_from_text(text):
#     vocab = {"<pad>": 0, "<unk>": 1}
#     words = list(set(text.lower().split()))
#     for i, word in enumerate(words, start=2):
#         vocab[word] = i
#     return vocab


# ====== Build Vocabulary dari teks user ======
def build_vocab_from_text(text):
    vocab = {"<pad>": 0, "<unk>": 1}
    tokens = tokenize(text)
    unique_tokens = sorted(set(tokens))
    for i, word in enumerate(unique_tokens, start=2):
        vocab[word] = i
    return vocab

# === Unit Test dengan input bebas dari user ===
def test_encoder_user_text_dynamic():
    user_text = input("Masukkan kalimat: ")  # contoh: saya suka makan nasi goreng

    vocab = build_vocab_from_text(user_text)
    vocab_size = len(vocab)

    # Tokenisasi teks user menjadi ID sesuai vocab
    tokens = user_text.lower().split()
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, seq_len]

    # Inisialisasi encoder di CPU
    model = Encoder(
        vocab_size=vocab_size,
        max_len=50,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.1
    )

    output = model(input_tensor)
    print("\n✅ Vocabulary:", vocab)
    print("✅ Token IDs:", token_ids)
    print("✅ Output shape:", output.shape)
    print("✅ Output tensor:\n", output)

if __name__ == "__main__": 
    # test_encoder_random()
    test_encoder_user_text_dynamic()