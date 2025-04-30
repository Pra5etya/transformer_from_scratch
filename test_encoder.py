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

# ====== Dictionary typo dan kata gaul ======
typo_dict = {
    "kmu": "kamu",
    "gk": "nggak",
    "smangat": "semangat",
    "gpp": "tidak apa-apa",
    "btw": "ngomong-ngomong",
    "wkwk": "haha",
    "wkwwk": "haha",
    "wkwkwk": "haha",
    "pls": "please",
    "thx": "terima kasih",
    "makasi": "terima kasih",
    "makasih": "terima kasih",
    "tq": "terima kasih",
    "u": "you",
    "luv": "love",
    "omg": "ya ampun",
}

def normalize_typo(text):
    words = text.split()
    normalized_words = [typo_dict.get(word.lower(), word) for word in words]

    return ' '.join(normalized_words)


# ====== Tokenizer: bersihkan simbol dan lowercase ======
# def tokenize(text):

#     """
#     \b\w+\b – Kata biasa (alphanumeric word);                               🔍 Menangkap contoh: email, kirim, konfirmasi, sample2, 123data
#     [^\w\s] – Simbol atau karakter non-kata dan non-spasi;                  🔍 Menangkap contoh: @, ., !, ,, ?, :, #
#     (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) – Alamat email       🔍 Menangkap contoh: sample@gmail.com, nama.user+1@domain.co.id
#     """

#     tokens = re.findall(r'\b\w+\b|[^\w\s]|(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text.lower())
    
#     # Pisahkan kalimat berdasarkan tanda titik, tanda tanya, atau tanda seru
#     sentences = re.split(r'[.!?]', text)
#     sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences
    
#     return tokens, sentences


def tokenize_and_split_sentences(text):
    # Normalisasi typo/gaul
    text = normalize_typo(text)

    # Pola regex
    token_pattern = r"""
        (?:https?://[^\s]+) |                                          # URL
        (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) |           # Email
        (?:@\w+) |                                                     # Mention (@user)
        (?:\#\w+) |                                                    # Hashtag (#topic)
        
        (?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}) |                            # Tanggal: 29/04/2025 atau 04-29-25
        (?:\d{4}[-/]\d{1,2}[-/]\d{1,2}) |                              # Tanggal: 2025-04-29
        (?:\d{1,2}\s+\w+\s+\d{2,4}) |                                  # Tanggal: 29 April 2025
        (?:\w+\s+\d{1,2},\s+\d{4}) |                                   # Tanggal: April 29, 2025

        (?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?) |                        # Angka ribuan/decimal: 1,000 / 2.5M
        (?:\d+(?:\.\d+)?%) |                                           # Persentase: 50%, 12.5%
        (?:[$€£¥₩₹]\s?\d+(?:[.,]\d+)*) |                               # Simbol mata uang + angka
        (?:rp\s?\d+(?:[.,]\d+)*) |                                     # Khusus Rupiah: Rp50000, Rp 50.000

        (?:\b\w+\b) |                                                  # Kata biasa
        (?:[^\w\s])                                                    # Simbol lain (.,!?)
    """
    
    tokens = re.findall(token_pattern, text.lower(), flags=re.VERBOSE)
    
    # Pisahkan kalimat berdasarkan tanda akhir
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return tokens, sentences




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
    tokens, sentences = tokenize(text)
    
    unique_tokens = sorted(set(tokens))

    for i, word in enumerate(unique_tokens, start=2):
        vocab[word] = i
    
    return vocab, sentences

# === Unit Test dengan input bebas dari user ===
def test_encoder_user_text_dynamic():
    user_text = input("Masukkan kalimat: ")  

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
    print("✅ Output shape:", output.shape, "\n")
    print("✅ Output tensor:\n", output)

if __name__ == "__main__": 
    # test_encoder_random()
    test_encoder_user_text_dynamic()