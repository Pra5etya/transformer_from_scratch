from src.model.encoder import Encoder
from src.config import config

import torch
import re
import string
import spacy

# ====================== Load Multilingual spaCy Model ======================
try:
    # Path tujuan penyimpanan model
    model_name = 'xx_sent_ud_sm'
    model_path = f'src/spacy/{model_name}'

    nlp = spacy.load(model_path)

except OSError:
    raise RuntimeError(f"Model spaCy {model_name} belum diunduh.")

# ====================== Kamus Typo dan Kata Gaul ======================
typo_dict = {
    "kmu": "kamu", "gk": "nggak", "smangat": "semangat", "gpp": "tidak apa-apa",
    "btw": "ngomong-ngomong", "wkwk": "haha", "wkwwk": "haha", "wkwkwk": "haha",
    "pls": "please", "thx": "terima kasih", "makasi": "terima kasih", 
    "makasih": "terima kasih", "tq": "terima kasih", "u": "you", 
    "luv": "love", "omg": "ya ampun",
}

# ====================== Daftar Akronim yang Dikecualikan dari Lowercase ======================
known_acronyms = {
    "USA", "NASA", "UNESCO", "AI", "ML", "IT", "CPU", "GPU", "HTML",
    "API", "JSON", "PDF", "SQL", "UN", "WHO", "COVID", "WWW", "URL",
    "HTTP", "HTTPS", "ID", "KTP"
}


# ====================== Normalisasi Typo ======================
def normalize_typo(text):
    text = re.sub(f"([{re.escape(string.punctuation)}])", r" \1 ", text)
    words = text.split()
    normalized_words = [typo_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)


# ====================== Smart Lowercase untuk Token ======================
def smart_lowercase(token):
    if token in known_acronyms or (token.isupper() and len(token) <= 5) or token == "I":
        return token
    return token.lower()


# ====================== Tokenisasi dan Pemisahan Kalimat (Multilingual) ======================
def tokenize_and_split_sentences(text):
    # Preprocessing: ubah ... menjadi .
    text = re.sub(r"\.\.\.+", ".", text)

    # Normalisasi typo/gaul
    text = normalize_typo(text)

    # Gunakan spaCy multilingual untuk pemisahan kalimat
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Gunakan regex tokenizer kustom
    token_pattern = r"""
        (?:https?://[^\s]+) |
        (?:\b(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:/[^\s]*)? |
        (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) |
        (?:@\w+) |
        (?:\#\w+) |
        (?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}) |
        (?:\d{4}[-/]\d{1,2}[-/]\d{1,2}) |
        (?:\d{1,2}\s+\w+\s+\d{2,4}) |
        (?:\w+\s+\d{1,2},\s+\d{4}) |
        (?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?) |
        (?:\d+(?:\.\d+)?%) |
        (?:[$€£¥₩₹]\s?\d+(?:[.,]\d+)*) |
        (?:rp\s?\d+(?:[.,]\d+)*) |
        (?:\b\w+\b) |
        (?:[^\w\s])
    """

    all_tokens = []
    for sentence in sentences:
        raw_tokens = re.findall(token_pattern, sentence, flags=re.VERBOSE)
        tokens = [smart_lowercase(tok) for tok in raw_tokens if tok.strip()]
        all_tokens.extend(tokens)

    return all_tokens, sentences


# ====================== Bangun Vocab dari Teks ======================
def build_vocab_from_text(text):
    tokens, sentences = tokenize_and_split_sentences(text)
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(sorted(set(tokens)), start=2):
        vocab[word] = i
    return vocab, tokens, sentences


# ====================== Unit Test Dinamis untuk Encoder ======================
def test_encoder_user_text_dynamic():
    user_text = input("Masukkan kalimat: ")

    vocab, tokens, sentences = build_vocab_from_text(user_text)
    vocab_size = len(vocab)

    # Tokenisasi per kalimat
    all_token_ids = []
    max_len = 0
    for sentence in sentences:
        toks, _ = tokenize_and_split_sentences(sentence)
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
        all_token_ids.append(ids)
        max_len = max(max_len, len(ids))

    # Padding tensor [num_sentences, max_len]
    padded_tensor = torch.full((len(all_token_ids), max_len), vocab["<pad>"], dtype=torch.long)
    for i, ids in enumerate(all_token_ids):
        padded_tensor[i, :len(ids)] = torch.tensor(ids)

    # Tokenisasi keseluruhan teks
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    # Inisialisasi model Encoder
    model = Encoder(
        vocab_size=vocab_size,
        max_len=max(max_len, len(token_ids)),
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.1
    )

    output_batch = model(padded_tensor)       # Output per kalimat (batched)
    output = model(input_tensor)              # Output keseluruhan teks

    # Print hasil
    print("\n✅ Vocabulary:", vocab)
    print("✅ Tokens:", tokens)
    print("✅ Token IDs:", token_ids)

    print("\n✅ Sentences:", sentences)
    print("✅ Input tensor shape (full text):", input_tensor.shape)
    print("✅ Output shape (full text):", output.shape)
    print("✅ Output tensor (full text):\n", output)

    print("\n📌 Input tensor shape (batch kalimat):", padded_tensor.shape)
    print("📌 Output shape (batch kalimat):", output_batch.shape)
    print("📌 Output tensor (batch kalimat):\n", output_batch)


if __name__ == "__main__":

    """
    with lib spacy
    """

    test_encoder_user_text_dynamic()
