# Sample 1
```bash
# def test_encoder_user_text_dynamic():
#     user_text = input("Masukkan kalimat: ")

#     vocab, tokens, sentences = build_vocab_from_text(user_text)
#     vocab_size = len(vocab)

#     # Tokenisasi per kalimat
#     all_token_ids = []
#     max_len = 0

#     for sentence in sentences:
#         toks, _ = tokenize_and_split_sentences(sentence)
#         ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
#         all_token_ids.append(ids)
#         max_len = max(max_len, len(ids))

#     # Padding manual agar ukuran seragam
#     padded_tensor = torch.full((len(all_token_ids), max_len), vocab["<pad>"], dtype=torch.long)
#     for i, ids in enumerate(all_token_ids):
#         padded_tensor[i, :len(ids)] = torch.tensor(ids)

#     token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
#     input_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, seq_len]

#     # Inisialisasi encoder di CPU
#     model = Encoder(
#         vocab_size=vocab_size,
#         max_len=50,
#         embedding_dim=32,
#         num_layers=2,
#         num_heads=4,
#         ffn_dim=64,
#         dropout=0.1
#     )

#     output = model(input_tensor)
#     print("\n✅ Vocabulary:", vocab)
#     print("✅ Tokens:", tokens)
#     print("✅ Token IDs:", token_ids)

#     print("\n✅ Sentences:", sentences)
#     print("✅ Output shape:", output.shape)
#     print("✅ Output tensor:\n", output)
```