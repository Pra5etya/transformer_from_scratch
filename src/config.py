config = {
    "batch_size": 2,                # Jumlah data / input yang diproses dalam satu batch (2 kalimat / token sequence sekaligus)
    "seq_len": 10,                  # Panjang setiap sequence input (misal jumlah token / kata dalam satu kalimat)

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
    "device": "cuda"                # device use ("gpu" atau "cpu")
}