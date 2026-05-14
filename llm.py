from core.tokenizer.simple_bpe_optimized import SimpleBPE
from core.embeding.simple_embeding import Embedding


if __name__ == "__main__":
    # Load tokenizer yang sudah kita buat
    bpe = SimpleBPE()
    bpe.load_tokenizer("core/tokenizer/tokenizer_indonesia")

    # Buat Embedding
    embedding = Embedding(
        vocab_size=bpe.get_vocab_size(),
        embed_dim=512,           # bisa 256, 512, 768, 1024, dll
        max_seq_len=2048
    )

    # Test
    texts = [
        "Bahasa Indonesia adalah bahasa yang indah.",
        "Kita sedang membangun tokenizer dan embedding dari nol."
    ]

    print("=== TESTING EMBEDDING ===\n")
    for text in texts:
        token_ids = bpe.encode(text)
        emb = embedding.forward(token_ids)
        
        print(f"Teks     : {text}")
        print(f"Tokens   : {len(token_ids)}")
        print(f"Embedding Shape : {emb.shape}")
        print(f"Contoh vektor (10 dimensi pertama):")
        print(emb[0][:10].round(4))
        print("-" * 70)