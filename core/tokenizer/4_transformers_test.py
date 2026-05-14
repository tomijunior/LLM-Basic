from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_indonesia_transformer")

# Contoh teks dengan Huruf Kapital, Spasi Berlebih, dan Karakter Aksen (é)
teks_kotor = "  Halo   Dunia! Ini   Café   kustom saya.  "

# Efek Normalisasi: Teks akan otomatis diproses seolah-olah berbunyi " halo dunia! ini cafe kustom saya. "
tokens = tokenizer.tokenize(teks_kotor)
print("Hasil Token (Sudah Normal):", tokens)
token_ids = tokenizer.encode(teks_kotor, add_special_tokens=False)
teks_asli = tokenizer.decode(token_ids)
print("Hasil Decode (Teks Asli):", teks_asli)

# Menguji integrasi Special Tokens untuk skenario Chat LLM
teks_chat = "<|im_start|>user\nSiapa kamu?<|im_end|>\n<|im_start|>assistant\n"
print("Token IDs Chat:", tokenizer.encode(teks_chat))
