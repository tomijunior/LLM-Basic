from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_indonesia_transformer")

teks_kotor = "  Halo   Dunia! Ini   Café   kustom saya.  "
token_ids = tokenizer.encode(teks_kotor, add_special_tokens=False)
teks_asli = tokenizer.decode(token_ids)
print("Hasil Decode (Teks Asli):", teks_asli)

# Menguji integrasi Special Tokens untuk skenario Chat LLM
teks_chat = "<|im_start|>user\nSiapa kamu?<|im_end|>\n<|im_start|>assistant\n"
print("Token IDs Chat:", tokenizer.encode(teks_chat))
