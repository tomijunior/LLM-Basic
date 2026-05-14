from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors, normalizers
from transformers import PreTrainedTokenizerFast

# 1. Siapkan data teks (contoh: dataset wikitext atau file teks lokal Anda)
# Untuk file lokal, gunakan: load_dataset("text", data_files={"train": "data.txt"})
dataset = load_dataset("indonesian-nlp/wikipedia-10k", "wikipedia-id", split="test")

# 2. Buat generator iterator untuk menghemat memori RAM
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

# 2. Inisialisasi arsitektur Tokenizer kosong dengan model BPE
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 3. TAMBAHKAN BLOK NORMALISASI (Diurutkan secara sekuensial)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),              # Memisahkan karakter aksen (misal: é menjadi e + ´)
    normalizers.Lowercase(),        # Mengubah semua teks menjadi huruf kecil
    normalizers.StripAccents(),     # Menghapus komponen aksen hasil pemisahan NFD
    normalizers.Replace(" {2,}", " ") # Regex untuk mereduksi spasi ganda menjadi satu spasi
])

# 4. Tambahkan Pre-Tokenizer (ByteLevel untuk model modern)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 5. DEFINISIKAN DAFTAR SPECIAL TOKENS KUSTOM
# Menyertakan token kontrol chat standard (<|im_start|>, <|im_end|>)
special_tokens_list = [
    "<unk>",        # Unknown token (ID: 0)
    "<s>",          # Start of string (ID: 1)
    "</s>",         # End of string (ID: 2)
    "<pad>",        # Padding token (ID: 3)
    "<|im_start|>", # Batas awal instruksi user/system (ID: 4)
    "<|im_end|>",    # Batas akhir instruksi user/system (ID: 5)
    "<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>",
    "<fim_pad>", "<filename>", "<gh_stars>", "<issue_start>",
    "<issue_comment>", "<issue_closed>", "<jupyter_start>",
    "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
    "<empty_output>", "<commit_before>", "<commit_msg>",
    "<commit_after>", "<reponame>",
]

# 6. Konfigurasi Trainer BPE
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens_list
)

# 7. Jalankan pelatihan menggunakan iterator data
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# 8. Pasang Decoder & Post-Processor
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# 9. Bungkus ke dalam kelas Transformers dengan pemetaan Special Tokens yang eksplisit
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    additional_special_tokens=["<|im_start|>", "<|im_end|>",
    "<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>",
    "<fim_pad>", "<filename>", "<gh_stars>", "<issue_start>",
    "<issue_comment>", "<issue_closed>", "<jupyter_start>",
    "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
    "<empty_output>", "<commit_before>", "<commit_msg>",
    "<commit_after>", "<reponame>"]
)

# 10. Simpan hasil latihan Anda
fast_tokenizer.save_pretrained("./tokenizer_indonesia_transformer")
print("Pelatihan tokenizer selesai dengan normalisasi teks!")