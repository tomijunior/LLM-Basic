from datasets import load_dataset
import regex as re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors, Regex
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from transformers import PreTrainedTokenizerFast

# 1. Siapkan data teks (contoh: dataset wikitext atau file teks lokal Anda)
# Untuk file lokal, gunakan: load_dataset("text", data_files={"train": "data.txt"})
dataset = load_dataset("indonesian-nlp/wikipedia-10k", "wikipedia-id", split="test")
total = len(dataset)
print(f"✅ Dataset loaded : {total:,} articles\n")

# 2. Buat generator iterator untuk menghemat memori RAM
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

# 2. Inisialisasi arsitektur Tokenizer kosong dengan model BPE
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 3. Tambahkan Pre-Tokenizer (ByteLevel untuk model modern)
pattern_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# Compile regex menggunakan pustaka 'regex' eksternal untuk validasi Unicode
compiled_regex = re.compile(pattern_str)

tokenizer.pre_tokenizer = Sequence([
    # Komponen 1: Split berdasarkan Regex
    Split(
        pattern=Regex(pattern_str),
        behavior="removed",  # "behavior": "Removed" di JSON
        invert=True          # "invert": true di JSON
    ),
    # Komponen 2: ByteLevel tanpa regex internal tambahan
    ByteLevel(
        add_prefix_space=False, # "add_prefix_space": false
        trim_offsets=True,      # "trim_offsets": true
        use_regex=False         # "use_regex": false
    )
])

# 4. DEFINISIKAN DAFTAR SPECIAL TOKENS KUSTOM
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

# 5. Konfigurasi Trainer BPE
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens_list
)

# 6. Jalankan pelatihan menggunakan iterator data
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# 7. Pasang Decoder & Post-Processor
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# 8. Bungkus ke dalam kelas Transformers dengan pemetaan Special Tokens yang eksplisit
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

# 9. Simpan hasil latihan Anda
fast_tokenizer.save_pretrained("./tokenizer_indonesia_transformer")
print("Pelatihan tokenizer selesai dengan normalisasi teks!")