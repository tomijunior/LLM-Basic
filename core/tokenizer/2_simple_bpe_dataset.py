from tqdm import tqdm
import time
from collections import Counter
from typing import List, Tuple, Dict, Iterator
import regex as re
import json
import os
from datetime import datetime
from datasets import load_dataset


# ===========================================================================
# GPT-2 byte-to-unicode mapping — satu-satunya cara aman untuk serialize
# semua 256 byte ke string yang unik dan reversibel (tanpa collision).
# Byte yang sudah printable dipetakan 1:1; sisanya dipetakan ke range
# Unicode ĀāĂ… (U+0100 ke atas) yang aman untuk JSON.
# ===========================================================================
def _bytes_to_unicode() -> Dict[int, str]:
    """Bijeksi byte (0-255) → karakter Unicode yang printable & unik."""
    bs = (
        list(range(ord("!"), ord("~") + 1))        # 33-126
        + list(range(ord("¡"), ord("¬") + 1))      # 161-172
        + list(range(ord("®"), ord("ÿ") + 1))      # 174-255
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE: Dict[int, str] = _bytes_to_unicode()
UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in BYTE_TO_UNICODE.items()}


class SimpleBPE:
    """
    Implementasi Byte Pair Encoding (BPE) dari nol untuk tujuan pembelajaran.
    """

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}   # (id1, id2) -> new_id
        self.vocab: Dict[int, bytes] = {}              # id -> byte representation
        self.vocab_size: int = 0
        self.special_tokens: Dict[str, int] = {}       # token_str -> id

    def clean_text(self, text: str) -> str:
        """Membersihkan teks sebelum pre-tokenization."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in text.split('\n')]
        text = ' '.join(lines)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
        return text.strip()

    # ===================== SPECIAL TOKENS (Granite Style) =====================
    def _add_special_tokens(self):
        """Special tokens seperti yang digunakan IBM Granite."""
        special_list = [
            "<|endoftext|>",
            "<fim_prefix>",
            "<fim_middle>",
            "<fim_suffix>",
            "<fim_pad>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<empty_output>",
            "<commit_before>",
            "<commit_msg>",
            "<commit_after>",
            "<reponame>",
        ]
        for i, token_str in enumerate(special_list):
            self.special_tokens[token_str] = i
            self.vocab[i] = token_str.encode("utf-8")
        self.vocab_size = len(special_list)
        print(f"✅ Added {len(special_list)} special tokens (IBM Granite style)")

    def _add_byte_tokens(self):
        """Tambahkan 256 byte tokens setelah special tokens."""
        start_id = self.vocab_size
        for i in range(256):
            byte_id = start_id + i
            self.vocab[byte_id] = bytes([i])
        self.vocab_size += 256
        print(f"✅ Added 256 byte tokens (ID {start_id} - {self.vocab_size - 1})")

    # ===================== PRE-TOKENIZATION (Granite Style) =====================
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenizer mendekati IBM Granite."""
        text = self.clean_text(text)
        text = re.sub(r'(\d)', r' \1 ', text)
        pattern = r"""
            (?i:'s|'t|'re|'ve|'m|'ll|'d)
            | \p{L}+
            | \p{N}+
            | [^\s\p{L}\p{N}]+
            | \s+
        """
        tokens = re.findall(pattern, text, re.VERBOSE | re.UNICODE)
        return [t for t in tokens if t]

    def get_pairs(self, token_ids: List[int]) -> Counter:
        """Menghitung frekuensi semua pasangan token bersebelahan."""
        pairs = Counter()
        for i in range(len(token_ids) - 1):
            pairs[(token_ids[i], token_ids[i + 1])] += 1
        return pairs

    def _pair_to_str(self, pair: Tuple[int, int]) -> str:
        """Mengubah pair (id1, id2) ke representasi string yang mudah dibaca."""
        try:
            t1 = self.vocab[pair[0]].decode('utf-8', errors='replace')
            t2 = self.vocab[pair[1]].decode('utf-8', errors='replace')
            t1 = repr(t1)[1:-1] if t1 else "∅"
            t2 = repr(t2)[1:-1] if t2 else "∅"
            return f"({t1} , {t2})"
        except Exception:
            return f"ID({pair[0]},{pair[1]})"

    # ===================== SERIALIZATION HELPERS =====================
    def _token_bytes_to_repr(self, token_bytes: bytes) -> str:
        """
        Ubah bytes token ke string representasi yang aman untuk JSON.
        Menggunakan GPT-2 byte-to-unicode mapping agar setiap byte
        dipetakan ke karakter unik yang reversibel.
        """
        return "".join(BYTE_TO_UNICODE[b] for b in token_bytes)

    def _repr_to_token_bytes(self, token_repr: str) -> bytes:
        """Kebalikan dari _token_bytes_to_repr."""
        return bytes(UNICODE_TO_BYTE[c] for c in token_repr)

    # ===================== TRAINING =====================
    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """Melatih BPE Tokenizer dari teks mentah."""
        text = self.clean_text(text)
        self._add_special_tokens()
        self._add_byte_tokens()

        pre_tokens = self.pre_tokenize(text)
        print(f"Pre-tokenization menghasilkan {len(pre_tokens)} token awal.\n")

        offset = len(self.special_tokens)
        tokens: List[int] = []
        for pt in pre_tokens:
            for b in pt.encode("utf-8"):
                tokens.append(offset + b)

        start_id = self.vocab_size
        print(f"Training dimulai dengan {len(tokens):,} byte tokens...\n")

        for new_id in range(start_id, vocab_size):
            pairs = self.get_pairs(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            if freq < 2:
                print("Tidak ada lagi pasangan yang muncul >1 kali.")
                break

            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_token_bytes
            self.merges[best_pair] = new_id

            i, new_tokens = 0, []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.vocab_size = new_id + 1

            if verbose and (new_id < 700 or new_id % 100 == 0):
                pair_str = self._pair_to_str(best_pair)
                print(f"Merge {new_id:5d} | Freq {freq:3d} | "
                      f"Pair: {pair_str:28} | "
                      f"Tokens: {len(tokens):6,}")

        print(f"✅ Training selesai! Vocab size akhir: {self.vocab_size:,}")

    # ===================== ENCODE & DECODE =====================
    def encode(self, text: str) -> List[int]:
        """Encode teks ke token IDs."""
        if not text:
            return []

        pre_tokens = self.pre_tokenize(text)
        token_ids = []
        offset = len(self.special_tokens)

        for pt in pre_tokens:
            ids = [offset + b for b in pt.encode("utf-8")]
            changed = True
            while changed and len(ids) > 1:
                changed = False
                i, new_ids = 0, []
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self.merges:
                        new_ids.append(self.merges[(ids[i], ids[i + 1])])
                        i += 2
                        changed = True
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            token_ids.extend(ids)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs kembali ke teks."""
        if not token_ids:
            return ""

        bytes_data = b""
        offset = len(self.special_tokens)

        for tid in token_ids:
            if tid in self.vocab:
                bytes_data += self.vocab[tid]
            elif tid >= offset:
                bytes_data += bytes([tid - offset])
            else:
                bytes_data += bytes([tid])

        return bytes_data.decode("utf-8", errors="replace")

    # ===================== KOMPRESI =====================
    def compression_ratio(self, text: str, encoded: List[int] = None):
        if encoded is None:
            encoded = self.encode(text)
        original_bytes = len(text.encode("utf-8"))
        num_tokens = len(encoded)
        ratio = original_bytes / num_tokens if num_tokens > 0 else 0
        return {
            "original_bytes": original_bytes,
            "num_tokens": num_tokens,
            "compression_ratio": round(ratio, 3),
            "avg_bytes_per_token": round(ratio, 3),
        }

    def get_vocab_size(self):
        return self.vocab_size

    # ===================== DATASET TRAINING =====================
    def train_from_dataset(self,
                           dataset_name: str = "indonesian-nlp/wikipedia-10k",
                           subset: str = "wikipedia-id",
                           vocab_size: int = 15000,
                           verbose: bool = True,
                           max_examples: int = None,
                           show_merge_every: int = 5000):
        """Train tokenizer dari dataset dengan progress bar + ETA."""
        print(f"🔄 Loading dataset: {dataset_name} | subset: {subset}")

        dataset = load_dataset(dataset_name, subset, split="test")
        if max_examples:
            dataset = dataset.select(range(max_examples))

        total_articles = len(dataset)
        print(f"✅ Dataset loaded: {total_articles:,} articles\n")

        self._add_special_tokens()
        self._add_byte_tokens()

        offset = len(self.special_tokens)
        tokens: List[int] = []
        chunk_size = 450_000
        merge_counter = 0

        print("🚀 Mulai training tokenizer...\n")

        pbar = tqdm(total=total_articles, desc="Processing articles",
                    unit="art", smoothing=0.1, dynamic_ncols=True)
        start_time = time.time()

        for text in dataset:
            title   = text.get("title", "").strip()
            content = text.get("text",  "").strip()
            combined = f"{title}\n\n{content}" if title else content

            if len(combined.strip()) < 100:
                pbar.update(1)
                continue

            for pt in self.pre_tokenize(combined):
                for b in pt.encode("utf-8"):
                    tokens.append(offset + b)

            pbar.update(1)
            pbar.set_postfix({"tokens": f"{len(tokens):,}", "vocab": self.vocab_size})

            if len(tokens) >= chunk_size:
                pbar.set_description(f"Training chunk {len(tokens):,} tokens")
                self._train_from_tokens(tokens, vocab_size, verbose=False,
                                        show_merge_every=show_merge_every,
                                        global_counter=merge_counter)
                merge_counter += len(tokens)
                tokens = []

        pbar.close()

        if tokens:
            print(f"\nTraining sisa token ({len(tokens):,} tokens)...")
            self._train_from_tokens(tokens, vocab_size, verbose=True,
                                    show_merge_every=show_merge_every)

        total_time = time.time() - start_time
        print(f"\n🎉 Training selesai! Vocab Size = {self.vocab_size:,}")
        print(f"⏱️  Total waktu training: {total_time / 60:.1f} menit")
        return self

    def _train_from_tokens(self, tokens: List[int], vocab_size: int,
                           verbose: bool = True,
                           show_merge_every: int = 5000,
                           global_counter: int = 0):
        """Core training dengan progress bar per chunk."""
        if self.vocab_size == 0:
            self._add_special_tokens()
            self._add_byte_tokens()

        initial_size = self.vocab_size
        target_size  = min(vocab_size, initial_size + 2000)

        print(f"   Training chunk: {len(tokens):,} tokens → target vocab {target_size:,}")

        pbar = tqdm(total=target_size - initial_size,
                    desc="   Merging", unit="merge", leave=False)
        start_time = time.time()

        for new_id in range(self.vocab_size, target_size):
            pairs = self.get_pairs(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            if freq < 2:
                break

            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_token_bytes
            self.merges[best_pair] = new_id

            i, new_tokens = 0, []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.vocab_size = new_id + 1

            pbar.update(1)

            if (new_id - initial_size) % show_merge_every == 0 and new_id > initial_size:
                pair_str = self._pair_to_str(best_pair)
                elapsed  = time.time() - start_time
                print(f"   → Merge {new_id:6d} | Freq {freq:4d} | "
                      f"Pair: {pair_str:35} | Tokens: {len(tokens):6,}")

        pbar.close()

    # ===================== SAVE =====================
    def save_tokenizer(self, directory: str = "tokenizer_indonesia"):
        """
        Simpan tokenizer ke format Hugging Face.

        Menggunakan GPT-2 byte-to-unicode mapping sehingga setiap byte
        direpresentasikan oleh karakter Unicode yang unik — tidak ada
        collision, tidak ada data hilang saat round-trip save/load.
        """
        os.makedirs(directory, exist_ok=True)

        # --- Vocab: {repr_string: id} ---
        vocab: Dict[str, int] = {}
        vocab_id_set: set = set()          # FIX: O(1) lookup, bukan O(n)

        # Special tokens (disimpan apa adanya — string-nya sudah aman UTF-8)
        for token_str, tid in self.special_tokens.items():
            vocab[token_str] = tid
            vocab_id_set.add(tid)

        # Byte tokens — gunakan GPT-2 mapping agar setiap byte → karakter unik
        offset = len(self.special_tokens)
        for i in range(256):
            byte_id   = offset + i
            byte_repr = BYTE_TO_UNICODE[i]   # FIX: mapping yang benar & reversibel
            vocab[byte_repr] = byte_id
            vocab_id_set.add(byte_id)

        # Merged tokens
        for tid in range(self.vocab_size):
            if tid in self.vocab and tid not in vocab_id_set:
                token_repr = self._token_bytes_to_repr(self.vocab[tid])  # FIX
                vocab[token_repr] = tid
                vocab_id_set.add(tid)

        # --- Merges: ["repr1 repr2", ...] diurutkan berdasarkan new_id ---
        merges = []
        for (id1, id2), _ in sorted(self.merges.items(), key=lambda x: x[1]):
            repr1 = self._token_bytes_to_repr(self.vocab[id1])  # FIX
            repr2 = self._token_bytes_to_repr(self.vocab[id2])  # FIX
            merges.append(f"{repr1} {repr2}")

        # --- tokenizer.json ---
        tokenizer_json = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
                "unk_token": None,
                "byte_fallback": True,
            },
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Digits", "individual_digits": True},
                    {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
                ],
            },
            "decoder": {"type": "ByteLevel", "add_prefix_space": False},
            "post_processor": None,
            "added_tokens": [],
        }

        with open(os.path.join(directory, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        config = {
            "model_type":    "bpe",
            "vocab_size":    self.vocab_size,
            "special_tokens": list(self.special_tokens.keys()),
            "trained_on":    "indonesian-nlp/wikipedia-10k",
            "date":          datetime.now().isoformat(),
        }
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ Tokenizer berhasil disimpan ke: {directory}/")
        print(f"   Vocab Size : {self.vocab_size:,}")
        print(f"   Merges     : {len(merges):,}")

    # ===================== LOAD =====================
    def load_tokenizer(self, directory: str = "tokenizer_indonesia"):
        """Load tokenizer dari folder Hugging Face style."""
        tokenizer_path = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"File tokenizer.json tidak ditemukan di: {directory}"
            )

        print(f"🔄 Loading tokenizer dari: {directory}")

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model       = data.get("model", {})
        vocab_dict  = model.get("vocab", {})     # repr_str -> id  (int)
        merges_list = model.get("merges", [])

        # Reset state
        self.merges = {}
        self.vocab  = {}
        self.special_tokens = {}

        # --- Kumpulkan set special token yang diketahui ---
        # (sama dengan _add_special_tokens agar deteksi tidak bergantung pada
        #  heuristik string saja)
        _known_special = {
            "<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>",
            "<fim_pad>", "<filename>", "<gh_stars>", "<issue_start>",
            "<issue_comment>", "<issue_closed>", "<jupyter_start>",
            "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
            "<empty_output>", "<commit_before>", "<commit_msg>",
            "<commit_after>", "<reponame>",
        }

        # FIX 1: iterasi yang benar — vocab_dict adalah {str: int}, bukan {int: str}
        for token_repr, token_id in vocab_dict.items():
            token_id = int(token_id)   # pastikan int

            # Decode repr → bytes menggunakan GPT-2 reverse mapping
            # FIX 2: gunakan _repr_to_token_bytes, bukan token_repr.encode("utf-8")
            if token_repr in _known_special:
                # Special token: simpan bytes aslinya
                self.vocab[token_id] = token_repr.encode("utf-8")
                self.special_tokens[token_repr] = token_id
            else:
                try:
                    self.vocab[token_id] = self._repr_to_token_bytes(token_repr)
                except KeyError:
                    # Fallback untuk token yang mungkin disimpan tanpa mapping
                    self.vocab[token_id] = token_repr.encode("utf-8")

        self.vocab_size = max(self.vocab.keys()) + 1 if self.vocab else 0

        # FIX 3: Merge ID diambil dari vocab_dict (bukan offset+rank yang arbitrary)
        # Buat reverse lookup: repr_str -> id
        repr_to_id: Dict[str, int] = {k: int(v) for k, v in vocab_dict.items()}

        loaded_merges = 0
        for merge_str in merges_list:
            if " " not in merge_str:
                continue

            repr1, repr2 = merge_str.split(" ", 1)
            merged_repr  = repr1 + repr2   # repr dari token hasil merge

            id1    = repr_to_id.get(repr1)
            id2    = repr_to_id.get(repr2)
            new_id = repr_to_id.get(merged_repr)   # FIX: lookup ID yang benar

            if id1 is not None and id2 is not None and new_id is not None:
                self.merges[(id1, id2)] = new_id
                loaded_merges += 1

        print(f"✅ Tokenizer berhasil dimuat!")
        print(f"   Vocab Size     : {self.vocab_size:,}")
        print(f"   Special Tokens : {len(self.special_tokens)}")
        print(f"   Merges         : {loaded_merges:,}")

        if loaded_merges == 0:
            print("⚠️  Warning: Tidak ada merges yang dimuat — "
                  "tokenizer hanya akan pakai byte level.")

        return self


# ===========================================================================
if __name__ == "__main__":
    bpe = SimpleBPE()

    # --- Uncomment untuk training ---
    bpe.train_from_dataset(
         dataset_name="indonesian-nlp/wikipedia-10k",
         subset="wikipedia-id",
         vocab_size=5000,
         verbose=True,
         max_examples=1000,
         show_merge_every=3000
    )
    bpe.save_tokenizer("tokenizer_indonesia")

    # --- Load & test ---
    bpe.load_tokenizer("tokenizer_indonesia")

    test_text = "Bahasa Indonesia adalah bahasa nasional yang kaya akan budaya."
    encoded   = bpe.encode(test_text)
    decoded   = bpe.decode(encoded)

    print("\n=== TEST HASIL LOAD ===")
    print("Teks         :", test_text)
    print("Token IDs    :", encoded[:30], "..." if len(encoded) > 30 else "")
    print("Jumlah Token :", len(encoded))
    print("Decode back  :", decoded)
    print("Round-trip OK:", test_text == decoded)

    stats = bpe.compression_ratio(test_text, encoded)
    print(f"Kompresi     : {stats['compression_ratio']:.2f} bytes/token")