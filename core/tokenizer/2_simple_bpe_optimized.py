"""
SimpleBPE — Implementasi BPE dari nol (versi teroptimasi)

Optimasi utama vs versi awal:
  1. Word-level vocabulary  →  kata unik diproses sekali (bukan n kali)
  2. Incremental pair freq  →  O(k) per merge, bukan O(n)
  3. Lazy max-heap          →  O(log p) untuk cari pair terbaik, bukan O(p)
  4. Parallel pre-tokenize  →  ProcessPoolExecutor untuk dataset besar
"""

import heapq
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import regex as re
from datasets import load_dataset
from tqdm import tqdm


# ===========================================================================
# GPT-2 byte-to-unicode mapping — bijeksi sempurna, tidak ada collision,
# aman untuk JSON round-trip (termasuk byte 0x80–0xFF yang bukan UTF-8 valid)
# ===========================================================================
def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE: Dict[int, str] = _bytes_to_unicode()
UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in BYTE_TO_UNICODE.items()}


# ===========================================================================
# Worker function di level modul — wajib untuk ProcessPoolExecutor (pickling)
# ===========================================================================
def _pretokenize_worker(args: Tuple[str, int]) -> Dict[Tuple[int, ...], int]:
    """
    Pre-tokenize satu artikel dan kembalikan word-frequency dict.
    Fungsi ini harus ada di level modul (bukan method class) agar bisa
    di-pickle oleh ProcessPoolExecutor di Windows maupun Linux.
    """
    text, offset = args

    # --- clean text ---
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = text.strip()
    if len(text) < 50:
        return {}

    # --- pre-tokenize ---
    text = re.sub(r"(\d)", r" \1 ", text)
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+"
    tokens = re.findall(pattern, text, re.UNICODE)

    word_freq: Dict[Tuple[int, ...], int] = {}
    for tok in tokens:
        if tok:
            word = tuple(offset + b for b in tok.encode("utf-8"))
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


# ===========================================================================
# Kelas utama
# ===========================================================================
class SimpleBPE:
    """Implementasi BPE dari nol — versi dengan optimasi kecepatan."""

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.vocab_size: int = 0
        self.special_tokens: Dict[str, int] = {}

    # ------------------------------------------------------------------ utils
    def clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = " ".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
        return text.strip()

    def _token_bytes_to_repr(self, token_bytes: bytes) -> str:
        return "".join(BYTE_TO_UNICODE[b] for b in token_bytes)

    def _repr_to_token_bytes(self, token_repr: str) -> bytes:
        return bytes(UNICODE_TO_BYTE[c] for c in token_repr)

    @staticmethod
    def _word_to_pairs(word: Tuple[int, ...]) -> Counter:
        """Hitung semua adjacent pair dalam satu word (sebagai tuple ID)."""
        c: Counter = Counter()
        for i in range(len(word) - 1):
            c[(word[i], word[i + 1])] += 1
        return c

    def _pair_to_str(self, pair: Tuple[int, int]) -> str:
        try:
            t1 = repr(self.vocab[pair[0]].decode("utf-8", errors="replace"))[1:-1]
            t2 = repr(self.vocab[pair[1]].decode("utf-8", errors="replace"))[1:-1]
            return f"({t1} , {t2})"
        except Exception:
            return f"ID({pair[0]},{pair[1]})"

    # -------------------------------------------------- inisialisasi vocab
    _SPECIAL_LIST = [
        "<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>",
        "<fim_pad>", "<filename>", "<gh_stars>", "<issue_start>",
        "<issue_comment>", "<issue_closed>", "<jupyter_start>",
        "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
        "<empty_output>", "<commit_before>", "<commit_msg>",
        "<commit_after>", "<reponame>",
    ]

    def _add_special_tokens(self):
        for i, tok in enumerate(self._SPECIAL_LIST):
            self.special_tokens[tok] = i
            self.vocab[i] = tok.encode("utf-8")
        self.vocab_size = len(self._SPECIAL_LIST)
        print(f"✅ Special tokens: {self.vocab_size}")

    def _add_byte_tokens(self):
        start = self.vocab_size
        for i in range(256):
            self.vocab[start + i] = bytes([i])
        self.vocab_size += 256
        print(f"✅ Byte tokens   : ID {start}–{self.vocab_size - 1}")

    # -------------------------------------------------- pre-tokenization
    def pre_tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        text = re.sub(r"(\d)", r" \1 ", text)
        pattern = r"""
            (?i:'s|'t|'re|'ve|'m|'ll|'d)
            | \p{L}+ | \p{N}+ | [^\s\p{L}\p{N}]+ | \s+
        """
        return [t for t in re.findall(pattern, text, re.VERBOSE | re.UNICODE) if t]

    # ------------------------------------------------------------------ encode/decode
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        offset = len(self.special_tokens)
        ids = []
        for pt in self.pre_tokenize(text):
            seq = [offset + b for b in pt.encode("utf-8")]
            changed = True
            while changed and len(seq) > 1:
                changed = False
                i, new_seq = 0, []
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) in self.merges:
                        new_seq.append(self.merges[(seq[i], seq[i + 1])])
                        i += 2
                        changed = True
                    else:
                        new_seq.append(seq[i])
                        i += 1
                seq = new_seq
            ids.extend(seq)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        offset = len(self.special_tokens)
        data = b""
        for tid in token_ids:
            if tid in self.vocab:
                data += self.vocab[tid]
            elif tid >= offset:
                data += bytes([tid - offset])
            else:
                data += bytes([tid])
        return data.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str, encoded: List[int] = None):
        if encoded is None:
            encoded = self.encode(text)
        nb = len(text.encode("utf-8"))
        nt = len(encoded)
        r = nb / nt if nt else 0
        return {"original_bytes": nb, "num_tokens": nt,
                "compression_ratio": round(r, 3), "avg_bytes_per_token": round(r, 3)}

    def get_vocab_size(self):
        return self.vocab_size

    # ==================================================================
    #  CORE OPTIMIZED TRAINING — bekerja langsung pada word vocabulary
    # ==================================================================
    def _train_from_word_vocab(
        self,
        word_vocab: Dict[Tuple[int, ...], int],
        vocab_size: int,
        verbose: bool = True,
        show_merge_every: int = 500,
    ):
        """
        Inti training BPE yang dioptimasi.

        Struktur data utama:
          word_vocab  : {(id1, id2, ...): count}  — kata unik → frekuensi
          pair_freq   : {(id1, id2): total_count}  — diupdate incremental
          heap        : [(-freq, pair)]             — lazy max-heap

        Kompleksitas per merge:
          • Lama  : O(n)  + O(p)   [n = total token, p = unique pairs]
          • Baru  : O(k log p)     [k = kemunculan pair terbaik, k << n]
        """
        # Hitung pair frequencies awal dari word_vocab
        pair_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        for word, cnt in word_vocab.items():
            for pair, pc in self._word_to_pairs(word).items():
                pair_freq[pair] += pc * cnt

        # Bangun max-heap (lazy: simpan (-freq, pair))
        # Python heapq adalah min-heap → simpan negatif frekuensi
        heap: List[Tuple[int, Tuple[int, int]]] = [
            (-f, p) for p, f in pair_freq.items() if f >= 2
        ]
        heapq.heapify(heap)

        start_id = self.vocab_size
        target   = vocab_size
        total_merges = target - start_id

        if total_merges <= 0:
            print("Vocab sudah mencapai target.")
            return

        print(f"\n   Word vocab    : {len(word_vocab):,} kata unik")
        print(f"   Unique pairs  : {len(pair_freq):,}")
        print(f"   Target merges : {total_merges:,}\n")

        pbar = tqdm(total=total_merges, desc="   BPE merging", unit="merge", leave=True)
        t0 = time.time()

        for new_id in range(start_id, target):
            # --- Ambil pair terbaik (lazy deletion) ---
            # Entry di heap bisa "stale" jika frekuensinya sudah berubah.
            # Kita pop terus sampai ketemu entry yang masih valid.
            best_pair, best_freq = None, 0
            while heap:
                neg_f, pair = heapq.heappop(heap)
                f = -neg_f
                if pair_freq.get(pair, 0) == f:   # ← masih valid
                    best_pair, best_freq = pair, f
                    break

            if best_pair is None or best_freq < 2:
                print("\n   Tidak ada lagi pair yang valid, training berhenti.")
                break

            # --- Buat token baru ---
            a, b = best_pair
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges[best_pair] = new_id
            self.vocab_size = new_id + 1

            # --- Update word_vocab + pair_freq secara incremental ---
            # Untuk setiap kata yang mengandung pair (a,b):
            #   1. Hitung pair-diff antara kata lama dan kata baru
            #   2. Terapkan diff ke pair_freq
            #   3. Push entry baru ke heap untuk pair yang frekuensinya naik
            #
            # Ini JAUH lebih cepat daripada menghitung ulang seluruh pair_freq.
            changes: Dict[Tuple[int, int], int] = defaultdict(int)
            new_word_vocab: Dict[Tuple[int, ...], int] = {}

            for word, cnt in word_vocab.items():
                # Quick-reject: kalau 'a' tidak ada di word, tidak mungkin ada pair (a,b)
                if a not in word:
                    new_word_vocab[word] = cnt
                    continue

                # Cek apakah pair (a,b) benar-benar ada
                has_pair = any(
                    word[i] == a and i + 1 < len(word) and word[i + 1] == b
                    for i in range(len(word) - 1)
                )
                if not has_pair:
                    new_word_vocab[word] = cnt
                    continue

                # Hitung pairs sebelum merge
                old_pairs = self._word_to_pairs(word)

                # Apply merge: ganti setiap (a, b) → new_id
                new_word_list: List[int] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word_list.append(new_id)
                        i += 2
                    else:
                        new_word_list.append(word[i])
                        i += 1
                new_word = tuple(new_word_list)

                # Hitung pairs sesudah merge
                new_pairs = self._word_to_pairs(new_word)

                # Hitung diff dan akumulasi ke changes
                # Menggunakan Counter diff — handles semua edge case
                # termasuk pair yang overlap dan pair yang hilang sepenuhnya
                all_keys = set(old_pairs) | set(new_pairs)
                for pair in all_keys:
                    delta = (new_pairs.get(pair, 0) - old_pairs.get(pair, 0)) * cnt
                    if delta != 0:
                        changes[pair] += delta

                new_word_vocab[new_word] = cnt

            word_vocab = new_word_vocab

            # Hapus pair yang baru saja di-merge dari pair_freq
            pair_freq.pop(best_pair, None)

            # Apply semua perubahan ke pair_freq dan push ke heap
            for pair, delta in changes.items():
                if pair == best_pair:
                    continue
                new_f = pair_freq.get(pair, 0) + delta
                if new_f <= 0:
                    pair_freq.pop(pair, None)
                else:
                    pair_freq[pair] = new_f
                    if delta > 0:
                        # Hanya push jika frekuensi naik (lazy: yang lama akan terdeteksi stale)
                        heapq.heappush(heap, (-new_f, pair))

            pbar.update(1)
            if verbose and (new_id - start_id) % show_merge_every == 0 and new_id > start_id:
                elapsed = time.time() - t0
                speed   = (new_id - start_id) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "freq": best_freq,
                    "pairs": len(pair_freq),
                    "words": len(word_vocab),
                    "merge/s": f"{speed:.1f}",
                })

        pbar.close()

    # ==================================================================
    #  train_fast — training dari teks tunggal (menggantikan train())
    # ==================================================================
    def train_fast(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """
        Training BPE yang dioptimasi dari teks tunggal.
        Gunakan ini sebagai pengganti train() untuk kecepatan jauh lebih tinggi.
        """
        text = self.clean_text(text)
        self._add_special_tokens()
        self._add_byte_tokens()

        offset = len(self.special_tokens)

        # Bangun word vocabulary dari pre-tokens
        pre_tokens = self.pre_tokenize(text)
        word_vocab: Dict[Tuple[int, ...], int] = {}
        for pt in pre_tokens:
            word = tuple(offset + b for b in pt.encode("utf-8"))
            word_vocab[word] = word_vocab.get(word, 0) + 1

        total_words = sum(word_vocab.values())
        print(f"\n📊 Pre-tokenization selesai")
        print(f"   Unique words  : {len(word_vocab):,}")
        print(f"   Total tokens  : {total_words:,}")

        self._train_from_word_vocab(word_vocab, vocab_size, verbose)
        print(f"\n✅ Training selesai! Vocab size: {self.vocab_size:,}")

    # ------------------------------------------------------------------
    # train() lama — tetap tersedia, tapi jauh lebih lambat
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """Training klasik (flat list). Gunakan train_fast() untuk performa lebih baik."""
        text = self.clean_text(text)
        self._add_special_tokens()
        self._add_byte_tokens()
        pre_tokens = self.pre_tokenize(text)
        print(f"Pre-tokenization: {len(pre_tokens)} token awal\n")
        offset = len(self.special_tokens)
        tokens: List[int] = []
        for pt in pre_tokens:
            for b in pt.encode("utf-8"):
                tokens.append(offset + b)
        start_id = self.vocab_size
        print(f"Training dari {len(tokens):,} byte tokens...")
        for new_id in range(start_id, vocab_size):
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = new_id
            i, nt = 0, []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    nt.append(new_id)
                    i += 2
                else:
                    nt.append(tokens[i])
                    i += 1
            tokens = nt
            self.vocab_size = new_id + 1
            if verbose and (new_id < 700 or new_id % 100 == 0):
                print(f"Merge {new_id:5d} | Freq {pairs[best]:3d} | "
                      f"Pair: {self._pair_to_str(best):28} | Tokens: {len(tokens):6,}")
        print(f"✅ Training selesai! Vocab size: {self.vocab_size:,}")

    # ==================================================================
    #  train_from_dataset_fast — dataset training dengan multiprocessing
    # ==================================================================
    def train_from_dataset_fast(
        self,
        dataset_name: str = "indonesian-nlp/wikipedia-10k",
        subset: str = "wikipedia-id",
        vocab_size: int = 15000,
        max_examples: int = None,
        num_workers: int = 4,         # jumlah CPU core untuk pre-tokenization
        show_merge_every: int = 500,
        verbose: bool = True,
    ):
        """
        Training dari HuggingFace dataset dengan dua optimasi besar:
          1. ProcessPoolExecutor untuk pre-tokenization paralel
          2. train_from_word_vocab untuk training yang jauh lebih cepat

        Parameter:
          num_workers    : jumlah proses paralel (sesuaikan dengan jumlah core CPU)
          show_merge_every: tampilkan progress setiap N merge
        """
        print(f"🔄 Loading dataset: {dataset_name} / {subset}")
        dataset = load_dataset(dataset_name, subset, split="test")
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        total = len(dataset)
        print(f"✅ Dataset loaded : {total:,} articles\n")

        self._add_special_tokens()
        self._add_byte_tokens()
        offset = len(self.special_tokens)

        # --- Kumpulkan semua teks ---
        texts: List[str] = []
        for item in dataset:
            title   = item.get("title", "").strip()
            content = item.get("text",  "").strip()
            combined = f"{title}\n\n{content}" if title else content
            if len(combined.strip()) >= 100:
                texts.append(combined)

        print(f"🚀 Pre-tokenizing {len(texts):,} artikel dengan {num_workers} workers...")

        # --- Pre-tokenization paralel ---
        # Setiap worker menerima (teks, offset) dan mengembalikan word_freq dict.
        # Kita merge semua dict setelah selesai.
        args = [(text, offset) for text in texts]
        global_word_vocab: Dict[Tuple[int, ...], int] = {}

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm(
                executor.map(_pretokenize_worker, args, chunksize=max(1, len(args) // (num_workers * 4))),
                total=len(texts),
                desc="Pre-tokenizing",
                unit="art",
            )
            for word_freq in pbar:
                for word, cnt in word_freq.items():
                    global_word_vocab[word] = global_word_vocab.get(word, 0) + cnt
                pbar.set_postfix({"unique_words": f"{len(global_word_vocab):,}"})

        elapsed = time.time() - t0
        total_tokens = sum(global_word_vocab.values())
        print(f"\n✅ Pre-tokenization selesai dalam {elapsed:.1f}s")
        print(f"   Unique words : {len(global_word_vocab):,}")
        print(f"   Total tokens : {total_tokens:,}")

        # --- Training ---
        print(f"\n🧠 Training BPE → vocab size {vocab_size:,}...")
        self._train_from_word_vocab(global_word_vocab, vocab_size,
                                    verbose=verbose, show_merge_every=show_merge_every)

        total_time = time.time() - t0
        print(f"\n🎉 Selesai! Vocab Size = {self.vocab_size:,}")
        print(f"⏱️  Total waktu        = {total_time / 60:.1f} menit")
        return self

    # ------------------------------------------------------------------
    # train_from_dataset lama — tetap tersedia
    # ------------------------------------------------------------------
    def train_from_dataset(self, dataset_name="indonesian-nlp/wikipedia-10k",
                           subset="wikipedia-id", vocab_size=15000, verbose=True,
                           max_examples=None, show_merge_every=5000):
        """Versi lama (chunk-based, lebih lambat). Gunakan train_from_dataset_fast()."""
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
        pbar = tqdm(total=total_articles, desc="Processing articles", unit="art",
                    smoothing=0.1, dynamic_ncols=True)
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
        print(f"\n🎉 Training selesai! Vocab Size = {self.vocab_size:,}")
        return self

    def _train_from_tokens(self, tokens, vocab_size, verbose=True,
                           show_merge_every=5000, global_counter=0):
        if self.vocab_size == 0:
            self._add_special_tokens()
            self._add_byte_tokens()
        initial_size = self.vocab_size
        target_size  = min(vocab_size, initial_size + 2000)
        print(f"   Chunk: {len(tokens):,} tokens → target vocab {target_size:,}")
        pbar = tqdm(total=target_size - initial_size, desc="   Merging",
                    unit="merge", leave=False)
        start_time = time.time()
        for new_id in range(self.vocab_size, target_size):
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = new_id
            i, nt = 0, []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    nt.append(new_id)
                    i += 2
                else:
                    nt.append(tokens[i])
                    i += 1
            tokens = nt
            self.vocab_size = new_id + 1
            pbar.update(1)
        pbar.close()

    # ==================================================================
    #  SAVE & LOAD
    # ==================================================================
    def save_tokenizer(self, directory: str = "tokenizer_indonesia"):
        os.makedirs(directory, exist_ok=True)

        vocab: Dict[str, int] = {}
        vocab_id_set: set = set()

        for tok_str, tid in self.special_tokens.items():
            vocab[tok_str] = tid
            vocab_id_set.add(tid)

        offset = len(self.special_tokens)
        for i in range(256):
            repr_ = BYTE_TO_UNICODE[i]
            vocab[repr_] = offset + i
            vocab_id_set.add(offset + i)

        for tid in range(self.vocab_size):
            if tid in self.vocab and tid not in vocab_id_set:
                vocab[self._token_bytes_to_repr(self.vocab[tid])] = tid
                vocab_id_set.add(tid)

        merges = []
        for (id1, id2), _ in sorted(self.merges.items(), key=lambda x: x[1]):
            r1 = self._token_bytes_to_repr(self.vocab[id1])
            r2 = self._token_bytes_to_repr(self.vocab[id2])
            merges.append(f"{r1} {r2}")

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
            "model_type": "bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens.keys()),
            "trained_on": "indonesian-nlp/wikipedia-10k",
            "date": datetime.now().isoformat(),
        }
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ Tokenizer disimpan ke: {directory}/")
        print(f"   Vocab Size : {self.vocab_size:,}")
        print(f"   Merges     : {len(merges):,}")

    def load_tokenizer(self, directory: str = "tokenizer_indonesia"):
        tp = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(tp):
            raise FileNotFoundError(f"tokenizer.json tidak ditemukan di: {directory}")
        print(f"🔄 Loading tokenizer dari: {directory}")
        with open(tp, "r", encoding="utf-8") as f:
            data = json.load(f)
        model       = data.get("model", {})
        vocab_dict  = model.get("vocab", {})
        merges_list = model.get("merges", [])

        self.merges = {}
        self.vocab  = {}
        self.special_tokens = {}

        _known_special = set(self._SPECIAL_LIST)

        for token_repr, token_id in vocab_dict.items():
            token_id = int(token_id)
            if token_repr in _known_special:
                self.vocab[token_id] = token_repr.encode("utf-8")
                self.special_tokens[token_repr] = token_id
            else:
                try:
                    self.vocab[token_id] = self._repr_to_token_bytes(token_repr)
                except KeyError:
                    self.vocab[token_id] = token_repr.encode("utf-8")

        self.vocab_size = max(self.vocab.keys()) + 1 if self.vocab else 0

        repr_to_id: Dict[str, int] = {k: int(v) for k, v in vocab_dict.items()}
        loaded = 0
        for merge_str in merges_list:
            if " " not in merge_str:
                continue
            r1, r2 = merge_str.split(" ", 1)
            id1    = repr_to_id.get(r1)
            id2    = repr_to_id.get(r2)
            new_id = repr_to_id.get(r1 + r2)
            if id1 is not None and id2 is not None and new_id is not None:
                self.merges[(id1, id2)] = new_id
                loaded += 1

        print(f"✅ Tokenizer dimuat!")
        print(f"   Vocab Size     : {self.vocab_size:,}")
        print(f"   Special Tokens : {len(self.special_tokens)}")
        print(f"   Merges         : {loaded:,}")
        if loaded == 0:
            print("⚠️  Warning: 0 merges dimuat — hanya byte level.")
        return self


# ===========================================================================
if __name__ == "__main__":
    bpe = SimpleBPE()

    # ── Opsi 1: training dari teks kecil (untuk eksperimen) ──────────────────
    # sample = "Bahasa Indonesia adalah bahasa nasional Indonesia yang digunakan " * 200
    # bpe.train_fast(sample, vocab_size=500, verbose=True)

    # ── Opsi 2: training dari dataset Wikipedia (DIREKOMENDASIKAN) ───────────
    # Sesuaikan num_workers dengan jumlah core CPU Anda
    #
    #bpe.train_from_dataset_fast(
    #    dataset_name="indonesian-nlp/wikipedia-10k",
    #    subset="wikipedia-id",
    #    vocab_size=15000,
    #    max_examples=None,       # None = semua data
    #    num_workers=6,           # sesuaikan dengan CPU: os.cpu_count()
    #    show_merge_every=200,
    #)
    #bpe.save_tokenizer("tokenizer_indonesia")

    # ── Load & test ───────────────────────────────────────────────────────────
    bpe.load_tokenizer("tokenizer_indonesia")

    test = "Bahasa Indonesia adalah bahasa nasional yang kaya akan budaya."
    enc  = bpe.encode(test)
    dec  = bpe.decode(enc)

    print("\n=== TEST ===")
    print(f"Teks         : {test}")
    print(f"Token IDs    : {enc[:20]}{'...' if len(enc) > 20 else ''}")
    print(f"Jumlah token : {len(enc)}")
    print(f"Decode back  : {dec}")
    print(f"Round-trip OK: {test == dec}")
    stats = bpe.compression_ratio(test, enc)
    print(f"Kompresi     : {stats['compression_ratio']:.2f} bytes/token")