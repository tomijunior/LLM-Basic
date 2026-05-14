from collections import Counter
from typing import List, Tuple, Dict
import regex as re

class SimpleBPE:
    """
    Implementasi Byte Pair Encoding (BPE) dari nol untuk tujuan pembelajaran.
    """
    
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}      # (id1, id2) -> new_id
        self.vocab: Dict[int, bytes] = {}                 # id -> byte representation
        self.vocab_size: int = 0
        self.special_tokens: Dict[str, int] = {}    # token_str -> id
    
    def clean_text(self, text: str) -> str:
        """Membersihkan teks sebelum pre-tokenization"""
        # Normalisasi line ending
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Hilangkan indentasi berlebih
        lines = [line.strip() for line in text.split('\n')]
        text = ' '.join(lines)
        
        # Hilangkan multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Hilangkan karakter kontrol tersisa
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
        
        return text.strip()

    # ===================== SPECIAL TOKENS (Granite Style) =====================
    def _add_special_tokens(self):
        """Special tokens seperti yang digunakan IBM Granite"""
        special_list = [
            "<|endoftext|>",   # 0 - juga dipakai sebagai pad, bos, eos
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
        """Tambahkan 256 byte tokens (0-255) setelah special tokens"""
        start_id = self.vocab_size
        for i in range(256):
            byte_id = start_id + i
            self.vocab[byte_id] = bytes([i])
        self.vocab_size += 256
        print(f"✅ Added 256 byte tokens (ID {start_id} - {self.vocab_size-1})")

    # ===================== PRE-TOKENIZATION (Granite Style) =====================
    def pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenizer yang mendekati IBM Granite:
        1. Pisahkan digit per karakter
        2. Gunakan regex ByteLevel GPT-2 style (yang dipakai Granite)
        """
        text = self.clean_text(text)

        # Step 1: Pisahkan setiap digit (individual_digits=True)
        text = re.sub(r'(\d)', r' \1 ', text)

        # Step 2: Regex ByteLevel yang paling mirip dengan Hugging Face (GPT-2 / Granite)
        # 2. Regex Pattern yang sangat mirip dengan ByteLevel + GPT-2 / Granite
        pattern = r"""
            (?i:'s|'t|'re|'ve|'m|'ll|'d)           # contractions
            | \p{L}+                                # letters (Unicode)
            | \p{N}+                                # numbers
            | [^\s\p{L}\p{N}]+                      # punctuation & symbols
            | \s+                                   # whitespace
        """
        
        tokens = re.findall(pattern, text, re.VERBOSE | re.UNICODE)
        return [token for token in tokens if token]  # remove empty

    def get_pairs(self, token_ids: List[int]) -> Counter:
        """
        Menghitung frekuensi semua pasangan token yang bersebelahan.
        
        Matematika:
        freq(a,b) = jumlah kemunculan a diikuti langsung oleh b
        """
        pairs = Counter()
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            pairs[pair] += 1
        return pairs

    def _pair_to_str(self, pair: Tuple[int, int]) -> str:
        """Mengubah pair (id1, id2) menjadi representasi string yang mudah dibaca"""
        try:
            t1 = self.vocab[pair[0]].decode('utf-8', errors='replace')
            t2 = self.vocab[pair[1]].decode('utf-8', errors='replace')
            
            # Escape karakter khusus agar tidak merusak tampilan
            t1 = repr(t1)[1:-1] if len(t1) > 0 else "∅"
            t2 = repr(t2)[1:-1] if len(t2) > 0 else "∅"
            
            return f"({t1} , {t2})"
        except:
            return f"ID({pair[0]},{pair[1]})"

    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """
        Melatih BPE Tokenizer.
        
        Parameter:
            vocab_size : ukuran vocabulary akhir (minimal 256)
        """
        text = self.clean_text(text)

        # Inisialisasi Vocabulary
        self._add_special_tokens()
        self._add_byte_tokens()

        # Pre-tokenization
        pre_tokens = self.pre_tokenize(text)
        print(f"Pre-tokenization menghasilkan {len(pre_tokens)} token awal.\n")

        offset = len(self.special_tokens)
        tokens: List[int] = []
        for pt in pre_tokens:
            for b in pt.encode("utf-8"):
                tokens.append(offset + b)   # mapping ke ID byte yang benar

        start_id = self.vocab_size
        print(f"Training dimulai dengan {len(tokens):,} byte tokens...\n")

        # Training Loop
        for new_id in range(start_id, vocab_size):
            
            pairs = self.get_pairs(tokens)
            
            if not pairs:
                break
                
            # Cari pasangan dengan frekuensi tertinggi
            best_pair = max(pairs, key=pairs.get)          # (a, b)
            freq = pairs[best_pair]
            
            if freq < 2:
                print("Tidak ada lagi pasangan yang muncul >1 kali.")
                break

            # === Merge ===
            # Buat token baru dengan menggabungkan byte representation
            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_token_bytes
            self.merges[best_pair] = new_id   # Simpan aturan merge

            # Ganti semua kemunculan (a,b) dengan new_id
            i = 0
            new_tokens = []
            while i < len(tokens):
                # Cek apakah pasangan saat ini adalah best_pair
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
            self.vocab_size = new_id + 1

            # Logging
            if verbose and (new_id < 700 or new_id % 100 == 0):
                pair_str = self._pair_to_str(best_pair)
                # Format yang lebih rapi dan tidak mudah terpotong
                print(f"Merge {new_id:5d} | Freq {freq:3d} | "
                      f"Pair: {pair_str:28} | "
                      f"Tokens: {len(tokens):6,}")

        print(f"✅ Training selesai! Vocab size akhir: {self.vocab_size:,}")

    # ===================== ENCODE & DECODE =====================
    def encode(self, text: str) -> List[int]:
        pre_tokens = self.pre_tokenize(text)
        token_ids = []
        
        for pt in pre_tokens:
            ids = list(pt.encode("utf-8"))
            # Terapkan semua merge rules
            for pair, new_id in self.merges.items():
                i = 0
                new_list = []
                while i < len(ids):
                    if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
                        new_list.append(new_id)
                        i += 2
                    else:
                        new_list.append(ids[i])
                        i += 1
                ids = new_list
            token_ids.extend(ids)
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        bytes_data = b""
        for tid in token_ids:
            bytes_data += self.vocab.get(tid, bytes([tid]))
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

    

# ================== CONTOH PENGGUNAAN ==================
if __name__ == "__main__":
    bpe = SimpleBPE()

    corpus = """Halo, ini adalah contoh teks panjang untuk belajar BPE tokenizer. 
    Bahasa Indonesia sangat kaya akan kosakata. 
    Kita sedang mempelajari algoritma Byte Pair Encoding dari dasar menggunakan pure Python."""

    bpe.train(corpus, vocab_size=800, verbose=True)

    test_text = """Selamat datang di dunia pembelajaran mesin. 
    Hari ini kita belajar tentang tokenizer dan rasio kompresi token."""

    encoded = bpe.encode(test_text)
    stats = bpe.compression_ratio(test_text, encoded)

    print(stats)

    print("\nTeks asli:", test_text)
    print("Token IDs:", encoded)