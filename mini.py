import torch
import torch.nn as nn
import torch.nn.functional as F
from core.tokenizer.simple_bpe_optimized import SimpleBPE
from datasets import load_dataset
from tqdm import tqdm  # Library untuk progress bar

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # dim harus genap karena vektor diproses berpasangan (x, y)
        self.dim = dim
        
        # Hitung kebalikan frekuensi: 1 / (theta ** (2i / dim))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Buat matriks posisi (0, 1, 2, ..., max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Hitung perkalian luar (outer product) untuk mendapatkan sudut rotasi
        freqs = torch.outer(t, self.inv_freq) # (max_seq_len, dim // 2)
        
        # Gandakan frekuensi untuk mencocokkan total dimensi token
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, dim)
        
        # Simpan nilai cosinus dan sinus sebagai cache
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        # Memutar setengah bagian vektor untuk rumus rotasi matriks 2D
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        # x shape: (Batch, Head, Seq_Len, Head_Dim)
        cos = self.cos_cached[:seq_len, :].get_device() # Ambil sesuai panjang teks
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # (1, 1, Seq_Len, Head_Dim)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # (1, 1, Seq_Len, Head_Dim)
        
        # Rumus RoPE: R = X * cos(theta) + rotate_half(X) * sin(theta)
        return (x * cos) + (self._rotate_half(x) * sin)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Lebih efisien di CPU dibanding LayerNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class TransformerBlockWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads"
        
        # 1. Komponen RoPE
        self.rope = RotaryPositionEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        
        # 2. Proyeksi Linear untuk Attention
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # 3. Normalisasi Lapisan (RMSNorm)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        
        # 4. SwiGLU / Feed-Forward Network (Standar LLM Modern)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(), # Aktivasi Swish/SiLU
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # B = Batch Size, T = Sequence Length (Konteks), C = d_model
        B, T, C = x.shape
        
        # --- BAGIAN 1: CAUSAL MULTI-HEAD ATTENTION DENGAN ROPE ---
        # Simpan salinan x untuk koneksi residual
        residual = x
        x_norm = self.attention_norm(x)
        
        # Proyeksi Q, K, V dan ubah dimensi menjadi Multi-Head
        # Hasil bentuk: (B, num_heads, T, head_dim)
        q = self.wq(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # TERAPKAN RoPE: Hanya pada Query (Q) dan Key (K) [1, 2]
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)
        
        # Hitung skor Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Buat Causal Mask agar model tidak melihat masa depan
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Hitung Probabilitas Atensi & Kalikan dengan Value (V)
        attention_weights = F.softmax(scores, dim=-1)
        context = attention_weights @ v # (B, num_heads, T, head_dim)
        
        # Gabungkan kembali semua kepala (Heads)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        attention_output = self.wo(context)
        
        # Tambahkan residual pertama
        x = residual + attention_output
        
        # --- BAGIAN 2: FEED-FORWARD NETWORK (FFN) ---
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        
        # Tambahkan residual kedua
        output = residual + ffn_output
        
        return output
    

from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    def __init__(self, token_ids, context_length):
        """
        token_ids: List atau Numpy Array berisi ID token hasil dari Tokenizer Anda.
        context_length: Jumlah token maksimal yang dibaca model dalam satu waktu.
        """
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        # Dikurangi context_length karena potongan terakhir harus menyisakan ruang untuk target Y
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx):
        # Input X: Ambil token dari indeks saat ini sepanjang context_length
        x = self.token_ids[idx : idx + self.context_length]
        
        # Target Y: Ambil token yang sama persis, tetapi digeser 1 langkah ke depan
        y = self.token_ids[idx + 1 : idx + self.context_length + 1]
        
        return x, y



# --- MASUKKAN KODE RMSNorm, RotaryPositionEmbedding, DAN TransformerBlockWithRoPE DI SINI ---
# (Gunakan implementasi dari langkah kita sebelumnya)

class RoPETansformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()
        # Token Embedding: Mengubah 15.000 kemungkinan ID menjadi vektor berdimensi d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Blok Transformer Penuh dengan RoPE & SwiGLU (Bisa ditumpuk jika ingin model lebih dalam)
        self.transformer_block = TransformerBlockWithRoPE(d_model, num_heads, max_seq_len)
        
        # Normalisasi Akhir sebelum proyeksi kata
        self.final_norm = RMSNorm(d_model)
        
        # Language Model Head: Mengubah kembali vektor d_model menjadi 15.000 nilai prediksi (logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Ikat bobot embedding dengan lm_head (Weight Tying) untuk menghemat memori & mempercepat konvergensi CPU
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # idx shape: (Batch_Size, Context_Length)
        _, T = idx.shape
        
        # 1. Konversi token ke vektor
        x = self.token_embedding(idx) # (B, T, d_model)
        
        # 2. Alirkan lewat blok Transformer (RoPE diaplikasikan di dalam blok ini)
        x = self.transformer_block(x) # (B, T, d_model)
        
        # 3. Normalisasi akhir
        x = self.final_norm(x)
        
        # 4. Proyeksi ke ruang kosakata untuk prediksi kata berikutnya
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Bentuk ulang matriks (Flatten) agar sesuai dengan standar fungsi CrossEntropyLoss PyTorch
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    

if __name__ == "__main__":
    bpe = SimpleBPE()
    bpe.load_tokenizer("core/tokenizer/tokenizer_indonesia")

    # --- 1. KONFIGURASI PARAMETER ---
    VOCAB_SIZE = bpe.get_vocab_size()       # Sesuai spesifikasi Tokenizer Anda
    D_MODEL = 64             # Dimensi vektor (Gunakan angka kecil seperti 64/128 agar CPU tidak berat)
    NUM_HEADS = 4            # Head_dim = 64 // 4 = 16 (Harus genap untuk kebutuhan RoPE)
    MAX_SEQ_LEN = 128        # Batas kapasitas memori posisi model
    CONTEXT_LENGTH = 32      # Panjang potongan teks per baris saat dilatih
    BATCH_SIZE = 4           # Jumlah sampel teks yang diproses bersamaan di CPU
    EPOCHS = 2               # Berapa kali model mengulang membaca seluruh dataset
    LEARNING_RATE = 5e-4     # Kecepatan penyesuaian bobot

    # ==========================================
    # 1. PEMUATAN DATASET HUGGING FACE
    # ==========================================
    dataset_name: str = "indonesian-nlp/wikipedia-10k"
    subset: str = "wikipedia-id"
    max_examples = 5  # Batasi jumlah artikel awal agar memori CPU tidak penuh

    # Memuat dataset
    dataset = load_dataset(dataset_name, subset, split="test")
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    total = len(dataset)
    print(f"✅ Dataset loaded : {total:,} articles\n")

    # ==========================================
    # 2. PROSES TOKENISASI SELURUH ARTIKEL
    # ==========================================
    print("🔄 Memulai proses tokenisasi teks Wikipedia...")
    semua_token_ids = []

    # Asumsi: Anda memiliki objek tokenizer bernama `my_tokenizer`
    # Ganti `my_tokenizer.encode()` dengan fungsi encode/tokenize milik Anda sendiri
    for i, item in enumerate(dataset):
        teks_artikel = item['text'] # Mengambil kolom teks dari baris artikel
        
        # Ubah teks mentah menjadi daftar ID angka menggunakan Tokenizer Anda
        token_ids = bpe.encode(teks_artikel) 
        
        # Satukan seluruh token dari semua artikel ke dalam satu list besar
        semua_token_ids.extend(token_ids)
        
        if (i + 1) % 100 == 0:
            print(f"   [Sudah memproses {i + 1}/{total} artikel]")

    print(f"✅ Tokenisasi Selesai! Total corpus: {len(semua_token_ids):,} token.")

    # Inisialisasi Dataset dan Dataloader
    dataset_training = LLMDataset(semua_token_ids, context_length=CONTEXT_LENGTH)
    dataloader = DataLoader(dataset_training, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. INISIALISASI MODEL & OPTIMIZER ---
    model = RoPETansformerLLM(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
    )

    # Gunakan AdamW yang dioptimalkan untuk arsitektur Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print("=== MEMULAI PROSES TRAINING PADA CPU ===")
    model.train() # Setel model ke mode pelatihan

    for epoch in range(EPOCHS):
        total_loss = 0

        # Bungkus dataloader dengan tqdm untuk memunculkan progress bar di terminal
        # 'desc' memberikan label di sisi kiri, 'leave=True' mempertahankan bar saat epoch selesai
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
            
            # Langkah 1: Jalankan maju untuk mendapatkan prediksi dan nilai eror (Loss)
            logits, loss = model(X_batch, Y_batch)
            
            # Langkah 2: Bersihkan sisa gradien pada iterasi sebelumnya
            optimizer.zero_grad()
            
            # Langkah 3: Backpropagation otomatis oleh mesin Autograd CPU
            loss.backward()
            
            # Batasi gradien (Gradient Clipping) agar kalkulasi di CPU stabil dan tidak meledak (NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Langkah 4: Perbarui bobot matriks internal model
            optimizer.step()
            
            total_loss += loss.item()
            loss_sekarang = loss.item()
            
            # Perbarui informasi statistik di sisi kanan progress bar secara real-time
            progress_bar.set_postfix({
                "Loss": f"{loss_sekarang:.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
        rata_rata_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Rata-rata Loss: {rata_rata_loss:.4f}")

    print("\nModel selesai dilatih dari nol di CPU!")

    # Tentukan nama file penyimpanan (gunakan format snake_case sesuai kaidah Python)
    NAMA_FILE_MODEL = "rope_transformer_llm_v1.pt"

    # Ambil state_dict (matriks bobot) dari model Anda
    bobot_model = model.state_dict()

    # Simpan ke dalam penyimpanan komputer Anda
    torch.save(bobot_model, NAMA_FILE_MODEL)

    print(f"💾 Berhasil menyimpan bobot model ke file: '{NAMA_FILE_MODEL}'")

    # 1. Anda WAJIB menginisialisasi struktur arsitektur model kosong terlebih dahulu
    # Pastikan parameter dimensi (VOCAB_SIZE, D_MODEL, dll.) sama persis seperti saat melatih
    model_uji = RoPETansformerLLM(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        num_heads=NUM_HEADS, 
        max_seq_len=MAX_SEQ_LEN
    )

    # 2. Muat file biner bobot dari penyimpanan komputer
    bobot_tersimpan = torch.load("rope_transformer_llm_v1.pt", map_location="cpu")

    # 3. Masukkan bobot tersebut ke dalam struktur model kosong
    model_uji.load_state_dict(bobot_tersimpan)

    # 4. Setel model ke mode evaluasi/pengujian (mematikan fungsi pelatihan internal)
    model_uji.eval()

    print("✅ Model berhasil dimuat kembali dan siap digunakan untuk menghasilkan teks!")