import numpy as np

# 1. Inisialisasi Parameter & Aktivasi
def softmax(x):
    # Stabil secara numerik dengan mengurangi nilai maksimum
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 2. Arsitektur Causal Self-Attention Manual
class NumPyCausalAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = 1.0 / np.sqrt(d_model)
        
        # Inisialisasi bobot (Weights) secara acak
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        """
        x shape: (T, d_model) -> Contoh: 1 batch teks dengan panjang T
        """
        self.x = x
        T, _ = x.shape
        
        # Langkah 1: Proyeksi Linear ke Query, Key, Value
        self.q = dot_product_attention_q = x @ self.Wq  # (T, d_model)
        self.k = dot_product_attention_k = x @ self.Wk  # (T, d_model)
        self.v = dot_product_attention_v = x @ self.Wv  # (T, d_model)
        
        # Langkah 2: Hitung Skor Atensi (Matmul Q dan Kᵀ)
        scores = (self.q @ self.k.T) * self.scale  # (T, T)
        
        # Langkah 3: Terapkan Causal Masking (Matriks Segitiga Bawah)
        mask = np.tril(np.ones((T, T)))
        scores = np.where(mask == 1, scores, -1e9) # Mengisi masa depan dengan -inf
        
        # Langkah 4: Aktivasi Softmax
        self.attention_weights = softmax(scores)  # (T, T)
        
        # Langkah 5: Kalikan dengan Value
        context = self.attention_weights @ self.v  # (T, d_model)
        
        # Langkah 6: Proyeksi Output Akhir
        out = context @ self.Wo  # (T, d_model)
        return out

    def backward(self, dout):
        """
        Backpropagation manual dari atas ke bawah (Reverse-mode Autograd)
        dout shape: (T, d_model) -> Gradien dari lapisan setelahnya
        """
        T, _ = self.x.shape
        context = self.attention_weights @ self.v
        
        # 1. Gradien terhadap Wo dan Konteks
        dWo = context.T @ dout
        dcontext = dout @ self.Wo.T
        
        # 2. Gradien terhadap Bobot Atensi dan Value (V)
        dattention_weights = dcontext @ self.v.T
        dv = self.attention_weights.T @ dcontext
        
        # 3. Gradien mundur melewati Softmax & Masking
        # Turunan softmax: dS = S * (dA - sum(dA * S))
        dscores = self.attention_weights * (dattention_weights - np.sum(dattention_weights * self.attention_weights, axis=-1, keepdims=True))
        mask = np.tril(np.ones((T, T)))
        dscores = np.where(mask == 1, dscores, 0.0) # Gradien area mask bernilai 0
        dscores *= self.scale
        
        # 4. Gradien terhadap Query (Q) dan Key (K)
        dq = dscores @ self.k
        dk = dscores.T @ self.q
        
        # 5. Gradien terhadap Matriks Bobot Proyeksi (Wq, Wk, Wv)
        dWq = self.x.T @ dq
        dWk = self.x.T @ dk
        dWv = self.x.T @ dv
        
        # 6. Gradien terhadap Input X
        dx = (dq @ self.Wq.T) + (dk @ self.Wk.T) + (dv @ self.Wv.T)
        
        return dx, dWq, dWk, dWv, dWo


# Simulasi dimensi data
T_konteks = 4   # Panjang kalimat (4 kata)
dimensi = 8     # Dimensi embedding

# 1. Generate data input acak (X) dan gradien fiktif dari loss function (dout)
X_input = np.random.randn(T_konteks, dimensi)
dout_fiktif = np.random.randn(T_konteks, dimensi)

# 2. Inisialisasi Lapisan Atensi NumPy
layer = NumPyCausalAttention(d_model=dimensi)

# 3. Jalankan Forward Pass
output = layer.forward(X_input)
print("1. Shape Output Forward:", output.shape)

# 4. Jalankan Backward Pass (Backprop)
dx, dWq, dWk, dWv, dWo = layer.backward(dout_fiktif)
print("2. Shape Gradien Input (dx):", dx.shape)

# 5. Pembaruan Bobot (Gradient Descent Sederhana)
learning_rate = 0.01
layer.Wq -= learning_rate * dWq
layer.Wk -= learning_rate * dWk
layer.Wv -= learning_rate * dWv
layer.Wo -= learning_rate * dWo
print("3. Bobot berhasil diperbarui dengan NumPy!")