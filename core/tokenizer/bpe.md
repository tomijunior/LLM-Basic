Konsep Matematika Dasar BPE

Byte Pair Encoding (BPE) adalah algoritma kompresi berbasis frekuensi yang bekerja secara iteratif:

1. Mulai dengan vocabulary awal berupa 256 byte (0–255).
2. Pada setiap iterasi, kita mencari pasangan token yang paling sering muncul:$$(a, b) = \arg\max_{(x,y)} \text{freq}(x,y)$$
3. Gabungkan pasangan tersebut menjadi token baru dengan ID baru.
4. Ganti semua kemunculan $  (a, b)  $ dengan token baru tersebut.
5. Ulangi sampai ukuran vocabulary mencapai target.

Penjelasan Matematika Rasio Kompresi
Rasio kompresi token mengukur seberapa efektif tokenizer mengompresi teks.
Rumus utama:
$$\text{Compression Ratio} = \frac{\text{Jumlah Byte Awal (Original Length)}}{\text{Jumlah Token setelah Encoding}}$$

Semakin tinggi rasio → semakin baik kompresinya (semakin sedikit token).
Rasio 1.0 = tidak ada kompresi (setiap byte jadi 1 token).
Rasio 3.0–5.0 = kompresi yang cukup baik untuk LLM.
