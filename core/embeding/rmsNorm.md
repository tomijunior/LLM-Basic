Rumus Matematika RMSNorm
$$RMSNorm(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma$$
Dimana:

$  x  $ = input vector
$  \gamma  $ = parameter learnable (scaling factor)
$  \epsilon  $ = kecil (biasa 1e-5 atau 1e-6) untuk menghindari division by zero

Catatan penting: RMSNorm tidak mengurangi mean (tidak ada centering), hanya melakukan scaling berdasarkan RMS.