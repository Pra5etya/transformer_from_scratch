import torch

print("Versi PyTorch:", torch.__version__)

cuda_availibility = torch.cuda.is_available()
print("Apakah CUDA tersedia?", cuda_availibility, '\n')

if torch.cuda.is_available():
    print("Jumlah GPU terdeteksi:", torch.cuda.device_count(), '\n')
    print("Nama GPU:", torch.cuda.get_device_name(0))

else:
    print("GPU CUDA tidak tersedia. Menggunakan CPU.")