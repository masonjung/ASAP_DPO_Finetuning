import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"mem_free/total_GB: {free/1e9:.2f}/{total/1e9:.2f}")

