import torch
import onnxruntime as ort

print("Torch CUDA:", torch.cuda.is_available())
print("Torch GPU:", torch.cuda.get_device_name(0))
print("ONNX Providers:", ort.get_available_providers())