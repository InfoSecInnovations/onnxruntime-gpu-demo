import torch
import onnxruntime

print("torch cuda is available: " + "Yes" if torch.cuda.is_available() else "No")

print("torch cuda version " + torch.version.cuda)
print("torch cudnn version: " + str(torch.backends.cudnn.version()))

print("preload DLLs")
onnxruntime.preload_dlls()
print(onnxruntime.get_available_providers())