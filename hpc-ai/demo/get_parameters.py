import torch
from torchvision.models import resnet152
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model, move to GPU, and set to evaluation mode
model = resnet152().to(device)
model.eval()

# Define input tensor (batch_size=1, 3 color channels, 224x224 image), move to GPU
input = torch.randn(1, 3, 224, 224).to(device)

# Compute FLOPs and parameter count
flops = FlopCountAnalysis(model, input)
print(f"Total FLOPs: {flops.total()/1e9:.2f} GFLOPs (forward pass)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total No. Parameters: {total_params}")
