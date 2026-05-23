import torch
import torch.nn as nn
import os

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    def forward(self, x):
        f = self.shared(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))

if __name__ == "__main__":
    device = torch.device("cpu")
    model = DuelingDQN(10, 2).to(device)
    
    # Load PyTorch weights
    model_path = "dino_ddqn_dueling_best.pth"
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, 10, device=device)
    
    # Export to ONNX directly into current directory
    onnx_path = "dino.onnx"
    print(f"Exporting ONNX to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['state'], 
        output_names=['q_values'], 
        dynamic_axes={'state': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
    )
    print("Export completed successfully!")
