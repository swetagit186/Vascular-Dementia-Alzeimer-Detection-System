import torch
from huggingface_hub import hf_hub_download

# Define the model architecture (this should match the model architecture used when saving the model)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Download the model file (.pth) from Hugging Face
model_path = hf_hub_download(
    repo_id="coilsnwdi34iu/Dementia-Model",  # replace with your actual repo ID
    filename="modelNew.pth"  # replace with your model filename
)

# Initialize the model
model = MyModel()

# Load the state dict (weights) into the model
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode (important for inference)
model.eval()

print("✅ Full model loaded successfully using PyTorch!")
