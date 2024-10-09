import time
from pylsl import StreamInlet, resolve_stream
import torch
from torch import nn
from torch.nn import functional as F
class_names = ['Girt', 'Blink', 'Rest']
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # 第一层卷积，使用一维卷积
        self.conv1 = nn.Conv1d(in_channels=14, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=16)

        # 深度卷积层
        self.depth_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, groups=16)
        self.batchnorm2 = nn.BatchNorm1d(num_features=32)

        # 分离卷积层
        self.separable_conv = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.batchnorm3 = nn.BatchNorm1d(num_features=32)

        # 平均池化层
        self.avgpool = nn.AvgPool1d(kernel_size=4)

        # 全连接层
        self.fc = nn.Linear(32 * 32 * 3, len(class_names))  # 640 / 4 = 160

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        x = F.relu(self.depth_conv(x))
        x = self.batchnorm2(x)

        x = F.relu(self.separable_conv(x))
        x = self.batchnorm3(x)

        x = self.avgpool(x)

        # 展平特征图以输入到全连接层
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
model = EEGNet()
# Load the trained model
model_path = './model/blink_girt_3s_still_10min_3classifications_0903.pth'
model=torch.load(model_path)
model.eval()

print("looking for a stream...")
# First resolve a Motion stream on the lab network
streams = resolve_stream('type', 'EEG')
print(streams)

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Define parameters
sampling_rate = 128  # Hz
window_duration = 3  # seconds
window_size = sampling_rate * window_duration  # Number of samples in a 3-second window
channel_indices = list(range(3, 17))  # Select channels from 4th to 17th (0-based index)

buffer = []  # To store the EEG data for 3 seconds

while True:
    # Pull sample
    sample, timestamp = inlet.pull_sample()

    if timestamp is not None:
        # Extract relevant EEG channels (from 4th to 17th element)
        eeg_data = [sample[i] for i in channel_indices]
        buffer.append(eeg_data)

        # If we have collected enough data for 3 seconds, classify the data
        if len(buffer) >= window_size:
            # Convert buffer to tensor for model input
            data_tensor = torch.tensor(buffer).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 384, 14)
            data_tensor = data_tensor.squeeze(0).permute(0, 2, 1)  # Shape: [1, 14, 384]
            # Perform classification
            with torch.no_grad():
                output = model(data_tensor)
                _, predicted_class = torch.max(output, 1)
                print(f"Predicted Class: {predicted_class.item()}")

            # Clear the buffer for the next window
            buffer = []

    # Add a small sleep to avoid overwhelming the CPU
    time.sleep(1 / sampling_rate)