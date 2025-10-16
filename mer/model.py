import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming const.py contains these constants
# from .const import SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH

class SimpleDenseModel(nn.Module):
    def __init__(self, max_timestep, n_freq, n_channel, batch_size, **kwargs):
        super().__init__()
        self.max_timestep = max_timestep
        self.n_freq = n_freq
        self.n_channel = n_channel
        self.batch_size = batch_size
        
        # Calculate flattened size after resize
        flattened_size = n_freq * 1024 * n_channel
        
        self.resize = nn.Upsample(size=(n_freq, 1024), mode='bilinear', align_corners=False)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(flattened_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 2)
        
    def forward(self, x):
        # x shape: (batch, max_timestep, n_freq, n_channel)
        # Permute to (batch, n_channel, max_timestep, n_freq) for resize
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.relu(self.dense5(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.conv1 = nn.Conv2d(neurons // 2 if neurons > 64 else 2, neurons, kernel_size=3, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(neurons, neurons // 2, kernel_size=1, padding=0)
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class SimpleConvModel(nn.Module):
    def __init__(self, max_timestep, n_freq, n_channel, batch_size, **kwargs):
        super().__init__()
        self.max_timestep = max_timestep
        self.n_freq = n_freq
        self.n_channel = n_channel
        self.batch_size = batch_size
        
        neuron_conv = [64, 128, 256, 512, 1024]
        
        self.resize = nn.Upsample(size=(n_freq, 512), mode='bilinear', align_corners=False)
        
        # Build conv blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = n_channel
        for neurons in neuron_conv:
            block = nn.Sequential(
                nn.Conv2d(in_channels, neurons, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(neurons, neurons // 2, kernel_size=1, padding=0),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.1)
            )
            self.conv_blocks.append(block)
            in_channels = neurons // 2
        
        # Calculate flattened size (after multiple pooling operations)
        # After 5 maxpool(2,2): n_freq/32 * 512/32 * (1024//2)
        flat_size = (n_freq // 32) * (512 // 32) * (1024 // 2)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(flat_size, 128)
        self.dense2 = nn.Linear(128, 2)
        
    def forward(self, x):
        # x shape: (batch, max_timestep, n_freq, n_channel)
        # Permute to (batch, n_channel, max_timestep, n_freq)
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, neurons):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, neurons, kernel_size=5, padding=0)
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(neurons, neurons // 2, kernel_size=1, padding=0)
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class SimpleCRNN(nn.Module):
    def __init__(self, spectrogram_time_length, frequency_length):
        super().__init__()
        self.spectrogram_time_length = spectrogram_time_length
        self.frequency_length = frequency_length
        
        self.resize = nn.Upsample(size=(frequency_length, 1024), mode='bilinear', align_corners=False)
        
        # Conv blocks
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 128, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 256, kernel_size=5, padding=0)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(128, 512, kernel_size=5, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.maxpool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv9 = nn.Conv2d(256, 512, kernel_size=2, padding=0)
        
        self.dropout = nn.Dropout(0.1)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 2)
        
    def forward(self, x):
        # x shape: (batch, time, freq, channel)
        # Permute to (batch, freq, time, channel)
        x = x.permute(0, 2, 1, 3)
        # Permute to (batch, channel, freq, time)
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        
        # Conv block 1
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = self.maxpool1(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = self.maxpool2(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = self.maxpool3(x)
        x = self.dropout(x)
        
        # Conv block 4
        x = F.leaky_relu(self.conv7(x), 0.1)
        x = F.leaky_relu(self.conv8(x), 0.1)
        x = self.maxpool4(x)
        x = self.dropout(x)
        
        # Additional pooling and conv
        x = self.maxpool5(x)
        x = F.leaky_relu(self.conv9(x), 0.1)
        x = self.dropout(x)
        
        # Reshape for LSTM: (batch, channels, height, width) -> (batch, width, height*channels)
        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 2, 1)  # (batch, width, height, channels)
        x = x.reshape(batch, width, -1)  # (batch, 59, 512)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x


class SimpleCRNN2(nn.Module):
    def __init__(self, spectrogram_time_length, frequency_length):
        super().__init__()
        self.spectrogram_time_length = spectrogram_time_length
        self.frequency_length = frequency_length
        
        self.resize = nn.Upsample(size=(frequency_length, 1024), mode='bilinear', align_corners=False)
        
        # Conv blocks with ReLU
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 128, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 256, kernel_size=5, padding=0)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(128, 512, kernel_size=5, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.1)
        
        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(1024, 128, batch_first=True, bidirectional=True)
        
        self.dense1 = nn.Linear(256, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, 2)
        
    def forward(self, x):
        # x shape: (batch, time, freq, channel)
        # Permute to (batch, freq, time, channel)
        x = x.permute(0, 2, 1, 3)
        # Permute to (batch, channel, freq, time)
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool3(x)
        x = self.dropout(x)
        
        # Conv block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.maxpool4(x)
        x = self.dropout(x)
        
        # Reshape for LSTM
        # Permute: (batch, channel, height, width) -> (batch, width, height, channel)
        x = x.permute(0, 3, 2, 1)
        batch, width, height, channels = x.shape
        x = x.reshape(batch, width, height * channels)  # (batch, 60, 1024)
        
        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        return x


class SimpleCRNN3(nn.Module):
    """CRNN that uses GRU"""
    def __init__(self, spectrogram_time_length, frequency_length):
        super().__init__()
        self.spectrogram_time_length = spectrogram_time_length
        self.frequency_length = frequency_length
        
        self.resize = nn.Upsample(size=(frequency_length, 1024), mode='bilinear', align_corners=False)
        
        # Conv blocks with ReLU
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 128, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 256, kernel_size=5, padding=0)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(128, 512, kernel_size=5, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.1)
        
        # GRU layers
        self.gru1 = nn.GRU(1024, 256, batch_first=True)
        self.gru2 = nn.GRU(256, 128, batch_first=True)
        self.gru3 = nn.GRU(128, 64, batch_first=True)
        
        self.dense1 = nn.Linear(64, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, 2)
        
    def forward(self, x):
        # x shape: (batch, time, freq, channel)
        # Permute to (batch, freq, time, channel)
        x = x.permute(0, 2, 1, 3)
        # Permute to (batch, channel, freq, time)
        x = x.permute(0, 3, 1, 2)
        x = self.resize(x)
        
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool3(x)
        x = self.dropout(x)
        
        # Conv block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.maxpool4(x)
        x = self.dropout(x)
        
        # Reshape for GRU
        # Permute: (batch, channel, height, width) -> (batch, width, height, channel)
        x = x.permute(0, 3, 2, 1)
        batch, width, height, channels = x.shape
        x = x.reshape(batch, width, height * channels)  # (batch, 60, 1024)
        
        # GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        return x

