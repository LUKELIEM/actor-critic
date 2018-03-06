import torch


class Policy(torch.nn.Module):

    def __init__(self, input_channels, num_actions):
        super(Policy, self).__init__()
        self.temperature = 1.0
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.features = self._init_features()
        self.lstm = self._init_lstm()
        self.action_head = self._init_action_head()
        self.value_head = self._init_value_head()
        
        self.saved_actions = []
        self.rewards = []

    def _init_features(self):
        layers = []
        # 80 x 80 x in_channels initial dimensions 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=8, stride=4, padding=2))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # 20 x 20 x 16 feature maps
        layers.append(torch.nn.Conv2d(16,
                                      32, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 10 x 10 x 32 feature maps
        layers.append(torch.nn.Conv2d(32,
                                      32, kernel_size=4, stride=2, padding=0))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 4 x 4 x 32 feature maps
        return torch.nn.Sequential(*layers)
    
    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        action = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        value = self.value_head(x)
        return action, value, (hx, cx)
