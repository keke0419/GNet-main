from Attention import *

class GlobalExpectationPooling1D(nn.Module):
    """Global Expect pooling operation for temporal data.
        # Arguments
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, steps, features)` while `channels_first`
                corresponds to inputs with shape
                `(batch, features, steps)`.
            mode: int
            m_trainable: A boolean variable,
                if m_trainable == True, the base will be trainable,
                else the base will be a constant
            m_value: A integer,
                the value of the base to calculate the prob
        # Input shape
            `(batch_size, steps, features,)`
        # Output shape
            2D tensor with shape:
            `(batch_size, features)`
        """

    def __init__(self, mode=0, m_trainable=False, m_value=49):
        super(GlobalExpectationPooling1D, self).__init__()
        self.m_value = m_value
        self.mode = mode
        self.m_trainable = m_trainable
        self.m = nn.Parameter(torch.randn((m_value,m_value)))

    def get_config(self):
        base_config = super(GlobalExpectationPooling1D, self).get_config()
        return dict(list(base_config.items()))

    def forward(self, x):
        if self.mode == 0:
            # transform the input
            now = x
            # print(now.shape)
            # x = x - max(x)
            max = torch.max(now, dim=-1, keepdim=True)[0]
            diff_1 = torch.sub(max, now)
            # x = mx
            diff = torch.matmul(diff_1, self.m)
            # prob =  exp(x_i)/sum(exp(x_j))
            prob = nn.Softmax(dim=-1)(diff)
            # Expectation = sum(Prob*x)
            expectation = torch.sum(torch.matmul(now, prob.permute(0, 2, 1)), dim=-1, keepdim=False)
            expectation = expectation.reshape(expectation.size()[0], expectation.size()[1], 1)
        else:
            # transform the input
            now = x.permute(0, 2, 1)
            # x  - mean(x)
            now_diff = now.sub(torch.max(now, dim=-1, keepdim=True)[0])
            # x = mx
            now_diff_m = torch.matmul(now_diff, self.m)
            # sgn(x)
            sgn_now = torch.sign(now_diff_m)
            # exp(x - mean) * sgn(x - mean(x))  + exp(x - mean(x))
            diff_2 = torch.add(torch.matmul(sgn_now, torch.exp(now_diff_m)), torch.exp(now_diff_m))
            # x = x/2
            diff_now = torch.div(diff_2, 2)
            # Prob = exp(x) / sum(exp(x))
            prob = diff_now / torch.sum(diff_now, dim=-1, keepdim=True)
            expectation = torch.sum(torch.matmul(now, prob), dim=-1, keepdim=False)

        return expectation

def upsample(x, out_size1):
    return F.interpolate(x, size=out_size1, mode='linear', align_corners=False)

def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False, groups=1):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))

def Highway1(X):
    X, H = X[:, :64, :], X[:, 64:, :]
    T = nn.Sigmoid()(X)
    out1 = (1-T) * X + T * H
    return out1

class FCNsignal(nn.Module):
    """FCNsignal for predicting base-resolution binding signals"""
    def __init__(self, motiflen=16):
        super(FCNsignal, self).__init__()
        # encoding
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.linear = nn.Linear(in_features=64, out_features=64, bias=True)
        # decoding
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # general functions
        self.relu = nn.ELU(alpha=0.1, inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encoding
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        out1_1, _ = self.gru1(out1)
        out1_2, _ = self.gru2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.gru_drop(out1)
        skip4 = out1.permute(0, 2, 1)
        up5 = self.aap(skip4)
        # decoding
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)

        return out_dense

class BPNet(nn.Module):
    """building BPNet on the Pytorch platform."""
    def __init__(self, motiflen=25, batchnormalization=False):
        super(BPNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen, padding=motiflen // 2)
        self.relu1 = nn.ReLU(inplace=True)
        # sequential model
        self.sequential_model = nn.ModuleList()
        for i in range(1, 10):
            if batchnormalization:
                self.sequential_model.append((nn.Sequential(
                    nn.BatchNorm1d(64),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
            else:
                self.sequential_model.append((nn.Sequential(
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
        self.convtranspose1 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=motiflen, padding=motiflen // 2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):

        """Construct a new computation graph at each froward"""
        b, c, l = data.size()
        x = self.conv1(data)
        x = self.relu1(x)
        for module in self.sequential_model:
            conv_x = module(x)
            x = conv_x + x
        bottleneck = x
        out = self.convtranspose1(bottleneck)
        # out = self.sigmoid(out)

        return out

class GNet(nn.Module):
    """ """
    def __init__(self, motiflen=16):
        super(GNet, self).__init__()
        # encoding
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        # decoding
        self.atten2 = DualAttention(64, 64)
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # general functions
        self.relu = nn.ELU(alpha=0.1, inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encoding
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = Highway1(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        T = self.sigmoid(out1)
        out1 = (1 - T) * skip2 + T * out1
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        T = self.sigmoid(out1)
        out1 = (1 - T) * skip3 + T * out1
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        k1, _ = self.gru1(out1)
        v1, _ = self.gru2(torch.flip(out1, [1]))
        v1 = torch.flip(v1, [1])
        out1_T = nn.Sigmoid()(k1)
        out1 = (1 - out1_T) * v1 + out1_T * k1
        # out1 = k1 + v1
        out1 = self.atten2(out1)
        # out1 = self.atten1(out1)
        out1 = self.gru_drop(out1)
        skip4 = out1.permute(0, 2, 1)
        up5 = self.aap(skip4)
        # decoding
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)

        return out_dense