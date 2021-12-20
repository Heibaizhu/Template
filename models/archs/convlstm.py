import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size):
        super().__init__()

        self.convI = nn.Sequential(
            nn.Conv2d(inChannels + outChannels, outChannels, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.convF = nn.Sequential(
            nn.Conv2d(inChannels + outChannels, outChannels, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.convG = nn.Sequential(
            nn.Conv2d(inChannels + outChannels, outChannels, kernel_size, padding=kernel_size//2),
            nn.Tanh()
        )
        self.convO = nn.Sequential(
            nn.Conv2d(inChannels + outChannels, outChannels, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):
        """

        :param x: [B, inChannels, H, W]
        :param h: [B, outChannels, H, W]
        :param c: [B, outChannels, H, W]
        :return:
        """
        x = torch.cat((x, h), dim=1)
        i = self.convI(x)  # input gate
        f = self.convF(x)  # forget fate
        g = self.convG(x)
        o = self.convO(x)
        c = f * c + i * g # memory cell
        h = o * torch.tanh(c) # final state
        return h, c






class MyConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int   也是输出的维度
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(MyConvLSTMCell, self).__init__()

        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.kernel_size = kernel_size
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=(kernel_size[0]//2, kernel_size[1]//2),
                              bias=bias)

    def forward(self, lstm_input, cur_state):
        """

        :param lstm_input: [x, h_cur]
        :param cur_state: 历史状态
        :return:
        """
        # h_cur, c_cur = cur_state

        # combined = torch.cat([lstm_input, h_cur], dim=1)  # concatenate along channel axis

        lstm_input = self.conv(lstm_input)
        cc_i, cc_f, cc_o, cc_g = torch.split(lstm_input, self.hidden_dim, dim=1)
        # i = torch.sigmoid(cc_i)  #(B, hidden_dim, H, W)
        # f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        # g = torch.tanh(cc_g)

        c_next = torch.sigmoid(cc_f) * cur_state + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        h_next = torch.sigmoid(cc_o) * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

