import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, h_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())
    #
    # def init_hidden(self, batch_size, height, width):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)),
    #             Variable(torch.zeros(batch_size, self.hidden_dim, height, width)))


class ConvLSTMCellReLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLU, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        # combined_conv = self.bn(combined_conv)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = cc_g

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)

        return h_next, c_next, h_next


class ConvLSTMCellReLUSkip(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLUSkip, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_f = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_i = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_g = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_y = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=(1,1),
                                padding=(0,0),
                                bias=self.bias)


        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        i = torch.sigmoid(self.conv_i(h_and_x))
        f = torch.sigmoid(self.conv_f(h_and_x))
        o = torch.sigmoid(self.conv_o(h_and_x))
        g = self.conv_g(h_and_x)

        c_next = f * c_cur + i * g
        h_hat = o * c_next
        h_next = F.relu(h_hat)

        h_and_x_and_h_hat = torch.cat([input_tensor, h_cur, h_hat], dim=1)  # concatenate along channel axis
        y = F.relu(self.conv_y(h_and_x_and_h_hat))

        return h_next, c_next, y


class ConvLSTMCellGeneral(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, activation='tanh', peephole=False, skip=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellGeneral, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.peephole = peephole
        self.skip = skip

        if activation == 'tanh':
            self.act_g = F.tanh
            self.act_c = F.tanh
        elif activation == 'relu':
            self.act_g = lambda x: x
            self.act_c = F.relu
        else:
            assert False, 'unknown activation function'

        self.conv_f = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_i = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_g = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        if skip:
            self.conv_y = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=(1,1),
                                    padding=(0,0),
                                    bias=self.bias)

        if peephole:
            self.w_cf = nn.Parameter(torch.Tensor(self.hidden_dim))
            self.w_ci = nn.Parameter(torch.Tensor(self.hidden_dim))
            self.w_co = nn.Parameter(torch.Tensor(self.hidden_dim))
            stdv = 1. / math.sqrt(self.hidden_dim)
            self.w_cf.data.uniform_(-stdv, stdv)
            self.w_ci.data.uniform_(-stdv, stdv)
            self.w_co.data.uniform_(-stdv, stdv)


        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        if self.peephole:
            i = torch.sigmoid(self.conv_i(h_and_x) + c_cur * self.w_ci)
            f = torch.sigmoid(self.conv_f(h_and_x) + c_cur * self.w_cf)
            o = torch.sigmoid(self.conv_o(h_and_x) + c_cur * self.w_co)
        else:
            i = torch.sigmoid(self.conv_i(h_and_x))
            f = torch.sigmoid(self.conv_f(h_and_x))
            o = torch.sigmoid(self.conv_o(h_and_x))

        g = self.act_g(self.conv_g(h_and_x))

        c_next = f * c_cur + i * g

        if self.skip:
            h_hat = o * c_next
            h_next = self.act_c(h_hat)

            h_and_x_and_h_hat = torch.cat([input_tensor, h_cur, h_hat], dim=1)  # concatenate along channel axis
            y = self.act_c(self.conv_y(h_and_x_and_h_hat))
        else:
            h_next = self.act_c(o * c_next)
            y = h_next

        return h_next, c_next, y

class ConvLSTMCellReLUBN(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLUBN, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv2 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.bn1 = nn.BatchNorm2d(4 * self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(4 * self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        conv_h = self.conv1(h_cur)
        conv_x = self.conv1(input_tensor)

        combined_conv = self.bn1(conv_h) + self.bn2(conv_x)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = self.bn3(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)

        return h_next, c_next, h_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())
    #
    # def init_hidden(self, batch_size, height, width):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)),
    #             Variable(torch.zeros(batch_size, self.hidden_dim, height, width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param