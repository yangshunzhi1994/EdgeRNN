'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
       
    
def confusion_matrix(preds, y, NUM_CLASSES):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat


class Multihead_Attention(nn.Module):
    """
    multihead_attention
    <https://www.github.com/kyubyong/transformer>
    1.split+cat
    2.matmul(q,k)
    3.mask k
    4.softmax
    5.mask q
    6.matmul(attn,v)
    7.split+cat
    8.res q
    9.norm
    """

    def __init__(self,
                 hidden_dim=128,
                 C_q=None,
                 C_k=None,
                 C_v=None,
                 num_heads=2,
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        C_v = C_v if C_v else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.linear_V = nn.Linear(C_v, hidden_dim)
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, C_q]
        :param K: A 3d tensor with shape of [N, T_k, C_k]
        :param V: A 3d tensor with shape of [N, T_v, C_v]
        :return:
        """
        num_heads = self.num_heads
        N = Q.size()[0]

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))
        K_l = nn.ReLU()(self.linear_K(K))
        V_l = nn.ReLU()(self.linear_V(V))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        V_split = V_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V_split, dim=0)  # (h*N, T_v, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(K).sum(dim=-1))  # (N, T_k)
        key_masks = key_masks.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, Q.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(key_masks) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = nn.Softmax(dim=2)(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(Q).sum(dim=-1))  # (N, T_q)
        query_masks = query_masks.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, K.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # broadcasting. (h*N, T_q, T_k)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = outputs.split(N, dim=0)  # (N, T_q, C)
        outputs = torch.cat(outputs, dim=2)

        # Residual connection
        outputs = outputs + Q_l

        # Normalize
        outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs

class LearnedGroupConv1D(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, condense_factor=None, dropout_rate=0.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condense_factor = condense_factor
        self.groups = groups
        self.dropout_rate = dropout_rate

        # Check if given configs are valid
        assert self.in_channels % self.groups == 0, "group value is not divisible by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor is not divisible by input channels"
        assert self.out_channels % self.groups == 0, "group value is not divisible by output channels"

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=1, bias=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
        # register conv buffers
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))

    def forward(self, x):
        # To mask the output
        weight = self.conv.weight * self.mask
        weight_bias = self.conv.bias
        out = F.conv1d(input=x, weight=weight, bias=weight_bias, stride=self.conv.stride, 
                            padding=self.conv.padding, dilation=self.conv.dilation, groups=1)
        ## Dropping here ##
        self.check_if_drop()
        if self.dropout_rate > 0:
            out = self.dropout(out)
        return out

    """
    Paper: Sec 3.1: Condensation procedure: number of epochs for each condensing stage: M/2(C-1)
    Paper: Sec 3.1: Condensation factor: allow each group to select R/C of inputs.
    - During training a fraction of (C−1)/C connections are removed after each of the C-1 condensing stages
    - we remove columns in Fg (by zeroing them out) if their L1-norm is small compared to the L1-norm of other columns.
    """
    def check_if_drop(self):
        current_progress = LearnedGroupConv1D.global_progress
        delta = 0
        # Get current stage
        for i in range(self.condense_factor - 1):   # 3 condensation stages
            if current_progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        # Check for actual dropping
        if not self.reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
            print(delta)
        if delta > 0:
            self.drop(delta)
        return

    def drop(self, delta):
        weight = self.conv.weight * self.mask
        # Sum up all kernels
        print(weight.size())
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        print(d_out.size())
        # Shuffle weights
        weight = weight.view(d_out, self.groups, self.in_channels)
        print(weight.size())

        weight = weight.transpose(0, 1).contiguous()
        print(weight.size())

        weight = weight.view(self.out_channels, self.in_channels)
        print(weight.size())
        # Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            # Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    def reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    