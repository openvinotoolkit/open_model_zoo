import os
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

os.environ["DGLBACKEND"] = "pytorch"


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)  # a graph convolution layer in which we specify the dimension of incoming and outgoing elements. In our case, this is a vector of words
        self.conv2 = GraphConv(h_feats, num_classes)  # the last layer, at the output we get dimension = 7 - the number of classes in the graph

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
