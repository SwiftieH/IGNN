import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse


class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #five layers
        self.ig1 = ImplicitGraph(nfeat, 4 * nhid, num_node, kappa)
        self.ig2 = ImplicitGraph(4*nhid, 2* nhid, num_node, kappa)
        self.ig3 = ImplicitGraph(2*nhid, 2*nhid, num_node, kappa)
        self.ig4 = ImplicitGraph(2*nhid, nhid, num_node, kappa)
        self.ig5 = ImplicitGraph(nhid, nclass, num_node, kappa)
        self.dropout = dropout
        #self.X_0 = Parameter(torch.zeros(nhid, num_node))
        self.X_0 = None
        #self.V = nn.Linear(nhid, nclass, bias=False)
        self.V = nn.Linear(nhid, nclass)
        self.V_0 = nn.Linear(nfeat, 4*nhid)
        self.V_1 = nn.Linear(4*nhid, 2*nhid)
        self.V_2 = nn.Linear(2*nhid, 2*nhid)
        self.V_3 = nn.Linear(2*nhid, nhid)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features

        #five layers
        x = F.elu(self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_0(x.T)).T
        x = F.elu(self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_1(x.T)).T
        x = F.elu(self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_2(x.T)).T
        x = F.elu(self.ig4(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_3(x.T)).T
        x = self.ig5(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V(x.T)
        #return F.log_softmax(x, dim=1)
        return x


