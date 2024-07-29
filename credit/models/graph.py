from torch import nn, reshape
from torch_geometric.nn import TransformerConv


class GraphResidualTransformer(nn.Module):
    def __init__(self, n_variables, hidden_size=128, heads=1, dropout=0):
        super().__init__()
        self.n_variables = n_variables
        self.hidden_size = hidden_size
        self.heads = heads
        self.dropout = dropout
        self.transformer = TransformerConv(-1, self.hidden_size,
                                           heads=self.heads, dropout=self.dropout)
        self.physics_linear = nn.LazyLinear(self.hidden_size)
        self.merge_linear = nn.Linear(self.hidden_size, self.n_variables)
        return

    def forward(self, x, edge_index):
        state = x
        x_table = reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3]))
        x_t = self.transformer(x_table, edge_index)
        x_t = nn.ReLU(x_t)
        x_t = self.physics_linear(x_t)
        x_t = nn.ReLU(x_t)
        x_t = self.merge_linear(x_t)
        tendency = reshape(x_t, (-1, x.shape[1], x.shape[2], x.shape[3]))
        tendency += state
        return tendency
