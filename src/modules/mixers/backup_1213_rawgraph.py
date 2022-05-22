from numpy.core.arrayprint import printoptions
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class Graph_QMixer(nn.Module):
#     def __init__(self, args, scheme):
#         super(Graph_QMixer, self).__init__()

#         self.args = args
#         self.n_agents = args.n_agents
#         self.state_dim = int(np.prod(args.state_shape))
#         self.obs_size = scheme["obs"]["vshape"]

#         self.embed_dim = args.mixing_embed_dim
        
#         # graph assist   self.input_size, self.output_size, self.dropout, self.alpha, self.device = args
#         # self.gat_module = GAT_Module(args = (scheme["state"]["vshape"], scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2))
#         self.gat_module = GAT(scheme["obs"]["vshape"], 64, args.n_agents, 0.2, 0.2, 4)
#         # scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2

#         if getattr(args, "hypernet_layers", 1) == 1:
#             self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
#             self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
#         elif getattr(args, "hypernet_layers", 1) == 2:
#             hypernet_embed = self.args.hypernet_embed
#             self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
#                                            nn.ReLU(),
#                                            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
#             self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
#                                            nn.ReLU(),
#                                            nn.Linear(hypernet_embed, self.embed_dim))
#         elif getattr(args, "hypernet_layers", 1) > 2:
#             raise Exception("Sorry >2 hypernet layers is not implemented!")
#         else:
#             raise Exception("Error setting number of hypernet layers.")

#         # State dependent bias for hidden layer
#         self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

#         # V(s) instead of a bias for the last layers
#         self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
#                                nn.ReLU(),
#                                nn.Linear(self.embed_dim, 1))

#     def get_graph_info(self, states, obs_ls, adj_ls):
#         # print(obs_ls.shape, adj_ls.shape)
#         obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
#         adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
#         gat_op = self.gat_module(obs_ls, adj_ls)
#         gat_weights = gat_op
#         return gat_weights

#     def get_weights_info(self, agent_qs, states, obs_ls, adj_ls):
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.view(-1, 1, self.n_agents)
#         graph_weights = self.get_graph_info(states, obs_ls, adj_ls)
#         return graph_weights
        

#     def forward(self, agent_qs, states, obs_ls, adj_ls):
#         bs = agent_qs.size(0)
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.view(-1, 1, self.n_agents)
#         obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
#         adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
#         graph_weights = self.get_graph_info(states, obs_ls, adj_ls)
#         agent_qs = th.bmm(graph_weights, agent_qs.permute(0, 2, 1)).permute(0, 2, 1)


#         # First layer
#         w1 = th.abs(self.hyper_w_1(states))
#         b1 = self.hyper_b_1(states)
#         w1 = w1.view(-1, self.n_agents, self.embed_dim)
#         b1 = b1.view(-1, 1, self.embed_dim)
#         hidden = F.elu(th.bmm(agent_qs, w1) + b1)
#         # Second layer
#         w_final = th.abs(self.hyper_w_final(states))
#         w_final = w_final.view(-1, self.embed_dim, 1)
#         # State-dependent bias
#         v = self.V(states).view(-1, 1, 1)
#         # Compute final output
#         y = th.bmm(hidden, w_final) + v
#         # Reshape and return
#         q_tot = y.view(bs, -1, 1)
#         return q_tot

class Graph_QMixer(nn.Module):
    def __init__(self, args, scheme):
        super(Graph_QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_size = scheme["obs"]["vshape"]

        self.embed_dim = args.mixing_embed_dim
        
        # graph assist   self.input_size, self.output_size, self.dropout, self.alpha, self.device = args
        # self.gat_module = GAT_Module(args = (scheme["state"]["vshape"], scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2))
        # self.gat_module = GAT(scheme["obs"]["vshape"], self.n_agents * self.n_agents, args.n_agents, 0.2, 0.2, 4)
        # scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2

        # if getattr(args, "hypernet_layers", 1) == 1:
        #     self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        #     self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        # elif getattr(args, "hypernet_layers", 1) == 2:
        #     hypernet_embed = self.args.hypernet_embed
        #     self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        #     self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hypernet_embed, self.embed_dim))
        # elif getattr(args, "hypernet_layers", 1) > 2:
        #     raise Exception("Sorry >2 hypernet layers is not implemented!")
        # else:
        #     raise Exception("Error setting number of hypernet layers.")
        
        self.weight_net = nn.Linear(self.state_dim, self.n_agents)
        self.hyper_w_1 = GAT(scheme["obs"]["vshape"], 32, args.n_agents * self.embed_dim, 0.1, 0.2, 4)
        self.hyper_w_final = GAT(scheme["obs"]["vshape"], 32, self.embed_dim, 0.1, 0.2, 4)
        


        # State dependent bias for hidden layer
        self.hyper_b_ls = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.n_agents)])

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def get_graph_info(self, states, obs_ls, adj_ls):
        # print(obs_ls.shape, adj_ls.shape)
        obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
        adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
        self.hyper_w1 = th.abs(self.hyper_w_1(obs_ls, adj_ls))
        self.hyper_wfinal = th.abs(self.hyper_w_final(obs_ls, adj_ls))

    def get_weights_info(self, agent_qs, states, obs_ls, adj_ls):
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        graph_weights = self.get_graph_info(states, obs_ls, adj_ls)
        return graph_weights
        

    def forward(self, agent_qs, states, obs_ls, adj_ls):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
        adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
        self.get_graph_info(states, obs_ls, adj_ls)

        dis_weights = th.abs(self.weight_net(states)).view(-1, self.n_agents, 1)
        # First layer
        hidden_ls = []
        for i in range(self.n_agents):
            w = self.hyper_w1[:,i,:].view(-1, self.n_agents, self.embed_dim)
            b = self.hyper_b_ls[i](states).view(-1, 1, self.embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w) + b)
            hidden_ls.append(hidden)
            
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        q_tot_ls = []    
        # Second layer
        for j in range(self.n_agents):
            w_final = self.hyper_wfinal[:,j,:].view(-1, self.embed_dim, 1)
            # Compute final output
            y = th.bmm(hidden_ls[j], w_final)
            # Reshape and return
            # q_tot_ = y.view(-1, 1, 1)
            q_tot_ls.append(y)
        # print(dis_weights.shape,th.cat(q_tot_ls, -1).shape)
        q_tot = (th.bmm(th.cat(q_tot_ls, -1), dis_weights) + v).view(bs, -1, 1)
        return q_tot


class GraphAttentionLayer_hyper(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hyper_input_size, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.hyper_input_size = hyper_input_size
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(th.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(th.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.W_parameter = nn.Linear(self.hyper_input_size, self.in_features * self.out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, state, h, adj):
        self.W = th.abs(self.W_parameter(state).reshape(-1, self.in_features, self.out_features))
        Wh = th.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = th.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = th.matmul(Wh, self.a[self.out_features:, :])
        # print(Wh1.shape, Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(th.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(th.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = th.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = th.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = th.matmul(Wh, self.a[self.out_features:, :])
        # print(Wh1.shape, Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = th.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.softmax(F.log_softmax(x, dim=1), -1)
        return F.log_softmax(x, dim=1)

class GAT_hyper(nn.Module):
    def __init__(self, hyper_input_size, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_hyper(hyper_input_size, nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_hyper(hyper_input_size, nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, state, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = th.cat([att(state, x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(state, x, adj))
        return F.softmax(F.log_softmax(x, dim=1), -1)




















if __name__ == "__main__":
    a = th.randn(2752, 8, 85)
    hyper_inputs = th.randn(32, 60, 120)
    adj = th.ones(a.shape[0], a.shape[1], a.shape[1])
    model = GAT(85, 32, 8, 0.6, 0.2, 4)
    c = model(a, adj)
    print(c.shape)