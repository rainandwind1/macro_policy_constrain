from audioop import mul
from cgi import test
import imp
from turtle import forward
from scipy import rand


import torch
import random
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import collections
from collections import deque



class Evaluate_module(nn.Module):
    def __init__(self, args):
        super(Evaluate_module, self).__init__()
        self.input_size, self.output_size, self.n_agents, self.device = args
        self.lr = 1e-3
        self.max_len = 50000
        self.embedding_size = 1
        self.ma_heads_num = 1
        self.ma_weight_net = Multihead_Module_no_hyper(args = (self.input_size, self.ma_heads_num, self.n_agents, self.embedding_size))
        self.evaluate_net = nn.Linear(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.parameters(), self.lr)
        self.buffer = deque(maxlen = self.max_len) 
        
    def forward(self, inputs):
        ind_eval_score = self.evaluate_net(inputs)
        ma_weights = self.ma_weight_net(inputs).unsqueeze(-1)
        evaluate_score = torch.matmul(ind_eval_score.permute(0, 2, 1), ma_weights).squeeze(-1)
        return evaluate_score
    
    def save_trans(self, trans):
        self.buffer.append(trans)
        
    
    def sample_batch(self, batch_size = 128):
        trans_batch = random.sample(self.buffer, batch_size)
        inputs_feats_ls, policy_reward_ls = [], []
        for trans in trans_batch:
            inputs_feats_ls.append(trans[0])
            policy_reward_ls.append(trans[1])
            
        return (
            torch.FloatTensor(inputs_feats_ls).to(self.device),
            torch.FloatTensor(policy_reward_ls).to(self.device).unsqueeze(-1)
        )
    
    
    def supervise_train(self, batch_size = 32):
        input_feats, policy_reward = self.sample_batch(batch_size)
        eval_score = self(input_feats)
        loss = ((eval_score - policy_reward)**2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        
        
        
        


class Multihead_Module_no_hyper(nn.Module):
    def __init__(self, args):
        super(Multihead_Module_no_hyper, self).__init__()
        self.input_size, self.num_heads, self.Seq_len, self.embedding_size = args

        self.q_net = nn.Linear(self.input_size, self.embedding_size)
        self.k_net = nn.Linear(self.input_size, self.embedding_size)
        self.v_net = nn.Linear(self.input_size, self.embedding_size)

        self.multihead_net = nn.MultiheadAttention(self.embedding_size, self.num_heads)


    def forward(self, input):
        inputs = input
        q_vec = self.q_net(inputs).permute(1, 0, 2)
        k_vec = self.k_net(inputs).permute(1, 0, 2)
        v_vec = self.v_net(inputs).permute(1, 0, 2)
        multihead_op, multihead_weights = self.multihead_net(q_vec, k_vec, v_vec)
        multihead_op = multihead_op.permute(1, 0, 2).squeeze(-1)
        return F.softmax(multihead_op, -1)
    
    

if __name__ == "__main__":
    ma_module = Multihead_Module_no_hyper(args = (24, 1, 4, 1))
    test_inputs = torch.randn(32, 4, 24)
    testop = ma_module(test_inputs)
    
    eval_model = Evaluate_module(args = (24, 1, 4, 1e-3))
    eval_score = eval_model(test_inputs)
    print(testop.shape, eval_score.shape, eval_score)
