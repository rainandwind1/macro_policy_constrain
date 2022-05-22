import copy
import imp
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.graph_qmix import Graph_QMixer
from modules.mixers.multi_head_qmix import Multihead_QMixer
from modules.mixers.multihead_multifeats_qmix import Multihead_multifeats_QMixer
from modules.mixers.multi_graph_qmix import Multi_Graph_QMixer
from modules.zpp_module.evaluate_model import Evaluate_module
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "graph_qmix":
                self.mixer = Graph_QMixer(args, scheme)
            elif args.mixer == "multi_head_qmix":
                self.mixer = Multihead_QMixer(args, scheme)
            elif args.mixer == "multihead_multifeats_qmix":
                self.mixer = Multihead_multifeats_QMixer(args, scheme)
            elif args.mixer == "multi_graph_qmix":
                self.mixer = Multi_Graph_QMixer(args, scheme)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        # self.input_size, self.output_size, self.n_agents 
        self.policy_size = 2    # action policy size
        self.evaluate_module = Evaluate_module(args = (scheme["obs"]["vshape"] + self.policy_size, 1, args.n_agents, args.device)).to(args.device)
        # self.params += list(self.evaluate_module.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, get_info=False, evaluate_aux = True):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_aux = mask
        # mask_aux[:,-1] = 0
        if self.policy_size != 2:
            mask_aux[:,-(self.policy_size - 2):] = 0
        avail_actions = batch["avail_actions"]
        agent_pos = batch["agent_pos"][:, :-1]
        
        # 0129 add
        if evaluate_aux:
            agent_obs = batch["obs"][:, :-1]
            
            next_mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_next_op = self.mac.forward(batch, t=t)
                next_mac_out.append(agent_next_op)
            next_mac_out = th.stack(next_mac_out, dim=1)
            actions_total = next_mac_out.argmax(dim=-1, keepdim=True)
            actions_cur = actions_total[:,:-1,:,:]
            actions_next = actions_total[:,1:,:,:]
            actions_next_3 = actions_total[:,2:,:,:]
            actions_next_4 = actions_total[:,3:,:,:]
            actions_next_5 = actions_total[:,4:,:,:]
            actions_next_6 = actions_total[:,5:,:,:]
            actions_next_7 = actions_total[:,6:,:,:]
            actions_next_8 = actions_total[:,7:,:,:]
            extra_fake_actions = th.zeros(actions_cur.shape[0], 1, actions_cur.shape[2], actions_cur.shape[3]).to(self.args.device)
            actions_next_3 = th.cat([actions_next_3, extra_fake_actions], 1)
            # print(agent_obs.shape, actions.shape, actions_next.shape)
            
            extra_fake_actions_ = th.zeros(actions_cur.shape[0], 2, actions_total.shape[2], actions_total.shape[3]).to(self.args.device)
            actions_next_4 = th.cat([actions_next_4, extra_fake_actions_], 1)
            
            extra_fake_actions_1 = th.zeros(actions_cur.shape[0], 3, actions_total.shape[2], actions_total.shape[3]).to(self.args.device)
            actions_next_5 = th.cat([actions_next_5, extra_fake_actions_1], 1)
            
            extra_fake_actions_2 = th.zeros(actions_cur.shape[0], 4, actions_total.shape[2], actions_total.shape[3]).to(self.args.device)
            actions_next_6 = th.cat([actions_next_6, extra_fake_actions_2], 1)
            
            extra_fake_actions_3 = th.zeros(actions_cur.shape[0], 5, actions_total.shape[2], actions_total.shape[3]).to(self.args.device)
            actions_next_7 = th.cat([actions_next_7, extra_fake_actions_3], 1)
            
            extra_fake_actions_4 = th.zeros(actions_cur.shape[0], 6, actions_total.shape[2], actions_total.shape[3]).to(self.args.device)
            actions_next_8 = th.cat([actions_next_8, extra_fake_actions_4], 1)
            
            raw_feats_ls = [agent_obs, actions_cur, actions_next, actions_next_3, actions_next_4, actions_next_5, actions_next_6, actions_next_7, actions_next_8]
            evaluate_feats = th.cat(raw_feats_ls[:3 + self.policy_size - 2], -1)
            # rint(evaluate_feats.shape)
            evaluate_feats = evaluate_feats.view(-1, self.args.n_agents, self.evaluate_module.input_size)
            evaluate_score = self.evaluate_module(evaluate_feats).view(self.args.batch_size, -1, 1)
            # print(evaluate_score.shape)
            evaluate_aux_loss = - evaluate_score
            

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.mixer == "graph_qmix":
                if get_info:
                    weights_info = self.mixer.get_weights_info(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1], batch["adjacency_matrix"][:, :-1])
                    return weights_info, agent_pos
    
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1], batch["adjacency_matrix"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:], batch["adjacency_matrix"][:, 1:])
            elif self.args.mixer == "multi_head_qmix":
                if get_info:
                    weights_info = self.mixer.get_weights_info(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
                    return weights_info, agent_pos
                
                # 1220 kl loss try
                # kl_loss, reconstruction_loss, chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1], train_mode = True)
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:])
             
                
                # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
                # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:])
                
            elif self.args.mixer == "multihead_multifeats_qmix":
                # feats_ls = (batch["move_feats"][:,:-1], batch["enemy_feats"][:, :-1], batch["ally_feats"][:, :-1], batch["own_feats"][:, :-1])
                # next_feats_ls = (batch["move_feats"][:, 1:], batch["enemy_feats"][:, 1:], batch["ally_feats"][:, 1:], batch["own_feats"][:, 1:])
                # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], feats_ls)
                # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], next_feats_ls)
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["obs"][:, 1:])
            elif self.args.mixer == "multi_graph_qmix":
                feats_ls = (batch["move_feats"][:,:-1], batch["enemy_feats"][:, :-1], batch["ally_feats"][:, :-1], batch["own_feats"][:, :-1])
                next_feats_ls = (batch["move_feats"][:, 1:], batch["enemy_feats"][:, 1:], batch["ally_feats"][:, 1:], batch["own_feats"][:, 1:])
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], feats_ls, batch["adjacency_matrix"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], next_feats_ls, batch["adjacency_matrix"][:, 1:])
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        # print(kl_loss.shape, reconstruction_loss.shape)
        if self.args.mixer == "multi_head_qmix":
            td_error = (chosen_action_qvals - targets.detach()) # (chosen_action_qvals - targets.detach() + kl_loss + reconstruction_loss)
        else:
            td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        mask_aux = mask_aux.expand_as(masked_td_error)
        masked_eval_error = evaluate_aux_loss * mask_aux

        
        # Normal L2 loss, take mean over actual data
        loss = ( (masked_td_error ** 2).sum() + masked_eval_error.sum()) / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
