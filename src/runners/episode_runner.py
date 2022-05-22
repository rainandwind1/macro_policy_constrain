import collections
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np



class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, evaluate_model = None, test_cnt=0):
        self.reset()

        # for debug
        # state_debug = self.env.get_state()
        # obs_debug = self.env.get_obs()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        
        # 2022/03/08 add for macro constrain
        action_history_ls = [[] for _ in range(self.args.n_agents)]
        
        # T
        T = 4
        eval_obs_ls = collections.deque(maxlen=T)
        actions_eval_ls = collections.deque(maxlen=T)
        reward_eval_ls = collections.deque(maxlen=T)
        
        while not terminated:
            
            # obs divided into 4 items: move feats、enemy feats、ally feats、own feats
            agent_obs_feats = self.env.get_obs_feats()
            eval_obs = self.env.get_obs()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "adjacency_matrix":[self.env.get_adjacency_matrix()],
                "move_feats":[agent_obs_feats[0]],
                "enemy_feats":[agent_obs_feats[1]],
                "ally_feats":[agent_obs_feats[2]],
                "own_feats":[agent_obs_feats[3]],
                "agent_pos":[self.env.get_pos_info()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            
            # 2022/03/08
            if test_mode:
                for idx, action_i in enumerate(actions[0]):
                    action_history_ls[idx].append(int(action_i))
            
            # save evaluate model transition 0129 add
            if evaluate_model is not None:
                if len(actions_eval_ls) >= T:
                    eval_feats = np.concatenate([eval_obs_ls[0]] + list(actions_eval_ls), -1)
                    evaluate_model.save_trans((eval_feats, sum(reward_eval_ls)))
                
                if len(evaluate_model.buffer) > 128:
                    print("supervise_train! ")
                    evaluate_model.supervise_train()
                
                eval_obs_ls.append(eval_obs)
                actions_eval_ls.append(actions.unsqueeze(-1)[0].cpu().numpy())
                reward_eval_ls.append(reward)
            
        # 2022/03/08
        if test_mode:
            np.save("./qmixgood_{}_agents_actions_epi_{}.npy".format("MMM2", test_cnt), np.array(action_history_ls))
        
        agent_obs_feats = self.env.get_obs_feats()
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "adjacency_matrix":[self.env.get_adjacency_matrix()],
            "move_feats":[agent_obs_feats[0]],
            "enemy_feats":[agent_obs_feats[1]],
            "ally_feats":[agent_obs_feats[2]],
            "own_feats":[agent_obs_feats[3]]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
