import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

log_step_interval = 50
A_HIDDEN = 256  # Actor网络的隐层神经元数量
C_HIDDEN = 256  # Critic网络的隐层神经元数量
time_span = 126  # 取半年的时间跨度


class MSU(nn.Module):
    def __init__(self, in_features, window_len, hidden_dim):  # 3, 20, 128
        super(MSU, self).__init__()
        self.in_features = in_features  # 3
        self.window_len = window_len  # 20
        self.hidden_dim = hidden_dim  # 128

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        self.lstm.flatten_parameters()
        # X = X.permute(1, 0, 2)
        # X = X.float().to("cuda")
        outputs, (h_n, c_n) = self.lstm(X)  # lstm version
        H_n = h_n.repeat((self.window_len, 1, 1))
        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))  # [L, B*N, 1]
        scores = scores.squeeze(2)  # [B*N, L]
        attn_weights = torch.softmax(scores, dim=1).transpose(1, 0)
        outputs = outputs.permute(1, 0, 2)  # [B*N, L, H]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        parameters = self.linear2(embed)
        # return parameters[:, 0], parameters[:, 1]   # mu, sigma
        return parameters.squeeze(-1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # 730 -> 512 -> 128 -> 81
        self.l1 = nn.Linear(state_dim, 256)
        # RNN
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)

        # MSU
        self.msu = MSU(in_features=14, window_len=time_span, hidden_dim=128)

        # 2
        self.max_action = max_action

    def forward(self, state, raw_data):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a))
        prob = self.msu(raw_data)
        mu = prob[:, 0]
        sigma = torch.log(1 + torch.exp(prob[:, 1]))
        m = Normal(mu, sigma)
        sample_rho = m.sample()
        rho = torch.clamp(sample_rho, 0.0, 1.0)
        rho_log_p = m.log_prob(sample_rho)

        return a * mu * (1 + rho)  # 2 * 1


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.lstm1 = nn.LSTM(64, hidden_size, num_layers=3, batch_first=True)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 64)
        self.lstm2 = nn.LSTM(64, hidden_size, num_layers=3, batch_first=True)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, hidden, action):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        sa = torch.cat([state, action.squeeze(0)], 1).unsqueeze(0)
        q1 = torch.tanh(self.l1(sa))
        q1 = torch.tanh(self.l2(q1))
        q1, hidden = self.lstm1(q1, hidden)
        q1 = torch.tanh(self.l3(q1))

        q2 = torch.tanh(self.l4(sa))
        q2 = torch.tanh(self.l5(q2))
        q2, hidden = self.lstm2(q2, hidden)
        q2 = torch.tanh(self.l6(q2))
        return q1, q2

    def Q1(self, state, hidden, action):
        sa = torch.cat([state, action.squeeze(0)], 1).unsqueeze(0)
        q1 = torch.tanh(self.l1(sa))
        q1 = torch.tanh(self.l2(q1))
        q1, hidden = self.lstm1(q1, hidden)
        q1 = torch.tanh(self.l3(q1))
        return q1, hidden


class Lstm_AC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=4,
    ):

        # actor net
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-7, weight_decay=1e-8)

        # critic net
        self.critic = Critic(state_dim, 256, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-7, weight_decay=1e-8)

        #
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.num_goal = state_dim

        #
        self.c_hx = torch.zeros(size=[3, 1, C_HIDDEN], dtype=torch.float).to(device)
        self.c_cx = torch.zeros(size=[3, 1, C_HIDDEN], dtype=torch.float).to(device)

        self.total_it = 0

    def select_action(self, state, processed_data):
        state = torch.FloatTensor(state.reshape(1, 271)).to(device)
        action = self.actor(state, processed_data)
        return action.reshape(1, 30).cpu().data.numpy()

    def train(self, replay_buffer, batch_size, c_hidden, global_iter_num, logger, processed_data):
        self.total_it += 1
        replay_len = replay_buffer.get_size()

        # 1 -> 1000+ / batch_size + 1 -> 2 + 1 -> 3 + 1
        k = 1 + replay_len / batch_size

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(int(k))

        # ========================================== state - action ================================================#

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # 添加平滑噪声
            # float clamp(float minnumber, float maxnumber, float parameter)
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # 根据next_state由actor目标网络算出下一个动作
            action = self.actor_target(next_state, processed_data)
            next_action = (
                    action + noise
            ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            # 目标真实Q值，现实Q值
            target_Q1, target_Q2 = self.critic_target(next_state, (self.c_hx, self.c_cx), next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # 取两者最小一个
            target_Q = reward + (1 - not_done) * self.discount * target_Q  # 得到目标Q值

        # Get current Q estimates
        # 预测的Q值; 估计Q值
        current_Q1, current_Q2 = self.critic(state, c_hidden, action)

        # Compute critic loss
        # 比较目标Q值与真实Q值
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # tensorboard
        logger.add_scalar("critic loss", critic_loss.item(), global_step=global_iter_num)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            # 输入状态s使得返回得action值能让q值最大
            actor_Q, (self.c_hx, self.c_cx) = self.critic.Q1(state, c_hidden, self.actor(state, processed_data))
            actor_loss = -actor_Q.mean()  # Value
            logger.add_scalar("actor loss", actor_loss.item(), global_step=global_iter_num)
            logger.add_scalar("actor Q value", actor_Q.mean(), global_step=global_iter_num)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            # 软更新
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss