import copy
import numpy as np
# from model import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

log_step_interval = 50
A_HIDDEN = 256      # Actor网络的隐层神经元数量
C_HIDDEN = 256      # Critic网络的隐层神经元数量


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        # RNN
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.l2 = nn.Linear(hidden_size, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, state, hidden):
        # state = torch.FloatTensor(state.reshape(1, 1, 181))
        a = F.relu(self.l1(state))
        self.lstm.flatten_parameters()
        a, hidden = self.lstm(a.unsqueeze(0), hidden)
        a = F.relu(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a)), hidden


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        # self.l2 = nn.Linear(64, 64)
        self.lstm1 = nn.LSTM(64, hidden_size, batch_first=True)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 64)
        # self.l5 = nn.Linear(128, 64)
        self.lstm2 = nn.LSTM(64, hidden_size, batch_first=True)
        self.l6 = nn.Linear(hidden_size, 1)

        # 
#        self.lstm1 = nn.LSTM(64, hidden_size, batch_first=True)

    def forward(self, state, hidden, action):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
    
        sa = torch.cat([state, action.squeeze(0)], 1).unsqueeze(0)

        q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        q1, hidden = self.lstm1(q1, hidden)
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        # q2 = F.relu(self.l5(q2))
        q2, hidden = self.lstm2(q2, hidden)
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, hidden, action):
        sa = torch.cat([state, action.squeeze(0)], 1).unsqueeze(0)

        q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        q1, hidden = self.lstm1(q1, hidden)
        q1 = self.l3(q1)
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
        self.actor = Actor(state_dim, 256, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        # critic net
        self.critic = Critic(state_dim, 256, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        #
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.num_goal = state_dim

        #
        self.a_hx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float).to(device)
        self.a_cx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float).to(device)
        self.c_hx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float).to(device)
        self.c_cx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float).to(device)

        self.total_it = 0

    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1, 730)).to(device)
        action, hidden = self.actor(state, (self.a_hx, self.a_cx))
        self.a_hx, self.a_cx = hidden
        return action.reshape(1, 81).cpu().data.numpy()

    def train(self, replay_buffer, batch_size, c_hidden, global_iter_num, logger):
        self.total_it += 1

        replay_len = replay_buffer.get_size()

        k = 1 + replay_len / batch_size

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(int(k))

        #========================================== state - action ================================================#
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # 添加平滑噪声
            # float clamp(float minnumber, float maxnumber, float parameter)
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # 根据next_state由actor目标网络算出下一个动作
            action, a_hidden = self.actor_target(next_state, (self.a_hx, self.a_cx))
            next_action = (
                    action + noise
            ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            # 目标真实Q值，现实Q值
            target_Q1, target_Q2 = self.critic_target(next_state, c_hidden, next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # 取两者最小一个
            target_Q = reward + (1 - not_done) * self.discount * target_Q  # 得到目标Q值

        # Get current Q estimates
        # 预测的Q值; 估计Q值
        current_Q1, current_Q2 = self.critic(state, c_hidden, action)

        # Compute critic loss
        # 比较目标Q值与真实Q值
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # if global_iter_num % log_step_interval == 0:
            # print("iter_num", global_iter_num)
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
            action_tmp, (self.a_hx, self.a_cx) = self.actor(state, a_hidden)
            actor_Q, (self.c_hx, self.c_cx) = self.critic.Q1(state, c_hidden, action_tmp)
            actor_loss = -actor_Q.mean()    # Value
            # if global_iter_num % log_step_interval == 0:
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