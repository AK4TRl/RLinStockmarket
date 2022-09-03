import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

#Lstm
class LstmModel(nn.Module):
	def __init__(self, obs_dim, act_dim):
		super(LstmModel, self).__init__()
		self.hidden_size = 64
		self.first = False
		self.act_dim = act_dim
		# 3层全连接网络
		self.fc1 = nn.Sequential(
			nn.Linear(obs_dim,128),
			nn.ReLU())

		self.fc2 = nn.Sequential(
			nn.Linear(self.hidden_size,128),
			nn.ReLU())
		self.fc3 = nn.Linear(128, act_dim)
		self.lstm = nn.LSTM(128, self.hidden_size, 1)      #[input_size,hidden_size,num_layers]

	def init_lstm_state(self,batch_size):
		self.h = np.zeros(shape=[1, batch_size, self.hidden_size], dtype='float32')
		self.c = np.zeros(shape=[1, batch_size, self.hidden_size], dtype='float32')
		self.first = True

	def forward(self, obs):
		# 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
		obs = self.fc1(obs)
		#每次训练开始前重置
		if self.first:
			x, (h, c) = self.lstm(obs, (self.h, self.c))  #obs:[batch_size,num_steps,input_size]
			self.first = False
		else:
			x, (h, c) = self.lstm(obs)  #obs:[batch_size,num_steps,input_size]
		x = x.reshape(shape=[-1, self.hidden_size])
		h2 = self.fc2(x)
		Q = self.fc3(h2)
		return Q

def distillation(student_scores, labels, teacher_scores, T, alpha):

	#####################################################################
	# loss_f = nn.KLDivLoss()
	# loss_ST1 = loss_f(student_scores, teacher_scores)
	# # loss_ST2 = loss_f(y, teacher_scores2)
	# L = nn.CrossEntropyLoss()
	# loss_SL1 = L(student_scores, labels)
	# # loss_SL2 = L(y, labels)
	# loss = float((1.0 - alpha) * (loss_SL1)) + alpha * T * T * (loss_ST1)
	######################################################################

	######################################################################
	L_soft = F.cross_entropy(F.softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1))
	L_hard = F.cross_entropy(F.softmax(student_scores, dim=1), labels)
	loss = (1 - alpha) * L_hard + alpha * L_soft
	######################################################################

	# KD_loss = nn.KLDivLoss()(F.log_softmax(student_scores / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * (alpha * T * T) + F.cross_entropy(student_scores, labels) * (1. - alpha)

	return loss

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):
		# actor net
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2 ** -14)

		# critic net
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		# self.critic_teacher = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2 ** -14)

		#
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def predict(self, state):
		# obs_trade可以扔到这里把action测出来
		# 对于state和action，可以打包为一个数组
		# 也就输current state 和 current action
		state = torch.FloatTensor(state.reshape(-1, )).to(device)
		return self.actor(state).reshape(1, 30).cpu().data.numpy(), None


	def train(self, teacher_buffer, student_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		# 从经验池中采样
		teacher_state, teacher_action, teacher_next_state, teacher_reward, teacher_done = teacher_buffer.sample(batch_size)
		student_state, student_action, student_next_state, student_reward, student_done = student_buffer.sample(batch_size)

		# reward = teacher_reward
		# if teacher_reward < student_reward:
		# 	reward = student_reward


		with torch.no_grad():
			# Select action according to policy and add clipped noise
			# 添加平滑噪声，Teacher_action的存在也为学生的操作带来了一点噪声
			# 对于Teacher_action怎么作为学生的指导引入呢？
			# 由于学生的原本噪声为0.2，设置教师的action影响也能作为噪声存在，可以取0.8
			# float clamp(float minnumber, float maxnumber, float parameter)
			noise = (
				torch.randn_like(student_action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			# 根据next_state由actor目标网络算出下一个动作
			next_action = (
					(self.actor_target(student_next_state)) * 0.8 + teacher_action * 0.2 + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			# 目标真实Q值，现实Q值
			# reward值取学生和教师间最大的一个，
			target_Q1, target_Q2  = self.critic_target(student_next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2) 					# 取两者最小一个
			target_Q = teacher_reward + self.discount * target_Q		# 得到目标Q值


		# 学生和老师预测的Q值; 估计Q值
		student_current_Q1, student_current_Q2 = self.critic(student_state, student_action)
		# teacher_current_Q1, teacher_current_Q2 = self.critic_teacher(teacher_state, teacher_action)

		# Get current Q estimates
		# 预测的Q值; 估计Q值
		# current_Q1, current_Q2 = self.critic(state, student_action)

		# Compute critic loss
		# 比较目标Q值与真实Q值
		# 计算其critic loss
		critic_loss = F.mse_loss(student_current_Q1, target_Q) + F.mse_loss(student_current_Q2, target_Q)
		# critic_loss = F.mse_loss(student_current_Q, target_Q)

		#########################################################################################################################################
		# 对TD3进行改造
		# 已知传输进来的参数为Teacher[], Student[]
		# 目前有Student_state和Teacher_state，这个可以算是Teacher和Student的next_state
		# 由于Student和Teacher的next_state可以推出next_action，并由于已知当前Teacher和Student的action
		# 当前action的用处？
		# Target_Q = self.critic_target(tearcher_state, )

		# def distillation(y, labels, teacher_scores, temp, alpha):
		# 	return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

		# loss = distillation(student_current_Q1, student_current_Q2, target_Q, teacher_current_Q1, teacher_current_Q2, temp=3.0, alpha=0.7) + critic_loss
		# loss = distillation(student_current_Q, target_Q, teacher_current_Q, T=3.0, alpha=0.7) + critic_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		# loss.backward()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		# 延迟策略更新
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			# actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			actor_loss = -self.critic.Q1(teacher_state, teacher_action).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			# 软更新
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				# teacher_param.data.copy_(self.tau * param.data + (1 - self.tau) * teacher_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		# return loss, student_current_Q.item(), teacher_current_Q.item(), target_Q.item()

	# 保存优化参数
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	# 加载参数
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		