import numpy as np
import torch
from torch import nn


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def get_size(self):
		return self.size


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def v_wrap(np_array, dtype=np.float32):
	if np_array.dtype != dtype:
		np_array = np_array.astype(dtype)
	return torch.from_numpy(np_array)


def set_init(layers):
	for layer in layers:
		nn.init.normal_(layer.weight, mean=0., std=0.1)
		nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
	if done:
		v_s_ = 0.               # terminal
	else:
		v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

	buffer_v_target = []
	for r in br[::-1]:    # reverse buffer r
		v_s_ = r + gamma * v_s_
		buffer_v_target.append(v_s_)
	buffer_v_target.reverse()
	loss = lnet.loss_func(
		v_wrap(np.vstack(bs)),
		v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
		v_wrap(np.array(buffer_v_target)[:, None]))

	# calculate local gradients and push local parameters to global
	opt.zero_grad()
	loss.backward()
	for lp, gp in zip(lnet.parameters(), gnet.parameters()):
		gp._grad = lp.grad
	opt.step()

	# pull global parameters
	lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
	with global_ep.get_lock():
		global_ep.value += 1
	with global_ep_r.get_lock():
		if global_ep_r.value == 0.:
			global_ep_r.value = ep_r
		else:
			global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
	res_queue.put(global_ep_r.value)
	print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value,)


class SharedAdam(torch.optim.Adam):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
		super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
		# State initialization
		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['step'] = 0
				state['exp_avg'] = torch.zeros_like(p.data)
				state['exp_avg_sq'] = torch.zeros_like(p.data)

				# share in memory
				state['exp_avg'].share_memory_()
				state['exp_avg_sq'].share_memory_()