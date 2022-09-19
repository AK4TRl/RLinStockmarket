import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# total number of stocks in our portfolio
# TODO: change the stock dim from 30 to 81
STOCK_DIM = 81
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
# TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4


class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140
                 , initial=True, previous_state=[], model_name='', iteration=''):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(730,))  # TODO: change the obs dim from 181 to 730
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        # TODO: new factor include mfi, cpop, cpcpy
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist() + \
                     self.data.mfi.values.tolist() + \
                     self.data.cpop.values.tolist() + \
                     self.data.cpcpy.values.tolist()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.action_detail = []
        # a flag that indicate how many buy or sell
        self.buy = []
        self.sell = []
        # self.reset()
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        op_action = 0
        # if self.turbulence < self.turbulence_threshold:
        op_action = min(abs(action), self.state[index + STOCK_DIM + 1])
        if self.state[index + STOCK_DIM + 1] > 0:
            # update balance
            self.state[0] += self.state[index + 1] * op_action * (1 - TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] -= op_action
            self.cost += self.state[index + 1] * op_action * TRANSACTION_FEE_PERCENT
            self.trades += 1

            # if index == 0:
            #     print("股票1卖了: ", op_action, " 目前股票1拥有： ", self.state[index + STOCK_DIM + 1])
        else:
            op_action = 0
        # else:
        #     op_action = self.state[index + STOCK_DIM + 1]
        #     # if turbulence goes over threshold, just clear out all positions
        #     # 动荡超过阈值，空仓
        #     if self.state[index + STOCK_DIM + 1] > 0:
        #         # update balance
        #         self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * (1 - TRANSACTION_FEE_PERCENT)
        #         self.state[index + STOCK_DIM + 1] = 0
        #         self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * TRANSACTION_FEE_PERCENT
        #         self.trades += 1
        #         # if index == 0:
        #         #     print("股票1清空了， 目前股票1拥有： ", self.state[index + STOCK_DIM + 1])
        #     else:
        #         op_action = 0
        if op_action >= 0:
            op_action = -op_action

        return [index, op_action]


    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        # (index , action)
        op_action = 0
        # if self.turbulence < self.turbulence_threshold:
        # 总价格除以当前的价格等于最大可买数
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))
        op_action = min(available_amount, action)
        # update balance
        self.state[0] -= self.state[index + 1] * op_action * (1 + TRANSACTION_FEE_PERCENT)
        self.state[index + STOCK_DIM + 1] += op_action
        self.cost += self.state[index + 1] * op_action * TRANSACTION_FEE_PERCENT
        self.trades += 1

            # if index == 0:
            #     print("股票1买了: ", op_action, " 目前股票1拥有： ", self.state[index + STOCK_DIM + 1])
        # else:
        #     # if turbulence goes over threshold, just stop buying
        #     op_action = 0
        return [index, op_action]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0] + sum(
                np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])) -
                                           self.asset_memory[0]))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            # 在验证阶段计算夏普比率
            # pct_change(1) 将当前的数据和前一个数据进行计算，求增长率
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            print("Sharpe: ", sharpe)

            #
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * HMAX_NORMALIZE
            # actions = np.round(actions, 2)
            # actions = (actions.astype(int))
            # if self.turbulence >= self.turbulence_threshold:
            #     actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self.action_detail.append(self._sell_stock(index, actions[index]))

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self.action_detail.append(self._buy_stock(index, actions[index]))

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data['turbulence'].values[0]
            # print(self.turbulence)
            # load next state
            # print("stock_shares:{}".format(self.state[29:]))
            # TODO: add mfi, cpop, cpcpy to the state
            self.state = [self.state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]) + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist() + \
                         self.data.mfi.values.tolist() + \
                         self.data.cpop.values.tolist() + \
                         self.data.cpcpy.values.tolist()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, self.action_detail

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=self.iteration
            self.rewards_memory = []
            # a flag that indicate how many buy or sell
            self.buy = []
            self.sell = []
            # initiate state
            # TODO: add mfi, cpop, cpcpy to the state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                         self.data.adjcp.values.tolist() + \
                         [0] * STOCK_DIM + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist() + \
                         self.data.mfi.values.tolist() + \
                         self.data.cpop.values.tolist() + \
                         self.data.cpcpy.values.tolist()
        else:
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(STOCK_DIM + 1)]) * np.array(
                                       self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            # a flag that indicate how many buy or sell
            self.buy = []
            self.sell = []
            # initiate state
            # self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
            # [0]*STOCK_DIM + \
            # TODO: add mfi, cpop, cpcpy to the state
            self.state = [self.previous_state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)] + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist() + \
                         self.data.mfi.values.tolist() + \
                         self.data.cpop.values.tolist() + \
                         self.data.cpcpy.values.tolist()

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]