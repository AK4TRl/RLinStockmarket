import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
# from stockmarket import d_ratio

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
REWARD_SCALING = 1e-4


class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(730,))  # TODO: change the obs dim from 181 to 730
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
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
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index + STOCK_DIM + 1] > 0:
            # update balance
            # after selling the shares, how much we total have now ?
            # total += (the profit after transaction fee paid, profit - transaction fee)
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)

            # update prices amount
            self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
            # transaction fee (cost)
            self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        # Is that all in ?
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))

        # update balance
        # select the minimal amount to buy
        # total -= price * amount * (1 + 手续费)
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

        # update the index th shares amount
        # select the minimal amount to add
        self.state[index + STOCK_DIM + 1] += min(available_amount, action)

        # transaction fee (cost)
        self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        # print(self.day)
        # 如果设置的day大于数据的日期，就是到了结束点
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            # 结束后，输出余额值变化情况
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            # 最后得到的总余额，包括在手的股份价格
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            # print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            # print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            # print("total_cost: ", self.cost)
            # print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            anusual_rewards = np.array(self.rewards_memory)
            # dratios = d_ratio(anusual_rewards, )
            # df_rewards.to_csv('results/account_rewards_train.csv')

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
             #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))
            # 规范化action，大概就是将action表示的买/卖数量归一化
            actions = actions * HMAX_NORMALIZE
            # actions = (actions.astype(int))

            # 当前step开始时的asset
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

            # 按参数排序，也就是返回对应序列元素的数组下表
            # [1， 4， 3， -1， 6， 9]
            # [3（-1）， 0（1）， 2（3）， 1（4）， 4（6）， 5（9）]
            argsort_actions = np.argsort(actions)

            # 查找小于0的actions
            # ①：共有多少个小于0的actions，代表有多少个要卖出去，并获取其下标及数量
            # ②：共有多少个大于0的actions，代表有多少个要买进来，并获取其下标及数量
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            # 对应买入卖出操作计算
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            #
            self.day += 1
            self.data = self.df.loc[self.day, :]
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

            # 利用profit作为agent的reward反馈
            # 前一个月波动率、前一个月涨跌幅等都可以作为state输入？但是这些状态都不会有很大的变化
            # 如何衡量当日的step的质量？
            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
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
        # iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=123456):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]