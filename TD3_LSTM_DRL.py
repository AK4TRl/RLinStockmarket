# common library
import time
import torch
import datetime
import glob

# RL models from stable-baselines
#from stable_baselines import GAIL, SAC
#from stable_baselines import ACER
#from stable_baselines3 import PPO
#from stable_baselines3 import A2C
#from stable_baselines3 import DDPG
#from model import TD3
#from model import TD3_test
from model import TD3_LSTM

from model import utils

#from stable_baselines.ddpg.policies import DDPGPolicy
#from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
#from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
#from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# from preprocessing.preprocessors import *
from preprocessing.implementTool import *

# customized env
from env.StockTradingEnvTrain import StockEnvTrain
from env.StockTradingEnvValidation import StockEnvValidation
from env.StockTradingEnvTrade import StockEnvTrade

from finrl.trade.backtest import backtest_stats

from tensorboardX import SummaryWriter

logger = SummaryWriter(log_dir="data/log")
A_HIDDEN = 256      # Actor网络的隐层神经元数量
C_HIDDEN = 256      # Critic网络的隐层神经元数量
RANDOM_SEED = 118068
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model = TD3_LSTM.Lstm_AC(730, 81, 1)
df_account_value = []

# 模型训练入口
def train_Lstm_model(env_train, timesteps):
    start = time.time()
    env_train.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    replay_buffer = utils.ReplayBuffer(730, 81)

    # network hidden layer
    c_hx = torch.zeros(size=[1, 1, C_HIDDEN], dtype=torch.float)
    c_cx = torch.zeros(size=[1, 1, C_HIDDEN], dtype=torch.float)

    state, done = env_train.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        # if t < 128:
        #     action = env_train.action_space.sample().reshape(1, 30)
        # else:
        action = (test_model.select_action(state) + np.random.normal(0, 0.1, size=81)).clip(-1, 1)

        # Perform action
        next_state, reward, done, _ = env_train.step(action)
        done_bool = 1 if done is True else 0
        # if t % 10 == 0:
        logger.add_scalar("pre_reward", reward, global_step=t)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        logger.add_scalar("episode_reward", episode_reward, global_step=t)

        # Train agent after collecting sufficient data
        # if t >= 128:
        loss = test_model.train(replay_buffer, 256, (c_hx, c_cx), t, logger)
            # print(f"Loss L: {loss}")

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward}")
            # Reset environment
            state, done = env_train.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            #torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0)
            # a_hx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float)  # 初始化隐状态
            # a_cx = torch.zeros(size=[1, 1, A_HIDDEN], dtype=torch.float)


    end = time.time()
    # model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (TD3): ', (end - start) / 60, ' minutes')
    return test_model

# 模型预测入口
def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    state = env_trade.reset()
    action_detail = None
    init_keep = state[0][82:163]

    for i in range(len(trade_data.index.unique())):
#       goal = model.select_goal(obs_trade)
#       trade_joint_state_goal = np.concatenate([obs_trade, goal], axis=1)
        action = model.select_action(state)
        next_state, rewards, dones, info = env_trade.step(action)
        state = next_state

        # 每次输出的action可以扔到模型里面作为label参考
        # 但是_states为下一个状态，即经过action后的state
        if i == (len(trade_data.index.unique()) - 2):
            action_detail = info[0]
            # action_detail = [1 * (30 * 62)]
            # print(env_test.render())
            # print("info : ", info)
            # df_account_value.append(info)
            # print("df_account_value : ", df_account_value)
            last_state = env_trade.render()

    # info invole the operations of the stocks in 62 days,
    # it may have 62 * 30 nums data
    # we have to devide it into 30 parts to record each stock detail
    # action_detail formulation : [index , action , buy/sell]
    buy_or_sell = []
    # num_buy = [0 for i in range(0, 81)]
    # num_sell = [0 for i in range(0, 81)]
    # num1_buy = []
    # num1_sell = []
    for j in range(0, len(action_detail), 81):
        tmp = action_detail[j: j + 81]
        tmp.sort(key=lambda x: x[0])
        for item in tmp:
            buy_or_sell.append(item[1])
            # if item[0] == 0:
            #     if item[1] > 0:
            #         num1_buy.append(item[1])
            #     else:
            #         num1_sell.append(item[1])
            # if item[1] > 0:
            #     num_buy[item[0]] += item[1]
            # else:
            #     num_sell[item[0]] += item[1]

    # print("##############################")
    # print("目前持有 : ", init_keep)
    #
    # print("持有改动 : ")
    # print("股票买入 ： ", num_buy)
    # print("股票卖出 ： ", num_sell)
    #
    # last_keep = last_state[82: 163]
    # print("最后持有 : ", last_keep)
    # sum = 0
    # for index in range(0, 81):
    #     sum += last_state[index + 1] * last_state[index + 81 + 1]
    # print("总价 : ", sum)
    # print("##############################")

    trade_data_noright = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num - 1])
    trade_data_noright.insert(16, 'tic_change', buy_or_sell)
    trade_data_noright.to_csv('wait_for_process/stock_detail_{}.csv'.format(iter_num), index=False)

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, iter_num), index=False)
    return last_state


# 模型验证
def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
#        goal = model.select_goal(test_obs)
        # joint the goal and the state
#        test_joint_state_goal = np.concatenate([test_obs, goal], axis=1)
        action = model.select_action(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


# 获取验证阶段的夏普率
def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    # print(df_total_value)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


# 模型跑完后处理结果文件，将相关交易数据进行整合
def process_the_value_data(unique_trade_date):

    # ========= find the data from results adn process ==========
    csv_list = glob.glob('results/account_value_trade_ensemble_*.csv')
    # print(u'共发现%s个CSV文件' % len(csv_list))
    # print(u'正在处理............')
    csv_list = sorted(csv_list, key=os.path.getmtime)
    account_list = []
    date_list = []
    total_data = []
    first_init = True
    for i in csv_list:  # 循环读取同文件夹下的csv文件
        # fr = open(i, 'rb').read()
        with open(i, "rb") as fr:
            lines = fr.readlines()
        pf = pd.read_csv(i)
        pf = pf.drop(labels='Unnamed: 0', axis=1)
        thefirst = True
        # with open('wait_for_process/result.csv', 'ab') as f: #将结果保存为result.csv
        # for line in lines:
        #     if thefirst:
        #         thefirst = False
        #         continue
        if first_init:
            first_init = False
            data = list(pf['0'])
        else:
            data = list(pf['0'].drop(0))
        account_list.append(data)
        # f.write(line)
        # f.write(fr)
    # print(account_list)
    # print(u'处理完毕')

    rebalance_window = 63
    validation_window = 63

    # process the date
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        start = unique_trade_date[i - rebalance_window]
        end = unique_trade_date[i]
        # from i - rebalance_window to i
        for j in range(i - rebalance_window, i):
            date = list(str(unique_trade_date[j]))
            date = np.insert(date, 4, '-')
            date = np.insert(date, 7, '-')
            date = "".join(date)
            date_list.append(date)
        # print("start from ", start , " to ", end)

    # merge the date and the data
    cnt = 0
    for values in account_list:
        for value in values:
            total_data.append([date_list[cnt], value])
            cnt += 1
    # print(total_data)
    name = ['date', 'account_value']
    pd.DataFrame(data=total_data, columns=name).to_csv('wait_for_process/result.csv', index=False)


# 主函数
def run_Lstm_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    student_sharpe_list = []

    model_use = []

    data_last_parsec = None
    time_to_train_student = 127

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.date<20151000) & (df.date>=20100000)]
    # TODO:chang subset from datadate to date
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['date'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        # TODO:chang subset from datadate to date
        end_date_index = df.index[df["date"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]

        # TODO:chang subset from datadate to date
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['date'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20101200, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation envpython pd insert()方法
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window], end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation, turbulence_threshold=turbulence_threshold, iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20101200, "to ", unique_trade_date[i - rebalance_window - validation_window])


        print("====== POMDP Model Training======")

        model_td3 = train_Lstm_model(env_train, timesteps=10000)
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_student = get_validation_sharpe(i)
        print("Student Model Sharpe Ratio: ", sharpe_student)
        student_sharpe_list.append(sharpe_student)


        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        print("Used Model: ", model_td3)
        # last_state_ensemble，每次迭代都拿上一次的结果来当这一次的初始
        last_state_ensemble = DRL_prediction(df=df, model=model_td3, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
    print("Student sharpe:", student_sharpe_list)

    print("====== Data Processing ======")

    process_the_value_data(unique_trade_date)

    print("====== Finish ======")

    print("==============Get Backtest Results===========")
    df_account_value = pd.read_csv('wait_for_process/result.csv')
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("backtesting/" + "perf_stats_all_" + now + '.csv')