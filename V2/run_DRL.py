# 2 preprocessors
# 第一个数据处理文件作为标准，面向琼斯指数
# 第二个数据处理文件在第一个的基础上进行了小扩建，面向纳斯达克
#from preprocessing.preprocessors import *
from preprocessing.implementTool import *
# config
# 2 models
# 第一个是多模型训练策略（标程）
# 第二个是单模型训练策略
# from model.models import *
from TD3_LSTM_DRL import *
import os


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "./data/dow30_done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.date > 20151001) & (data.date <= 20201231)].date.unique()
    # print("unique_trade_date", unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    run_Lstm_strategy(df=data, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, validation_window=validation_window)


if __name__ == "__main__":
    run_model()
