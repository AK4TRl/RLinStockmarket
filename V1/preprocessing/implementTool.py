import os
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from datetime import datetime, date


path = "../data/100 stocks"

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(['date', 'tic'], ignore_index=True)
    # data  = data[final_columns]
    data.index = data.date.factorize()[0]
    return data


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(path + '/' + file_name)
    return _data


def get_all_data_to_one():
    files = os.listdir(path)
    total_df = []
    conut = 0
    for file in files:
        new_date = []
        df = load_dataset(file_name=file)
        if len(df) < 2518: continue
        conut += 1
        # print(conut, " : ", file.split('.')[0])
        for date in df.Date:
            if date[4] == '/':
                now_date = datetime.strptime(date, '%Y/%m/%d')
            elif date[4] == '-':
                now_date = datetime.strptime(date, '%Y-%m-%d')
            now_date = now_date.strftime("%Y%m%d")
            new_date.append(now_date)
        df["Date"] = new_date
        df.insert(loc=1, column='tic', value=file.split('.')[0])
        total_df = total_df + df.values.tolist()
    print("The remain stocks : ", conut)
    # print(total_df)
    data = pd.DataFrame(total_df, columns=['date', 'tic', 'open', 'high', 'low', 'close', 'adjcp', 'volume'])
    data = data.sort_values(['tic', 'date'], ignore_index=True)
    # print(data)
    return data


def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())
    # close price 为 adjusted close price
    # stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    #
    cci = pd.DataFrame()
    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    adx = pd.DataFrame()
    mfi = pd.DataFrame()
    CPOP = pd.DataFrame()
    CPCPY = pd.DataFrame()

    # temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_adx = stock[stock.tic == unique_ticker[i]]['adx']
        temp_adx = pd.DataFrame(temp_adx)
        adx = adx.append(temp_adx, ignore_index=True)
        ## mfi
        temp_mfi = stock[stock.tic == unique_ticker[i]]['mfi']
        temp_mfi = pd.DataFrame(temp_mfi)
        mfi = mfi.append(temp_mfi, ignore_index=True)
        ## CPOP
        temp_cpop = stock[stock.tic == unique_ticker[i]]['close'] - stock[stock.tic == unique_ticker[i]]['open']
        temp_cpop = pd.DataFrame(temp_cpop)
        CPOP = CPOP.append(temp_cpop, ignore_index=True)
        ## CPCPY
        temp_cpcpy = stock[stock.tic == unique_ticker[i]]['close'][1:].values - stock[stock.tic == unique_ticker[i]]['close'][:-1].values
        temp_cpcpy = np.insert(temp_cpcpy, 0, 0)
        temp_cpcpy = pd.DataFrame(temp_cpcpy)
        CPCPY = CPCPY.append(temp_cpcpy, ignore_index=True)

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = adx
    df['mfi'] = mfi
    df['cpop'] = CPOP
    df['cpcpy'] = CPCPY

    return df


def preprocess_data():

    # 将所有数据合并成一组数据
    data = get_all_data_to_one()
    # df_preprocess = get_feature(df)
    # 在合并后的数据内添加技术因子
    df_final = add_technical_indicator(data)

    return df_final


def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df


def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot = df.pivot(index='date', columns='tic', values='adjcp')
    unique_date = df.date.unique()
    # start after a year
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0
    for i in range(start, len(unique_date)):
        '''

        start = 252, i = 252
        get the current price of today and process the data
        df_price_pivot: get Date Tic and Adjcp, just 3 parts data,
                        we set Date as index
                                tic as columns
                                adjcp as values
                        such as:
                        index       AAPL    AXP     BA
                        20100104    30      28      26
                        20100105    29      27      24
        unique_date: defines the unique date for all the stock env
        current_price: according to the date to get the every tic price(just one day price)
                    while i == 252, and unique_date[i] == 20100104
                     current_price = df_price_pivot[20100104]
        hist_price: get the values according to the date from 20090101-20091231

        '''
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]

        '''
         get the data and calculate the turbulence
        '''

        cov_temp = hist_price.cov()
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        if turbulence_temp > 1000:
            turbulence_temp %= 1000
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({'date': df_price_pivot.index,
                                     'turbulence': turbulence_index})
    return turbulence_index


# if __name__ == '__main__':
#     preprocessed_path = "../data/NASDAQ_data.csv"
#     data = preprocess_data()
#     data.fillna(method='bfill', inplace=True)
#     data = add_turbulence(data)
#     data.to_csv(preprocessed_path)

