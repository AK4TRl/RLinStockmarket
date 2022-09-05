### SB3 on Windows 10

To install stable-baselines3 on Windows, please look at the [documentation](https://stable-baselines3.readthedocs.io/).

If you have questions regarding Stable-baselines3 package, please refer to [Stable-baselines3 installation guide](https://github.com/DLR-RM/stable-baselines3). 
Install the Stable Baselines3 package using pip:
```
pip install stable-baselines3
```

### Questions

#### About Tensorflow 2.0 for TensorBoard

If you have questions regarding TensorFlow, note that tensorflow 2.0 is not compatible now, you may use 

```bash
pip install tensorflow==1.15.4
 ```

## Project Structure
├── backtesting  
│   ├── comparsion.xlsx  
│   ├── perf_stats_all_20220902-00h09_3285596.csv  
│   └──  perf_stats_all_assemble_20220902-09h56.csv  
│  
├── config  
│   └──  config.py  
│  
├── data  
│  
├── env  
│   ├── StockTradingEnvTrade.py  
│   ├── StockTradingEnvTrain.py  
│   └── StockTradingEnvValidation.py  
│  
├── model  
│   ├── models.py  
│   ├── TD3.py  
│   ├── TD3_LSTM.py  
│   └──utils.py  
│  
├── preprocessing  
│   └──preprocessors.py  
│  
├── results  
├── stock_detail  
├── trained_models  
├── wait_for_process  
├── Readme.md  
├── test_done_data.csv  
├── done_data.csv  
├── backtesting.py  
├── implementTool.py  
├── TD3_LSTM_DRL.py  
└── run_DRL.py  

#### data
```data
done_data.csv && NASDAQ_done_data.csv
The first one is dow_30 and the second one is NASDAQ 100.
NASDAQ 100 are filtered out from 100 stocks to 81 stocks.
```
#### env
```env
The env for RL involves Training, Validation and Trading.
```
#### model
```model
We have two models. 
One is models.py, it involves DDPG, A2C, PPO. 
Another TD3_LSTM is ours, solving the problems according the POMDP situation based on TD3.
```
#### preprocessing
```preprocessing
preprocessing.py && implementTool.py && backtesting.py
A tool that uses for processing the data
```
#### stock_detail
```stock_detail
That is a folder saves all the operations the AI did in stockmarket.
```
#### TD3_LSTM_DRL.py && models.py in model
```main project
This is the main body of our project.
In this part, we could find 3 phases, the model training phase, validating phase and trading phase.

1. If we want to train model use NASDAQ_100 in TD3_LSTM_DRL.py, we have to change the number 30 in line 314 to 81.
2. If we want to train model use NASDAQ_100 in models.py, we have to change the number 30 in line 193 to 81.
Conversely, we have to change the number 81 to 30 for using dow_30 data.
It both about the start_date_index.
```

## Run DRL Ensemble Strategy
```shell
python run_DRL.py
```
## Backtesting

Use the package named [finrl](https://github.com/AI4Finance-Foundation/FinRL) to do the backtesting.

See [backtesting script](preprocessing/backtesting.py) for use



## Data
The stock data we use is NASDAQ 100 from 2010.12 - 2021.7
<img src=figs/data.png width="500">

### One Agent for Strategy
Our purpose is to develop a highly robust Agent for trading strategy given.
