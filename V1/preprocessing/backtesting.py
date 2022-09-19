import pandas as pd
import datetime
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

print("==============Get Backtest Results===========")
df_account_value = pd.read_csv('wait_for_process/result_assemble.csv')
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("backtesting/" + "perf_stats_all_" + now + '.csv')