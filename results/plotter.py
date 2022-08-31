import pandas as pd 
import argparse
from utils.enums import StatsType

if __name__ == '__main__':
    fd = pd.read_csv('./results/results_ppo-lstm_reverse.csv')
    fd.groupby([str(StatsType.I_EPISODE)]).mean()