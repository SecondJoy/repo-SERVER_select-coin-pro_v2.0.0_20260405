
import numpy as np

def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    # 将参数n和symbol都转换为不带横杠的格式进行比较
    n_normalized = n.replace('-', '')
    symbol_normalized = df['symbol'].str.replace('-', '', regex=False)
    
    df[factor_name] = np.where(symbol_normalized == n_normalized, 1, np.nan)

    return df

