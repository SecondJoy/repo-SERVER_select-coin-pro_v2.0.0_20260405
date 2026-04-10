import pandas as pd
import os

# 定义路径
input_path = '/home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/data/回测结果/测试策略1/evl_output/2026/forward_eval_1h_6.csv'
output_path = os.path.join(os.path.dirname(input_path), 'forward_eval_symbol_joy.csv')

def extract_latest_symbols():
    try:
        # 读取 CSV
        df = pd.read_csv(input_path)
        
        # 找到最近的日期
        latest_date = df['date'].max()
        print(f"检测到最近日期为: {latest_date}")
        
        # 筛选数据并只保留 symbol 列
        latest_symbols = df[df['date'] == latest_date][['symbol']]
        
        # 保存结果 (index=False 表示不保存行索引)
        latest_symbols.to_csv(output_path, index=False)
        
        print(f"成功导出 {len(latest_symbols)} 个 symbol 到: {output_path}")

    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    extract_latest_symbols()