import pandas as pd
import glob
import os
#example 
# python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); print(pd.read_pickle(r'C:\Users\Public\0_QTclass\BBcode\select-coin-pro_v2.0.0_20260403\data\回测结果\测试策略1\选币结果.pkl').head(5))"
#按 candle_begin_time 降序排列
#python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); df = pd.read_pickle(r'C:\Users\Public\0_QTclass\BBcode\select-coin-pro_v2.0.0_20260403\data\回测结果\测试策略1\选币结果.pkl'); print(df.sort_values('candle_begin_time', ascending=False).head(5))"
# ==========================================
# 📝 只需要在这里修改你的数据文件夹路径
# ==========================================
#DATA_DIR = r'C:\Users\Public\0_QTclass\BBcode\select-coin-pro_v2.0.0_20260403\data\回测结果\测试策略1'  # 或者用 '.' 表示当前目录
DATA_DIR = r'D:\quantclass-data\coin-binance-spot-swap-preprocess-pkl-1h-2026-04-02'  # 或者用 '.' 表示当前目录

def main():
    # 1. 设置 Pandas 显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    # 2. 自动获取 .pkl 文件
    search_path = os.path.join(DATA_DIR, "*.pkl")
    pkl_files = glob.glob(search_path)

    if not pkl_files:
        print(f"💡 在路径 [{DATA_DIR}] 下未找到 .pkl 文件。")
        return

    print(f"🚀 发现 {len(pkl_files)} 个文件，将按 candle_begin_time 降序排列显示...\n")

    for path in pkl_files:
        file_name = os.path.basename(path)
        
        print(f"{'='*40}")
        print(f"📄 文件名: {file_name}")
        print(f"{'='*40}")

        try:
            df = pd.read_pickle(path)
            
            if isinstance(df, pd.DataFrame):
                # 🔍 核心逻辑：检查是否存在该列，存在则排序
                sort_col = 'candle_begin_time'
                if sort_col in df.columns:
                    # 按照 candle_begin_time 降序排列 (ascending=False)
                    df = df.sort_values(by=sort_col, ascending=False)
                    print(f"✅ 已按 {sort_col} 降序排列")
                else:
                    print(f"⚠️ 未在文件中找到列: {sort_col}，将按原顺序显示")

                print(df.head(5))
                print(f"\n📏 规模: {df.shape}")
            else:
                print("⚠️ 该文件不是 DataFrame 格式，无法排序。预览：")
                print(str(df)[:200])

        except Exception as e:
            print(f"❌ 读取/排序出错: {e}")
        
        print("\n")

    print(f"✅ 全部读取完成。")

if __name__ == "__main__":
    main()