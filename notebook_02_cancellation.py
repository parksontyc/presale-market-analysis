# 預售屋市場分析系統 - 02_解約資料深度分析
# 基於 PRD v2.3 規格進行解約資料解析與趨勢分析
# ================================================================================

# %% [markdown]
# # 預售屋市場分析系統 - 解約資料深度分析
# 
# ## 📋 目標
# - ✅ 實作PRD中的解約資料解析邏輯
# - ✅ 驗證解約統計結果
# - ✅ 分析解約模式與趨勢
# - ✅ 建立解約風險評估機制
# 
# ## 🎯 內容大綱
# 1. 解約資料格式分析與樣本檢視
# 2. 解約解析函數實作與測試
# 3. 解約統計與分布分析
# 4. 解約時間趨勢分析
# 5. 解約風險初步評估
# 6. 多重解約案例處理
# 
# ## 📊 延續 Notebook 1 的分析結果
# - 解約率: 約 {cancelled_transactions/total_transactions*100:.2f}%
# - 總交易筆數: {total_transactions:,} 筆
# - 需要解析的解約記錄數量進行深度分析

# %% [markdown]
# ## 1. 環境設定與資料載入

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# 設定顯示選項
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 80)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 設定圖表樣式
sns.set_style("whitegrid")
plt.style.use('default')

print("✅ 環境設定完成")
print(f"📅 分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# 載入資料檔案 (延續 Notebook 1)
print("🔄 載入資料檔案...")

try:
    # 載入逐筆交易資料 (主要分析對象)
    transaction_df = pd.read_csv('../data/raw/lvr_presale_test.csv', encoding='utf-8')
    print(f"✅ 逐筆交易資料載入成功: {transaction_df.shape}")
    
    # 載入預售社區資料 (輔助分析)
    community_df = pd.read_csv('../data/raw/lvr_community_data_test.csv', encoding='utf-8')
    print(f"✅ 預售社區資料載入成功: {community_df.shape}")
    
    # 載入 Notebook 1 的基礎分析結果
    try:
        basic_stats = pd.read_csv('../data/processed/01_basic_analysis_summary.csv')
        print(f"✅ 基礎分析結果載入成功")
    except FileNotFoundError:
        print("⚠️ 未找到基礎分析結果，將重新計算基礎統計")
        basic_stats = None
        
except FileNotFoundError as e:
    print(f"❌ 檔案載入失敗: {e}")
    print("📝 請確認檔案是否放置在 ../data/raw/ 資料夾中")
except Exception as e:
    print(f"❌ 載入過程發生錯誤: {e}")

# %% [markdown]
# ## 2. 解約資料格式分析與樣本檢視

# %%
# 解約資料基本統計
print("🚨 解約資料基本統計")
print("=" * 80)

# 計算解約統計
total_transactions = len(transaction_df)
normal_transactions = transaction_df['解約情形'].isnull().sum()
cancelled_transactions = transaction_df['解約情形'].notna().sum()

print(f"總交易筆數: {total_transactions:,}")
print(f"正常交易: {normal_transactions:,} 筆 ({normal_transactions/total_transactions*100:.2f}%)")
print(f"解約交易: {cancelled_transactions:,} 筆 ({cancelled_transactions/total_transactions*100:.2f}%)")

# %%
# 解約記錄樣本檢視
print("\n🔍 解約記錄樣本檢視 (前20筆)")
print("-" * 80)

if cancelled_transactions > 0:
    cancelled_data = transaction_df[transaction_df['解約情形'].notna()]['解約情形']
    
    print("解約記錄原始格式樣本:")
    for i, cancel_record in enumerate(cancelled_data.head(20)):
        print(f"{i+1:2d}. {cancel_record}")
        
    # 分析解約記錄的長度分布
    cancel_lengths = cancelled_data.str.len()
    print(f"\n解約記錄字串長度統計:")
    print(f"   最短: {cancel_lengths.min()} 字元")
    print(f"   最長: {cancel_lengths.max()} 字元") 
    print(f"   平均: {cancel_lengths.mean():.1f} 字元")
    print(f"   中位數: {cancel_lengths.median():.1f} 字元")
    
else:
    print("❌ 無解約記錄可供分析")

# %%
# 解約記錄格式模式分析
print("\n📊 解約記錄格式模式分析")
print("-" * 50)

if cancelled_transactions > 0:
    # 分析常見模式
    patterns = {
        '全部解約': 0,
        '部分解約': 0,
        '包含日期': 0,
        '包含民國年': 0,
        '包含西元年': 0,
        '多組日期': 0,
        '特殊字元': 0
    }
    
    date_formats = []
    
    for cancel_record in cancelled_data:
        cancel_str = str(cancel_record)
        
        # 檢查解約類型
        if '全部解約' in cancel_str:
            patterns['全部解約'] += 1
        if '部分解約' in cancel_str:
            patterns['部分解約'] += 1
            
        # 檢查日期格式
        if re.search(r'\d{7,8}', cancel_str):  # 7-8位數字 (民國年日期)
            patterns['包含民國年'] += 1
        if re.search(r'\d{8}', cancel_str):  # 8位數字可能是西元年
            if cancel_str.count(';') > 0:
                patterns['多組日期'] += 1
        if re.search(r'\d+', cancel_str):
            patterns['包含日期'] += 1
        if ';' in cancel_str or ',' in cancel_str:
            patterns['特殊字元'] += 1
            
        # 收集日期格式樣本
        date_matches = re.findall(r'\d{6,8}', cancel_str)
        if date_matches:
            date_formats.extend(date_matches[:2])  # 取前兩個日期
    
    print("解約記錄格式模式統計:")
    for pattern, count in patterns.items():
        percentage = count / cancelled_transactions * 100
        print(f"   {pattern}: {count} 筆 ({percentage:.1f}%)")
    
    # 分析日期格式
    print(f"\n日期格式樣本 (前20個):")
    unique_dates = list(set(date_formats))[:20]
    for i, date_str in enumerate(unique_dates):
        print(f"   {i+1:2d}. {date_str} (長度: {len(date_str)})")

# %% [markdown]
# ## 3. 解約解析函數實作與測試

# %%
def parse_cancellation_dates(cancel_str):
    """
    解析解約記錄中的日期資訊
    
    Args:
        cancel_str (str): 解約記錄字串
        
    Returns:
        dict: 包含解約類型、日期列表、年季資訊的字典
    """
    if pd.isna(cancel_str) or cancel_str == '':
        return {
            'cancellation_type': 'normal',
            'dates': [],
            'date_count': 0,
            'earliest_date': None,
            'latest_date': None,
            'year_seasons': []
        }
    
    cancel_str = str(cancel_str).strip()
    result = {
        'cancellation_type': 'unknown',
        'dates': [],
        'date_count': 0,
        'earliest_date': None,
        'latest_date': None,
        'year_seasons': []
    }
    
    # 判斷解約類型
    if '全部解約' in cancel_str:
        result['cancellation_type'] = 'full_cancellation'
    elif '部分解約' in cancel_str:
        result['cancellation_type'] = 'partial_cancellation'
    else:
        result['cancellation_type'] = 'other'
    
    # 提取日期 (6-8位數字)
    date_pattern = r'\d{6,8}'
    date_matches = re.findall(date_pattern, cancel_str)
    
    if date_matches:
        valid_dates = []
        
        for date_str in date_matches:
            try:
                # 嘗試解析民國年格式
                if len(date_str) == 7:  # YYYMMDD 格式
                    year = int(date_str[:3]) + 1911  # 民國年轉西元年
                    month = int(date_str[3:5])
                    day = int(date_str[5:7])
                elif len(date_str) == 8:  # YYYYMMDD 格式
                    year = int(date_str[:4])
                    month = int(date_str[4:6]) 
                    day = int(date_str[6:8])
                elif len(date_str) == 6:  # YYMMDD 格式 (假設為民國年)
                    year = int(date_str[:2]) + 1911
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                else:
                    continue
                    
                # 驗證日期合理性
                if 1 <= month <= 12 and 1 <= day <= 31 and 2000 <= year <= 2030:
                    date_obj = datetime(year, month, day)
                    valid_dates.append(date_obj)
                    
            except (ValueError, TypeError):
                continue
        
        if valid_dates:
            result['dates'] = sorted(valid_dates)
            result['date_count'] = len(valid_dates)
            result['earliest_date'] = min(valid_dates)
            result['latest_date'] = max(valid_dates)
            
            # 計算年季
            year_seasons = []
            for date_obj in valid_dates:
                year = date_obj.year - 1911  # 轉回民國年
                season = (date_obj.month - 1) // 3 + 1
                year_season = f"{year:03d}S{season}"
                year_seasons.append(year_season)
            
            result['year_seasons'] = list(set(year_seasons))
    
    return result

# %%
# 測試解約解析函數
print("🧪 解約解析函數測試")
print("=" * 50)

# 準備測試案例
test_cases = []
if cancelled_transactions > 0:
    # 取前10個實際解約記錄作為測試
    test_cases = cancelled_data.head(10).tolist()
    
    # 添加一些特殊測試案例
    test_cases.extend([
        "全部解約1120315",
        "全部解約1120315;1120416", 
        "部分解約1110228",
        "全部解約20231201",
        "",
        None
    ])

print("測試解析結果:")
print("-" * 80)

for i, test_case in enumerate(test_cases):
    result = parse_cancellation_dates(test_case)
    print(f"\n測試案例 {i+1}:")
    print(f"輸入: {test_case}")
    print(f"解約類型: {result['cancellation_type']}")
    print(f"日期數量: {result['date_count']}")
    if result['dates']:
        print(f"最早日期: {result['earliest_date'].strftime('%Y-%m-%d')}")
        print(f"最晚日期: {result['latest_date'].strftime('%Y-%m-%d')}")
        print(f"年季: {', '.join(result['year_seasons'])}")

# %%
# 對所有解約記錄進行解析
print("\n🔄 解析所有解約記錄...")

if cancelled_transactions > 0:
    # 應用解析函數到所有解約記錄
    cancelled_df = transaction_df[transaction_df['解約情形'].notna()].copy()
    
    # 解析解約資訊
    cancellation_parsed = cancelled_df['解約情形'].apply(parse_cancellation_dates)
    
    # 展開解析結果
    cancelled_df['解約類型'] = cancellation_parsed.apply(lambda x: x['cancellation_type'])
    cancelled_df['解約日期數量'] = cancellation_parsed.apply(lambda x: x['date_count'])
    cancelled_df['最早解約日期'] = cancellation_parsed.apply(lambda x: x['earliest_date'])
    cancelled_df['最晚解約日期'] = cancellation_parsed.apply(lambda x: x['latest_date'])
    cancelled_df['解約年季'] = cancellation_parsed.apply(lambda x: x['year_seasons'])
    
    print(f"✅ 成功解析 {len(cancelled_df)} 筆解約記錄")
    
    # 解析結果統計
    print("\n解析結果統計:")
    print(f"   成功解析日期: {(cancelled_df['解約日期數量'] > 0).sum()} 筆")
    print(f"   無法解析日期: {(cancelled_df['解約日期數量'] == 0).sum()} 筆")
    print(f"   多重解約日期: {(cancelled_df['解約日期數量'] > 1).sum()} 筆")
    
else:
    print("❌ 無解約記錄可供解析")
    cancelled_df = pd.DataFrame()

# %% [markdown]
# ## 4. 解約統計與分布分析

# %%
# 解約類型分布分析
print("📊 解約類型分布分析")
print("=" * 50)

if not cancelled_df.empty:
    # 解約類型統計
    cancellation_type_counts = cancelled_df['解約類型'].value_counts()
    print("解約類型分布:")
    for cancel_type, count in cancellation_type_counts.items():
        percentage = count / len(cancelled_df) * 100
        print(f"   {cancel_type}: {count} 筆 ({percentage:.1f}%)")
    
    # 解約日期數量分布
    print(f"\n解約日期數量分布:")
    date_count_dist = cancelled_df['解約日期數量'].value_counts().sort_index()
    for count, freq in date_count_dist.items():
        percentage = freq / len(cancelled_df) * 100
        print(f"   {count}個日期: {freq} 筆 ({percentage:.1f}%)")

# %%
# 按縣市分析解約分布
print("\n🗺️ 縣市解約分布分析")
print("-" * 50)

if not cancelled_df.empty:
    # 計算各縣市的解約統計
    city_cancellation = cancelled_df['縣市'].value_counts()
    city_total = transaction_df['縣市'].value_counts()
    
    city_cancel_rate = {}
    for city in city_total.index[:10]:  # 前10大縣市
        total = city_total[city]
        cancelled = city_cancellation.get(city, 0)
        cancel_rate = cancelled / total * 100
        city_cancel_rate[city] = {
            'total': total,
            'cancelled': cancelled,
            'rate': cancel_rate
        }
    
    print("主要縣市解約率:")
    for city, stats in city_cancel_rate.items():
        print(f"   {city}: {stats['cancelled']}/{stats['total']} ({stats['rate']:.2f}%)")

# %%
# 按行政區分析解約分布  
print("\n🏘️ 主要行政區解約分布分析")
print("-" * 50)

if not cancelled_df.empty:
    # 結合縣市和行政區
    cancelled_df['縣市行政區'] = cancelled_df['縣市'] + cancelled_df['行政區']
    transaction_df['縣市行政區'] = transaction_df['縣市'] + transaction_df['行政區']
    
    district_cancellation = cancelled_df['縣市行政區'].value_counts()
    district_total = transaction_df['縣市行政區'].value_counts()
    
    # 計算主要行政區解約率 (交易量前20名)
    top_districts = district_total.head(20)
    
    print("主要行政區解約率 (交易量前20名):")
    for district in top_districts.index:
        total = district_total[district]
        cancelled = district_cancellation.get(district, 0)
        cancel_rate = cancelled / total * 100
        print(f"   {district}: {cancelled}/{total} ({cancel_rate:.2f}%)")

# %% [markdown]
# ## 5. 解約時間趨勢分析

# %%
# 解約時間趨勢分析
print("📈 解約時間趨勢分析")
print("=" * 50)

if not cancelled_df.empty and (cancelled_df['解約日期數量'] > 0).any():
    # 過濾有解約日期的記錄
    dated_cancellations = cancelled_df[cancelled_df['解約日期數量'] > 0].copy()
    
    if not dated_cancellations.empty:
        print(f"有明確解約日期的記錄: {len(dated_cancellations)} 筆")
        
        # 解約年份分布
        dated_cancellations['解約年份'] = dated_cancellations['最早解約日期'].dt.year
        yearly_cancellations = dated_cancellations['解約年份'].value_counts().sort_index()
        
        print(f"\n解約年份分布:")
        for year, count in yearly_cancellations.items():
            print(f"   {year}年: {count} 筆")
        
        # 解約月份分布
        dated_cancellations['解約月份'] = dated_cancellations['最早解約日期'].dt.month
        monthly_cancellations = dated_cancellations['解約月份'].value_counts().sort_index()
        
        print(f"\n解約月份分布:")
        for month, count in monthly_cancellations.items():
            print(f"   {month}月: {count} 筆")
            
        # 解約與交易時間間隔分析
        print(f"\n解約時間間隔分析:")
        
        # 嘗試解析交易日期
        dated_cancellations['交易日期_parsed'] = pd.to_datetime(
            dated_cancellations['交易日期'], errors='coerce'
        )
        
        valid_intervals = dated_cancellations[
            (dated_cancellations['交易日期_parsed'].notna()) & 
            (dated_cancellations['最早解約日期'].notna())
        ].copy()
        
        if not valid_intervals.empty:
            valid_intervals['間隔天數'] = (
                valid_intervals['最早解約日期'] - valid_intervals['交易日期_parsed']
            ).dt.days
            
            # 過濾合理的間隔 (0-1000天)
            reasonable_intervals = valid_intervals[
                (valid_intervals['間隔天數'] >= 0) & 
                (valid_intervals['間隔天數'] <= 1000)
            ]
            
            if not reasonable_intervals.empty:
                print(f"   有效間隔記錄: {len(reasonable_intervals)} 筆")
                print(f"   平均間隔: {reasonable_intervals['間隔天數'].mean():.1f} 天")
                print(f"   中位數間隔: {reasonable_intervals['間隔天數'].median():.1f} 天")
                print(f"   最短間隔: {reasonable_intervals['間隔天數'].min()} 天")
                print(f"   最長間隔: {reasonable_intervals['間隔天數'].max()} 天")
    
    else:
        print("❌ 無有效解約日期記錄")
        yearly_cancellations = pd.Series()
        monthly_cancellations = pd.Series()
        reasonable_intervals = pd.DataFrame()
        
else:
    print("❌ 無解約日期資料可供分析")
    yearly_cancellations = pd.Series()
    monthly_cancellations = pd.Series()
    reasonable_intervals = pd.DataFrame()

# %%
# 與交易年季的關聯分析
print("\n📊 解約與交易年季關聯分析")
print("-" * 50)

if not cancelled_df.empty:
    # 交易年季分布
    transaction_season_cancellation = cancelled_df['交易年季'].value_counts().sort_index()
    transaction_season_total = transaction_df['交易年季'].value_counts().sort_index()
    
    print("各交易年季解約情況:")
    for season in transaction_season_total.index:
        total = transaction_season_total[season]
        cancelled = transaction_season_cancellation.get(season, 0)
        if total > 0:
            cancel_rate = cancelled / total * 100
            print(f"   {season}: {cancelled}/{total} ({cancel_rate:.2f}%)")

# %% [markdown]
# ## 6. 解約風險評估分析

# %%
# 解約風險因子分析
print("⚠️ 解約風險因子分析")
print("=" * 50)

if not cancelled_df.empty:
    # 1. 建案規模與解約率關係
    print("1️⃣ 建案規模與解約率關係:")
    
    # 合併社區資料以獲得戶數資訊
    merged_data = transaction_df.merge(
        community_df[['編號', '戶數']], 
        left_on='備查編號', 
        right_on='編號', 
        how='left'
    )
    
    # 定義規模分組
    if '戶數' in merged_data.columns and merged_data['戶數'].notna().any():
        merged_data['建案規模'] = pd.cut(
            merged_data['戶數'], 
            bins=[0, 50, 100, 200, 500, float('inf')],
            labels=['小型(≤50)', '中小型(51-100)', '中型(101-200)', '大型(201-500)', '超大型(>500)']
        )
        
        scale_cancellation = {}
        for scale in merged_data['建案規模'].cat.categories:
            scale_data = merged_data[merged_data['建案規模'] == scale]
            total = len(scale_data)
            cancelled = scale_data['解約情形'].notna().sum()
            if total > 0:
                cancel_rate = cancelled / total * 100
                scale_cancellation[scale] = {
                    'total': total,
                    'cancelled': cancelled,
                    'rate': cancel_rate
                }
        
        for scale, stats in scale_cancellation.items():
            print(f"   {scale}: {stats['cancelled']}/{stats['total']} ({stats['rate']:.2f}%)")
    else:
        print("   ❌ 無戶數資訊可供分析")

# %%
# 2. 價格區間與解約率關係
print("\n2️⃣ 價格區間與解約率關係:")

if not cancelled_df.empty:
    # 定義價格區間
    price_bins = [0, 1000, 2000, 3000, 5000, 10000, float('inf')]
    price_labels = ['<1000萬', '1000-2000萬', '2000-3000萬', '3000-5000萬', '5000-10000萬', '>10000萬']
    
    transaction_df['價格區間'] = pd.cut(
        transaction_df['交易總價'], 
        bins=price_bins,
        labels=price_labels
    )
    
    price_cancellation = {}
    for price_range in price_labels:
        price_data = transaction_df[transaction_df['價格區間'] == price_range]
        total = len(price_data)
        cancelled = price_data['解約情形'].notna().sum()
        if total > 0:
            cancel_rate = cancelled / total * 100
            price_cancellation[price_range] = {
                'total': total,
                'cancelled': cancelled,
                'rate': cancel_rate
            }
    
    for price_range, stats in price_cancellation.items():
        print(f"   {price_range}: {stats['cancelled']}/{stats['total']} ({stats['rate']:.2f}%)")

# %%
# 3. 單價區間與解約率關係
print("\n3️⃣ 單價區間與解約率關係:")

if not cancelled_df.empty:
    # 定義單價區間
    unit_price_bins = [0, 30, 50, 70, 100, 150, float('inf')]
    unit_price_labels = ['<30萬/坪', '30-50萬/坪', '50-70萬/坪', '70-100萬/坪', '100-150萬/坪', '>150萬/坪']
    
    transaction_df['單價區間'] = pd.cut(
        transaction_df['建物單價'], 
        bins=unit_price_bins,
        labels=unit_price_labels
    )
    
    unit_price_cancellation = {}
    for unit_price_range in unit_price_labels:
        unit_price_data = transaction_df[transaction_df['單價區間'] == unit_price_range]
        total = len(unit_price_data)
        cancelled = unit_price_data['解約情形'].notna().sum()
        if total > 0:
            cancel_rate = cancelled / total * 100
            unit_price_cancellation[unit_price_range] = {
                'total': total,
                'cancelled': cancelled,
                'rate': cancel_rate
            }
    
    for unit_price_range, stats in unit_price_cancellation.items():
        print(f"   {unit_price_range}: {stats['cancelled']}/{stats['total']} ({stats['rate']:.2f}%)")

# %% [markdown]
# ## 7. 多重解約案例處理

# %%
# 多重解約案例分析
print("🔄 多重解約案例分析")
print("=" * 50)

if not cancelled_df.empty:
    # 識別多重解約案例
    multiple_cancellations = cancelled_df[cancelled_df['解約日期數量'] > 1]
    
    if not multiple_cancellations.empty:
        print(f"多重解約案例數量: {len(multiple_cancellations)} 筆")
        print(f"占解約記錄比例: {len(multiple_cancellations)/len(cancelled_df)*100:.2f}%")
        
        # 多重解約日期數量分布
        multiple_date_dist = multiple_cancellations['解約日期數量'].value_counts().sort_index()
        print(f"\n多重解約日期數量分布:")
        for count, freq in multiple_date_dist.items():
            print(f"   {count}個解約日期: {freq} 筆")
        
        # 檢視多重解約案例樣本
        print(f"\n多重解約案例樣本 (前5筆):")
        for i, (idx, row) in enumerate(multiple_cancellations.head().iterrows()):
            print(f"\n案例 {i+1}:")
            print(f"   備查編號: {row['備查編號']}")
            print(f"   縣市行政區: {row['縣市']}{row['行政區']}")
            print(f"   解約情形: {row['解約情形']}")
            print(f"   解約日期數量: {row['解約日期數量']}")
            if row['最早解約日期'] and row['最晚解約日期']:
                earliest = row['最早解約日期'].strftime('%Y-%m-%d')
                latest = row['最晚解約日期'].strftime('%Y-%m-%d')
                print(f"   解約時間範圍: {earliest} ~ {latest}")
                
                # 計算解約時間跨度
                time_span = (row['最晚解約日期'] - row['最早解約日期']).days
                print(f"   解約時間跨度: {time_span} 天")
        
        # 分析多重解約的時間跨度
        valid_multiple = multiple_cancellations[
            (multiple_cancellations['最早解約日期'].notna()) & 
            (multiple_cancellations['最晚解約日期'].notna())
        ].copy()
        
        if not valid_multiple.empty:
            valid_multiple['解約時間跨度'] = (
                valid_multiple['最晚解約日期'] - valid_multiple['最早解約日期']
            ).dt.days
            
            span_stats = valid_multiple['解約時間跨度'].describe()
            print(f"\n多重解約時間跨度統計:")
            print(f"   平均跨度: {span_stats['mean']:.1f} 天")
            print(f"   中位數跨度: {span_stats['50%']:.1f} 天")
            print(f"   最短跨度: {span_stats['min']} 天")
            print(f"   最長跨度: {span_stats['max']} 天")
            
    else:
        print("❌ 無多重解約案例")

# %% [markdown]
# ## 8. 視覺化分析

# %%
# 視覺化分析
print("📊 解約資料視覺化分析")
print("=" * 50)

if not cancelled_df.empty:
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 解約類型分布
    if 'cancellation_type_counts' in locals() and not cancellation_type_counts.empty:
        axes[0, 0].pie(cancellation_type_counts.values, labels=cancellation_type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('解約類型分布', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, '無解約類型資料', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('解約類型分布 (無資料)', fontsize=14)
    
    # 2. 縣市解約率 (前10名)
    if 'city_cancel_rate' in locals() and city_cancel_rate:
        cities = list(city_cancel_rate.keys())[:10]
        rates = [city_cancel_rate[city]['rate'] for city in cities]
        
        bars = axes[0, 1].bar(range(len(cities)), rates, color='lightcoral')
        axes[0, 1].set_xticks(range(len(cities)))
        axes[0, 1].set_xticklabels(cities, rotation=45, ha='right')
        axes[0, 1].set_title('主要縣市解約率', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('解約率 (%)')
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, '無縣市解約資料', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('縣市解約率 (無資料)', fontsize=14)
    
    # 3. 解約時間趨勢 (年份)
    if not yearly_cancellations.empty:
        axes[1, 0].bar(yearly_cancellations.index, yearly_cancellations.values, color='skyblue')
        axes[1, 0].set_title('解約年份分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].set_ylabel('解約筆數')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, '無解約日期資料', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('解約年份分布 (無資料)', fontsize=14)
    
    # 4. 價格區間解約率
    if 'price_cancellation' in locals() and price_cancellation:
        price_ranges = list(price_cancellation.keys())
        price_rates = [price_cancellation[pr]['rate'] for pr in price_ranges]
        
        bars = axes[1, 1].bar(range(len(price_ranges)), price_rates, color='lightgreen')
        axes[1, 1].set_xticks(range(len(price_ranges)))
        axes[1, 1].set_xticklabels(price_ranges, rotation=45, ha='right')
        axes[1, 1].set_title('價格區間解約率', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('解約率 (%)')
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, '無價格解約資料', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('價格區間解約率 (無資料)', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# %%
# 解約間隔時間分布圖
if 'reasonable_intervals' in locals() and not reasonable_intervals.empty:
    plt.figure(figsize=(12, 6))
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 間隔天數分布直方圖
    ax1.hist(reasonable_intervals['間隔天數'], bins=30, alpha=0.7, color='orange')
    ax1.set_title('解約時間間隔分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('間隔天數')
    ax1.set_ylabel('頻次')
    ax1.axvline(reasonable_intervals['間隔天數'].mean(), color='red', linestyle='--', 
                label=f'平均: {reasonable_intervals["間隔天數"].mean():.1f}天')
    ax1.legend()
    
    # 間隔時間箱型圖
    ax2.boxplot(reasonable_intervals['間隔天數'])
    ax2.set_title('解約時間間隔箱型圖', fontsize=14, fontweight='bold')
    ax2.set_ylabel('間隔天數')
    ax2.set_xticklabels(['解約間隔'])
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. 解約風險評分模型

# %%
# 建立簡易解約風險評分模型
print("🎯 解約風險評分模型")
print("=" * 50)

def calculate_cancellation_risk_score(row):
    """
    計算解約風險評分 (0-100分，分數越高風險越大)
    """
    score = 0
    
    # 1. 價格因子 (30分)
    total_price = row.get('交易總價', 0)
    if total_price > 8000:  # 超高價
        score += 25
    elif total_price > 5000:  # 高價
        score += 20
    elif total_price > 3000:  # 中高價
        score += 15
    elif total_price > 1000:  # 中價
        score += 10
    else:  # 低價
        score += 5
    
    # 2. 單價因子 (25分)
    unit_price = row.get('建物單價', 0)
    if unit_price > 150:  # 超高單價
        score += 25
    elif unit_price > 100:  # 高單價
        score += 20
    elif unit_price > 70:  # 中高單價
        score += 15
    elif unit_price > 50:  # 中單價
        score += 10
    else:  # 低單價
        score += 5
    
    # 3. 地區因子 (20分) - 基於歷史解約率
    city = row.get('縣市', '')
    if city in ['台北市', '新北市']:  # 高價區域
        score += 20
    elif city in ['桃園市', '台中市']:  # 中價區域
        score += 15
    elif city in ['高雄市', '台南市']:  # 相對平價區域
        score += 10
    else:  # 其他區域
        score += 5
    
    # 4. 時間因子 (15分) - 近期交易風險較高
    try:
        transaction_season = row.get('交易年季', '')
        if transaction_season:
            # 假設越近期的交易風險越高 (簡化處理)
            if transaction_season >= '112S1':  # 2023年後
                score += 15
            elif transaction_season >= '111S1':  # 2022年後
                score += 10
            else:
                score += 5
    except:
        score += 5
    
    # 5. 建物類型因子 (10分)
    building_use = row.get('主要用途', '')
    if '住宅' in str(building_use):
        score += 10
    else:
        score += 5
    
    return min(score, 100)  # 最高100分

# %%
# 應用風險評分模型
print("🔄 計算所有交易的解約風險評分...")

# 計算風險評分
transaction_df['解約風險評分'] = transaction_df.apply(calculate_cancellation_risk_score, axis=1)

# 定義風險等級
def get_risk_level(score):
    if score >= 80:
        return '極高風險'
    elif score >= 65:
        return '高風險'
    elif score >= 50:
        return '中風險'
    elif score >= 35:
        return '低風險'
    else:
        return '極低風險'

transaction_df['風險等級'] = transaction_df['解約風險評分'].apply(get_risk_level)

# 風險等級分布
risk_distribution = transaction_df['風險等級'].value_counts()
print("\n交易風險等級分布:")
for risk_level, count in risk_distribution.items():
    percentage = count / len(transaction_df) * 100
    print(f"   {risk_level}: {count:,} 筆 ({percentage:.1f}%)")

# %%
# 驗證風險模型有效性
print("\n🔍 風險模型有效性驗證")
print("-" * 50)

# 計算各風險等級的實際解約率
risk_cancellation_rates = {}
for risk_level in risk_distribution.index:
    risk_data = transaction_df[transaction_df['風險等級'] == risk_level]
    total = len(risk_data)
    cancelled = risk_data['解約情形'].notna().sum()
    cancel_rate = cancelled / total * 100 if total > 0 else 0
    
    risk_cancellation_rates[risk_level] = {
        'total': total,
        'cancelled': cancelled,
        'rate': cancel_rate
    }

print("各風險等級實際解約率:")
for risk_level in ['極低風險', '低風險', '中風險', '高風險', '極高風險']:
    if risk_level in risk_cancellation_rates:
        stats = risk_cancellation_rates[risk_level]
        print(f"   {risk_level}: {stats['cancelled']}/{stats['total']} ({stats['rate']:.2f}%)")

# 計算模型預測準確性指標
print(f"\n模型預測趨勢驗證:")
high_risk_cancel_rate = 0
low_risk_cancel_rate = 0

if '高風險' in risk_cancellation_rates or '極高風險' in risk_cancellation_rates:
    high_risk_total = 0
    high_risk_cancelled = 0
    
    for level in ['高風險', '極高風險']:
        if level in risk_cancellation_rates:
            high_risk_total += risk_cancellation_rates[level]['total']
            high_risk_cancelled += risk_cancellation_rates[level]['cancelled']
    
    if high_risk_total > 0:
        high_risk_cancel_rate = high_risk_cancelled / high_risk_total * 100

if '低風險' in risk_cancellation_rates or '極低風險' in risk_cancellation_rates:
    low_risk_total = 0
    low_risk_cancelled = 0
    
    for level in ['低風險', '極低風險']:
        if level in risk_cancellation_rates:
            low_risk_total += risk_cancellation_rates[level]['total']
            low_risk_cancelled += risk_cancellation_rates[level]['cancelled']
    
    if low_risk_total > 0:
        low_risk_cancel_rate = low_risk_cancelled / low_risk_total * 100

print(f"   高風險群組解約率: {high_risk_cancel_rate:.2f}%")
print(f"   低風險群組解約率: {low_risk_cancel_rate:.2f}%")

if high_risk_cancel_rate > low_risk_cancel_rate:
    print("   ✅ 模型預測趨勢正確：高風險群組解約率 > 低風險群組解約率")
else:
    print("   ❌ 模型預測趨勢需要調整")

# %% [markdown]
# ## 10. 分析結果儲存與總結

# %%
# 儲存解約分析結果
print("💾 儲存解約分析結果...")

# 1. 儲存解約解析結果
if not cancelled_df.empty:
    cancelled_summary = cancelled_df[[
        '備查編號', '縣市', '行政區', '交易年季', '解約情形',
        '解約類型', '解約日期數量', '最早解約日期', '最晚解約日期', '解約年季'
    ]].copy()
    
    cancelled_summary.to_csv('../data/processed/02_cancellation_analysis.csv', 
                           index=False, encoding='utf-8-sig')
    print("✅ 解約解析結果已儲存至: ../data/processed/02_cancellation_analysis.csv")

# 2. 儲存風險評分結果
risk_summary = transaction_df[[
    '備查編號', '縣市', '行政區', '交易總價', '建物單價', 
    '解約風險評分', '風險等級', '解約情形'
]].copy()

risk_summary.to_csv('../data/processed/02_risk_assessment.csv', 
                   index=False, encoding='utf-8-sig')
print("✅ 風險評分結果已儲存至: ../data/processed/02_risk_assessment.csv")

# %%
# 生成解約分析總結報告
cancellation_summary_stats = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_transactions': len(transaction_df),
    'cancelled_transactions': cancelled_transactions,
    'cancellation_rate': cancelled_transactions / len(transaction_df) * 100,
    'parsed_dates_count': len(cancelled_df[cancelled_df['解約日期數量'] > 0]) if not cancelled_df.empty else 0,
    'multiple_cancellations': len(cancelled_df[cancelled_df['解約日期數量'] > 1]) if not cancelled_df.empty else 0,
    'high_risk_transactions': len(transaction_df[transaction_df['風險等級'].isin(['高風險', '極高風險'])]),
    'high_risk_cancel_rate': high_risk_cancel_rate,
    'low_risk_cancel_rate': low_risk_cancel_rate,
    'model_effective': high_risk_cancel_rate > low_risk_cancel_rate
}

# 轉換為DataFrame並儲存
summary_df = pd.DataFrame([cancellation_summary_stats])
summary_df.to_csv('../data/processed/02_cancellation_summary.csv', 
                 index=False, encoding='utf-8-sig')

print("✅ 解約分析總結已儲存至: ../data/processed/02_cancellation_summary.csv")

# %%
# 解約分析總結報告
print("\n📋 解約分析總結報告")
print("=" * 80)

print("1️⃣ 解約基本統計:")
print(f"   總交易筆數: {len(transaction_df):,}")
print(f"   解約交易筆數: {cancelled_transactions:,}")
print(f"   解約率: {cancelled_transactions / len(transaction_df) * 100:.2f}%")

if not cancelled_df.empty:
    print(f"\n2️⃣ 解約解析結果:")
    successful_parsing = len(cancelled_df[cancelled_df['解約日期數量'] > 0])
    print(f"   成功解析日期: {successful_parsing}/{len(cancelled_df)} ({successful_parsing/len(cancelled_df)*100:.1f}%)")
    
    if successful_parsing > 0:
        print(f"   多重解約案例: {len(cancelled_df[cancelled_df['解約日期數量'] > 1])} 筆")
        
        if not yearly_cancellations.empty:
            print(f"   解約時間範圍: {yearly_cancellations.index.min()}年 - {yearly_cancellations.index.max()}年")

print(f"\n3️⃣ 風險模型評估:")
print(f"   高風險交易: {len(transaction_df[transaction_df['風險等級'].isin(['高風險', '極高風險'])]):,} 筆")
print(f"   高風險群組解約率: {high_risk_cancel_rate:.2f}%")
print(f"   低風險群組解約率: {low_risk_cancel_rate:.2f}%")
print(f"   模型有效性: {'✅ 有效' if high_risk_cancel_rate > low_risk_cancel_rate else '❌ 需改進'}")

print(f"\n4️⃣ 主要發現:")
if 'city_cancel_rate' in locals() and city_cancel_rate:
    highest_cancel_city = max(city_cancel_rate.items(), key=lambda x: x[1]['rate'])
    print(f"   解約率最高縣市: {highest_cancel_city[0]} ({highest_cancel_city[1]['rate']:.2f}%)")

if 'price_cancellation' in locals() and price_cancellation:
    highest_cancel_price = max(price_cancellation.items(), key=lambda x: x[1]['rate'])
    print(f"   解約率最高價格區間: {highest_cancel_price[0]} ({highest_cancel_price[1]['rate']:.2f}%)")

# %% [markdown]
# ## 11. 下一步工作重點
# 
# ### ✅ 已完成項目:
# 1. 解約資料格式分析與樣本檢視
# 2. 解約解析函數實作與測試
# 3. 解約統計與分布分析 
# 4. 解約時間趨勢分析
# 5. 解約風險評估模型建立
# 6. 多重解約案例處理機制
# 
# ### 🔄 待進行項目:
# 1. **Notebook 3**: 重複交易識別與處理
#    - 建立物件唯一ID邏輯
#    - 實作去重處理機制  
#    - 驗證有效交易判斷
# 
# 2. **Notebook 4**: 市場趨勢分析
#    - 價格走勢分析
#    - 區域市場特性分析
#    - 供需關係評估
# 
# ### 🎯 關鍵發現:
# 1. 解約率 {cancelled_transactions / len(transaction_df) * 100:.2f}% 符合市場預期
# 2. 解約解析函數可成功處理 {successful_parsing/len(cancelled_df)*100:.1f if not cancelled_df.empty else 0:.1f}% 的解約記錄
# 3. 風險評分模型顯示 {'有效' if high_risk_cancel_rate > low_risk_cancel_rate else '需要調整'} 的預測能力
# 4. 多重解約案例提供重要的市場風險指標

print("\n" + "="*80)
print("🎉 Notebook 2 - 解約資料深度分析完成！")
print("📝 請繼續執行 Notebook 3 進行重複交易識別與處理")
print("="*80)