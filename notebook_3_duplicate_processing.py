# 預售屋市場分析系統 - 03_重複交易識別與處理
# 基於 PRD v2.3 規格進行重複交易去重與有效交易判斷
# ================================================================================

# %% [markdown]
# # 預售屋市場分析系統 - 重複交易識別與處理
# 
# ## 📋 目標
# - ✅ 實作PRD中的重複交易識別邏輯
# - ✅ 建立有效交易判斷機制
# - ✅ 驗證去重處理結果
# - ✅ 分析重複交易模式與影響
# 
# ## 🎯 內容大綱
# 1. 物件唯一ID建立邏輯
# 2. 重複交易識別與分組
# 3. 有效交易判斷規則實作
# 4. 去重處理結果驗證
# 5. 重複交易模式分析
# 6. 資料品質影響評估
# 
# ## 📊 延續 Notebook 1-2 的分析結果
# - 總交易筆數: 43,007 筆
# - 解約交易筆數: 293 筆 (0.68%)
# - 需要進行去重處理以提升資料品質

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
from collections import Counter
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
# 載入資料檔案 (延續 Notebook 1-2)
print("🔄 載入資料檔案...")

try:
    # 載入逐筆交易資料 (主要分析對象)
    transaction_df = pd.read_csv('../data/raw/lvr_presale_test.csv', encoding='utf-8')
    print(f"✅ 逐筆交易資料載入成功: {transaction_df.shape}")
    
    # 載入預售社區資料 (輔助分析)
    community_df = pd.read_csv('../data/raw/lvr_community_data_test.csv', encoding='utf-8')
    print(f"✅ 預售社區資料載入成功: {community_df.shape}")
    
    # 載入解約分析結果
    try:
        cancellation_df = pd.read_csv('../data/processed/02_cancellation_analysis.csv', encoding='utf-8')
        print(f"✅ 解約分析結果載入成功: {cancellation_df.shape}")
    except FileNotFoundError:
        print("⚠️ 未找到解約分析結果，將重新計算解約資訊")
        cancellation_df = None
        
except FileNotFoundError as e:
    print(f"❌ 檔案載入失敗: {e}")
    print("📝 請確認檔案是否放置在 ../data/raw/ 資料夾中")
except Exception as e:
    print(f"❌ 載入過程發生錯誤: {e}")

# %% [markdown]
# ## 2. 物件唯一ID建立邏輯

# %%
# 根據PRD定義建立物件唯一ID
print("🔑 建立物件唯一ID")
print("=" * 80)

def create_property_id(row):
    """
    根據PRD規格建立物件唯一識別碼
    物件唯一識別 = 備查編號 + 坐落街道 + 樓層
    
    Args:
        row (pd.Series): 交易記錄
        
    Returns:
        str: 物件唯一ID
    """
    try:
        # 備查編號
        property_code = str(row.get('備查編號', '')).strip()
        
        # 坐落街道
        street = str(row.get('坐落街道', '')).strip()
        
        # 樓層資訊
        floor_info = str(row.get('樓層', '')).strip()
        
        # 組合唯一ID
        property_id = f"{property_code}_{street}_{floor_info}"
        
        # 清理特殊字元
        property_id = re.sub(r'[^\w\-_]', '_', property_id)
        
        return property_id
        
    except Exception as e:
        return f"ERROR_{hash(str(row))}"

# 應用物件ID建立邏輯
print("🔄 建立所有交易記錄的物件唯一ID...")

transaction_df['物件唯一ID'] = transaction_df.apply(create_property_id, axis=1)

print(f"✅ 成功建立 {len(transaction_df)} 筆交易記錄的物件ID")

# 檢視物件ID樣本
print(f"\n物件ID樣本 (前10筆):")
sample_ids = transaction_df[['備查編號', '坐落街道', '樓層', '物件唯一ID']].head(10)
for i, (idx, row) in enumerate(sample_ids.iterrows()):
    print(f"{i+1:2d}. {row['物件唯一ID']}")
    print(f"    備查編號: {row['備查編號']}")
    print(f"    坐落街道: {row['坐落街道']}")
    print(f"    樓層: {row['樓層']}")

# %%
# 物件ID統計分析
print(f"\n📊 物件ID統計分析")
print("-" * 50)

# 計算唯一物件數量
unique_properties = transaction_df['物件唯一ID'].nunique()
total_transactions = len(transaction_df)

print(f"總交易筆數: {total_transactions:,}")
print(f"唯一物件數量: {unique_properties:,}")
print(f"平均每物件交易次數: {total_transactions/unique_properties:.2f}")

# 計算重複交易統計
property_counts = transaction_df['物件唯一ID'].value_counts()
duplicate_properties = property_counts[property_counts > 1]

print(f"\n重複交易統計:")
print(f"單次交易物件: {len(property_counts[property_counts == 1]):,} 個 ({len(property_counts[property_counts == 1])/len(property_counts)*100:.1f}%)")
print(f"重複交易物件: {len(duplicate_properties):,} 個 ({len(duplicate_properties)/len(property_counts)*100:.1f}%)")
print(f"重複交易筆數: {(property_counts - 1).sum():,} 筆")

# 重複交易次數分布
print(f"\n重複交易次數分布:")
repeat_distribution = duplicate_properties.value_counts().sort_index()
for count, frequency in repeat_distribution.items():
    print(f"   {count}次交易: {frequency} 個物件")

# %% [markdown]
# ## 3. 重複交易識別與分組

# %%
# 識別並分析重複交易案例
print("🔍 重複交易案例分析")
print("=" * 60)

if len(duplicate_properties) > 0:
    # 取得重複交易的詳細資訊
    duplicate_transaction_details = []
    
    for property_id, count in duplicate_properties.head(20).items():  # 分析前20個重複案例
        property_transactions = transaction_df[transaction_df['物件唯一ID'] == property_id].copy()
        
        # 按交易日期排序
        property_transactions = property_transactions.sort_values('交易日期')
        
        duplicate_info = {
            'property_id': property_id,
            'transaction_count': count,
            'transactions': property_transactions[['交易日期', '交易年季', '交易總價', '建物單價', '解約情形']].to_dict('records'),
            'price_range': property_transactions['交易總價'].max() - property_transactions['交易總價'].min(),
            'has_cancellation': property_transactions['解約情形'].notna().any(),
            'date_range': (property_transactions['交易日期'].max(), property_transactions['交易日期'].min())
        }
        
        duplicate_transaction_details.append(duplicate_info)
    
    print(f"重複交易案例樣本 (前10個):")
    print("-" * 80)
    
    for i, detail in enumerate(duplicate_transaction_details[:10]):
        print(f"\n案例 {i+1}: {detail['property_id']}")
        print(f"   交易次數: {detail['transaction_count']}")
        print(f"   價格變動範圍: {detail['price_range']:,.0f} 萬元")
        print(f"   是否有解約: {'是' if detail['has_cancellation'] else '否'}")
        
        # 顯示各次交易詳情
        for j, transaction in enumerate(detail['transactions']):
            cancellation_status = "解約" if pd.notna(transaction['解約情形']) else "正常"
            print(f"   交易 {j+1}: {transaction['交易日期']} | {transaction['交易總價']:.0f}萬 | {transaction['建物單價']:.1f}萬/坪 | {cancellation_status}")

else:
    print("❌ 無重複交易案例")
    duplicate_transaction_details = []

# %%
# 重複交易模式分析
print(f"\n📊 重複交易模式分析")
print("-" * 50)

if len(duplicate_properties) > 0:
    # 分析重複交易的特徵
    duplicate_df = transaction_df[transaction_df['物件唯一ID'].isin(duplicate_properties.index)].copy()
    
    print(f"重複交易物件涉及交易: {len(duplicate_df)} 筆")
    
    # 1. 重複交易的縣市分布
    duplicate_city_dist = duplicate_df['縣市'].value_counts()
    print(f"\n重複交易縣市分布:")
    for city, count in duplicate_city_dist.head(10).items():
        total_city_transactions = transaction_df[transaction_df['縣市'] == city].shape[0]
        percentage = count / total_city_transactions * 100
        print(f"   {city}: {count} 筆 ({percentage:.1f}%)")
    
    # 2. 重複交易的價格特徵
    print(f"\n重複交易價格特徵:")
    print(f"   平均交易總價: {duplicate_df['交易總價'].mean():,.0f} 萬元")
    print(f"   平均建物單價: {duplicate_df['建物單價'].mean():.1f} 萬/坪")
    
    # 與整體市場比較
    overall_avg_price = transaction_df['交易總價'].mean()
    overall_avg_unit_price = transaction_df['建物單價'].mean()
    
    print(f"   vs 整體市場總價: {overall_avg_price:,.0f} 萬元 (差異: {(duplicate_df['交易總價'].mean() - overall_avg_price):+,.0f})")
    print(f"   vs 整體市場單價: {overall_avg_unit_price:.1f} 萬/坪 (差異: {(duplicate_df['建物單價'].mean() - overall_avg_unit_price):+.1f})")
    
    # 3. 重複交易的解約情況
    duplicate_cancellation_rate = duplicate_df['解約情形'].notna().sum() / len(duplicate_df) * 100
    overall_cancellation_rate = transaction_df['解約情形'].notna().sum() / len(transaction_df) * 100
    
    print(f"\n重複交易解約情況:")
    print(f"   重複交易解約率: {duplicate_cancellation_rate:.2f}%")
    print(f"   整體市場解約率: {overall_cancellation_rate:.2f}%")
    print(f"   差異: {duplicate_cancellation_rate - overall_cancellation_rate:+.2f}%")

# %% [markdown]
# ## 4. 有效交易判斷規則實作

# %%
# 實作有效交易判斷邏輯
print("⚖️ 有效交易判斷規則實作")
print("=" * 60)

def determine_valid_transaction(property_transactions):
    """
    根據PRD規格判斷有效交易
    
    判斷邏輯：
    1. 優先選擇正常交易（非解約）
    2. 如有多筆正常交易，選擇最早的交易
    3. 如全部解約，選擇最早的解約交易但標記為無效
    
    Args:
        property_transactions (pd.DataFrame): 同一物件的所有交易記錄
        
    Returns:
        dict: 包含有效交易資訊和判斷結果
    """
    # 按交易日期排序
    sorted_transactions = property_transactions.sort_values('交易日期').copy()
    
    # 區分正常交易和解約交易
    normal_transactions = sorted_transactions[sorted_transactions['解約情形'].isna()]
    cancelled_transactions = sorted_transactions[sorted_transactions['解約情形'].notna()]
    
    result = {
        'total_transactions': len(sorted_transactions),
        'normal_count': len(normal_transactions),
        'cancelled_count': len(cancelled_transactions),
        'valid_transaction': None,
        'is_valid': False,
        'selection_reason': '',
        'duplicate_count': len(sorted_transactions) - 1  # 重複次數
    }
    
    if len(normal_transactions) > 0:
        # 選擇最早的正常交易
        valid_transaction = normal_transactions.iloc[0]
        result['valid_transaction'] = valid_transaction
        result['is_valid'] = True
        result['selection_reason'] = f'最早正常交易 (共{len(normal_transactions)}筆正常交易)'
        
    elif len(cancelled_transactions) > 0:
        # 全部解約，選擇最早的解約交易但標記為無效
        valid_transaction = cancelled_transactions.iloc[0]
        result['valid_transaction'] = valid_transaction
        result['is_valid'] = False
        result['selection_reason'] = f'全部解約，選擇最早解約 (共{len(cancelled_transactions)}筆解約)'
        
    else:
        # 理論上不應該發生
        result['selection_reason'] = '無有效交易記錄'
    
    return result

# %%
# 應用有效交易判斷邏輯
print("🔄 對所有重複交易物件進行有效交易判斷...")

if len(duplicate_properties) > 0:
    # 對每個重複交易物件進行有效交易判斷
    valid_transaction_results = []
    
    for property_id in duplicate_properties.index:
        property_transactions = transaction_df[transaction_df['物件唯一ID'] == property_id]
        result = determine_valid_transaction(property_transactions)
        result['property_id'] = property_id
        valid_transaction_results.append(result)
    
    # 轉換為DataFrame以便分析
    valid_results_df = pd.DataFrame(valid_transaction_results)
    
    print(f"✅ 完成 {len(valid_transaction_results)} 個重複交易物件的有效交易判斷")
    
    # 有效交易判斷結果統計
    print(f"\n有效交易判斷結果統計:")
    valid_count = valid_results_df['is_valid'].sum()
    invalid_count = len(valid_results_df) - valid_count
    
    print(f"   有效交易物件: {valid_count} 個 ({valid_count/len(valid_results_df)*100:.1f}%)")
    print(f"   無效交易物件: {invalid_count} 個 ({invalid_count/len(valid_results_df)*100:.1f}%)")
    
    # 選擇原因統計
    reason_counts = valid_results_df['selection_reason'].value_counts()
    print(f"\n選擇原因分布:")
    for reason, count in reason_counts.items():
        print(f"   {reason}: {count} 個物件")
    
    # 重複交易數量統計
    duplicate_count_stats = valid_results_df['duplicate_count'].describe()
    print(f"\n重複交易數量統計:")
    print(f"   平均重複次數: {duplicate_count_stats['mean']:.1f}")
    print(f"   最多重複次數: {duplicate_count_stats['max']:.0f}")
    print(f"   總重複交易筆數: {valid_results_df['duplicate_count'].sum()}")

else:
    print("❌ 無重複交易物件需要處理")
    valid_results_df = pd.DataFrame()

# %%
# 詳細展示有效交易判斷案例
print(f"\n🔍 有效交易判斷案例展示 (前5個)")
print("-" * 80)

if not valid_results_df.empty:
    for i, (idx, result) in enumerate(valid_results_df.head().iterrows()):
        print(f"\n案例 {i+1}: {result['property_id']}")
        print(f"   總交易數: {result['total_transactions']}")
        print(f"   正常交易: {result['normal_count']} 筆")
        print(f"   解約交易: {result['cancelled_count']} 筆")
        print(f"   判斷結果: {'✅ 有效' if result['is_valid'] else '❌ 無效'}")
        print(f"   選擇原因: {result['selection_reason']}")
        print(f"   重複次數: {result['duplicate_count']}")
        
        if result['valid_transaction'] is not None:
            valid_tx = result['valid_transaction']
            print(f"   選中交易: {valid_tx['交易日期']} | {valid_tx['交易總價']:.0f}萬 | {valid_tx['建物單價']:.1f}萬/坪")

# %% [markdown]
# ## 5. 去重處理結果生成

# %%
# 生成去重處理後的乾淨資料集
print("🧹 生成去重處理後的資料集")
print("=" * 60)

# 創建去重標記
transaction_df['是否重複交易'] = transaction_df['物件唯一ID'].isin(duplicate_properties.index)
transaction_df['是否有效交易'] = True  # 預設為有效
transaction_df['無效原因'] = ''
transaction_df['重複交易次數'] = transaction_df['物件唯一ID'].map(property_counts)

# 標記無效的重複交易
if not valid_results_df.empty:
    # 建立有效交易索引映射
    valid_transaction_map = {}
    
    for _, result in valid_results_df.iterrows():
        property_id = result['property_id']
        
        if result['valid_transaction'] is not None:
            # 找到對應的交易記錄索引
            property_transactions = transaction_df[transaction_df['物件唯一ID'] == property_id]
            valid_tx = result['valid_transaction']
            
            # 找到最匹配的交易記錄（通過多個欄位比對）
            matching_transactions = property_transactions[
                (property_transactions['交易日期'] == valid_tx['交易日期']) &
                (property_transactions['交易總價'] == valid_tx['交易總價']) &
                (property_transactions['建物單價'] == valid_tx['建物單價'])
            ]
            
            if len(matching_transactions) > 0:
                valid_transaction_map[property_id] = matching_transactions.index[0]
                
                # 標記該物件的其他交易為無效
                other_transactions = property_transactions[property_transactions.index != matching_transactions.index[0]]
                transaction_df.loc[other_transactions.index, '是否有效交易'] = False
                transaction_df.loc[other_transactions.index, '無效原因'] = '重複交易-非最早有效交易'
                
                # 如果選中的交易本身無效（全部解約情況）
                if not result['is_valid']:
                    transaction_df.loc[matching_transactions.index[0], '是否有效交易'] = False
                    transaction_df.loc[matching_transactions.index[0], '無效原因'] = '全部解約'

# 統計去重處理結果
total_before = len(transaction_df)
valid_after = transaction_df['是否有效交易'].sum()
removed_count = total_before - valid_after

print(f"去重處理結果統計:")
print(f"   處理前總筆數: {total_before:,}")
print(f"   處理後有效筆數: {valid_after:,}")
print(f"   移除重複筆數: {removed_count:,}")
print(f"   資料保留率: {valid_after/total_before*100:.2f}%")

# 無效原因統計
invalid_reasons = transaction_df[~transaction_df['是否有效交易']]['無效原因'].value_counts()
print(f"\n無效交易原因分布:")
for reason, count in invalid_reasons.items():
    print(f"   {reason}: {count:,} 筆")

# %%
# 創建乾淨的資料集
clean_transaction_df = transaction_df[transaction_df['是否有效交易']].copy()

print(f"\n📊 乾淨資料集特性:")
print(f"   有效交易筆數: {len(clean_transaction_df):,}")
print(f"   唯一物件數量: {clean_transaction_df['物件唯一ID'].nunique():,}")
print(f"   平均每物件交易次數: {len(clean_transaction_df)/clean_transaction_df['物件唯一ID'].nunique():.3f}")

# 比較去重前後的基本統計
print(f"\n📈 去重前後統計比較:")

comparison_stats = pd.DataFrame({
    '去重前': [
        transaction_df['交易總價'].mean(),
        transaction_df['建物單價'].mean(),
        transaction_df['總面積_數值'].mean(),
        transaction_df['解約情形'].notna().sum(),
        transaction_df['解約情形'].notna().sum() / len(transaction_df) * 100
    ],
    '去重後': [
        clean_transaction_df['交易總價'].mean(),
        clean_transaction_df['建物單價'].mean(),
        clean_transaction_df['總面積_數值'].mean(),
        clean_transaction_df['解約情形'].notna().sum(),
        clean_transaction_df['解約情形'].notna().sum() / len(clean_transaction_df) * 100
    ]
}, index=['平均交易總價(萬)', '平均建物單價(萬/坪)', '平均總面積(坪)', '解約筆數', '解約率(%)'])

comparison_stats['差異'] = comparison_stats['去重後'] - comparison_stats['去重前']
comparison_stats['差異率(%)'] = (comparison_stats['差異'] / comparison_stats['去重前'] * 100).round(2)

print(comparison_stats.round(2))

# %% [markdown]
# ## 6. 重複交易影響分析

# %%
# 分析重複交易對市場指標的影響
print("📊 重複交易影響分析")
print("=" * 60)

# 1. 對價格統計的影響
print("1️⃣ 對價格統計的影響:")

price_impact = {
    '總交易數量變化': len(clean_transaction_df) - len(transaction_df),
    '平均總價變化': clean_transaction_df['交易總價'].mean() - transaction_df['交易總價'].mean(),
    '平均單價變化': clean_transaction_df['建物單價'].mean() - transaction_df['建物單價'].mean(),
    '總價標準差變化': clean_transaction_df['交易總價'].std() - transaction_df['交易總價'].std(),
    '單價標準差變化': clean_transaction_df['建物單價'].std() - transaction_df['建物單價'].std(),
}

for indicator, change in price_impact.items():
    print(f"   {indicator}: {change:+.2f}")

# %%
# 2. 對縣市統計的影響
print("\n2️⃣ 對主要縣市統計的影響:")

# 計算各縣市去重前後的交易量變化
city_impact = {}
for city in transaction_df['縣市'].value_counts().head(5).index:
    before_count = len(transaction_df[transaction_df['縣市'] == city])
    after_count = len(clean_transaction_df[clean_transaction_df['縣市'] == city])
    
    city_impact[city] = {
        'before': before_count,
        'after': after_count,
        'removed': before_count - after_count,
        'removal_rate': (before_count - after_count) / before_count * 100
    }

for city, stats in city_impact.items():
    print(f"   {city}: {stats['before']} → {stats['after']} (-{stats['removed']}, -{stats['removal_rate']:.2f}%)")

# %%
# 3. 對解約統計的影響
print("\n3️⃣ 對解約統計的影響:")

cancellation_impact = {
    '去重前解約筆數': transaction_df['解約情形'].notna().sum(),
    '去重後解約筆數': clean_transaction_df['解約情形'].notna().sum(),
    '去重前解約率': transaction_df['解約情形'].notna().sum() / len(transaction_df) * 100,
    '去重後解約率': clean_transaction_df['解約情形'].notna().sum() / len(clean_transaction_df) * 100,
}

cancellation_impact['解約筆數變化'] = cancellation_impact['去重後解約筆數'] - cancellation_impact['去重前解約筆數']
cancellation_impact['解約率變化'] = cancellation_impact['去重後解約率'] - cancellation_impact['去重前解約率']

for indicator, value in cancellation_impact.items():
    if '率' in indicator:
        print(f"   {indicator}: {value:.3f}%")
    else:
        print(f"   {indicator}: {value}")

# %% [markdown]
# ## 7. 資料品質評估

# %%
# 資料品質評估
print("🔍 資料品質評估")
print("=" * 50)

# 1. 重複交易特徵分析
print("1️⃣ 重複交易特徵分析:")

if len(duplicate_properties) > 0:
    # 重複交易的物件特徵
    duplicate_transactions = transaction_df[transaction_df['是否重複交易']]
    
    print(f"   重複交易物件數量: {duplicate_properties.nunique()} 個")
    print(f"   重複交易總筆數: {len(duplicate_transactions)} 筆")
    print(f"   平均每個重複物件交易次數: {len(duplicate_transactions) / duplicate_properties.nunique():.2f}")
    
    # 重複交易的時間分布
    repeat_by_season = duplicate_transactions['交易年季'].value_counts().sort_index()
    print(f"\n   重複交易年季分布 (前5名):")
    for season, count in repeat_by_season.head().items():
        total_season = transaction_df[transaction_df['交易年季'] == season].shape[0]
        percentage = count / total_season * 100 if total_season > 0 else 0
        print(f"      {season}: {count} 筆 ({percentage:.1f}%)")

# %%
# 2. 資料完整性檢查
print("\n2️⃣ 資料完整性檢查:")

# 檢查關鍵欄位的完整性
key_fields = ['備查編號', '坐落街道', '樓層', '交易日期', '交易總價', '建物單價']

print("   關鍵欄位完整性 (去重後):")
for field in key_fields:
    if field in clean_transaction_df.columns:
        missing_count = clean_transaction_df[field].isna().sum()
        missing_rate = missing_count / len(clean_transaction_df) * 100
        print(f"      {field}: {len(clean_transaction_df) - missing_count}/{len(clean_transaction_df)} ({100-missing_rate:.1f}% 完整)")
    else:
        print(f"      {field}: 欄位不存在")

# %%
# 3. 異常值檢測
print("\n3️⃣ 異常值檢測:")

# 檢測價格異常值
def detect_price_outliers(df, column, method='iqr'):
    """檢測價格異常值"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

# 檢測總價異常值
total_price_outliers, tp_lower, tp_upper = detect_price_outliers(clean_transaction_df, '交易總價')
print(f"   交易總價異常值: {len(total_price_outliers)} 筆 ({len(total_price_outliers)/len(clean_transaction_df)*100:.2f}%)")
print(f"      正常範圍: {tp_lower:.0f} - {tp_upper:.0f} 萬元")

# 檢測單價異常值
unit_price_outliers, up_lower, up_upper = detect_price_outliers(clean_transaction_df, '建物單價')
print(f"   建物單價異常值: {len(unit_price_outliers)} 筆 ({len(unit_price_outliers)/len(clean_transaction_df)*100:.2f}%)")
print(f"      正常範圍: {up_lower:.1f} - {up_upper:.1f} 萬/坪")

# %%
# 4. 邏輯一致性檢查
print("\n4️⃣ 邏輯一致性檢查:")

consistency_issues = {}

# 檢查總價與單價、面積的一致性
clean_transaction_df['計算總價'] = clean_transaction_df['建物單價'] * clean_transaction_df['總面積_數值']
clean_transaction_df['價格差異'] = abs(clean_transaction_df['交易總價'] - clean_transaction_df['計算總價'])
clean_transaction_df['價格差異率'] = clean_transaction_df['價格差異'] / clean_transaction_df['交易總價'] * 100

# 設定容忍誤差為5%
price_inconsistent = clean_transaction_df[clean_transaction_df['價格差異率'] > 5]
consistency_issues['價格計算不一致'] = len(price_inconsistent)

# 檢查面積合理性
area_unreasonable = clean_transaction_df[
    (clean_transaction_df['總面積_數值'] < 5) | 
    (clean_transaction_df['總面積_數值'] > 200)
]
consistency_issues['面積不合理'] = len(area_unreasonable)

# 檢查單價合理性
unit_price_unreasonable = clean_transaction_df[
    (clean_transaction_df['建物單價'] < 5) | 
    (clean_transaction_df['建物單價'] > 300)
]
consistency_issues['單價不合理'] = len(unit_price_unreasonable)

print("   邏輯一致性問題統計:")
for issue, count in consistency_issues.items():
    percentage = count / len(clean_transaction_df) * 100
    print(f"      {issue}: {count} 筆 ({percentage:.2f}%)")

# %% [markdown]
# ## 8. 視覺化分析

# %%
# 創建視覺化圖表
print("📊 重複交易處理視覺化分析")
print("=" * 50)

# 創建圖表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 重複交易次數分布
if len(duplicate_properties) > 0:
    repeat_counts = duplicate_properties.value_counts().value_counts().sort_index()
    
    bars = axes[0, 0].bar(repeat_counts.index, repeat_counts.values, color='skyblue')
    axes[0, 0].set_title('重複交易次數分布', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('交易次數')
    axes[0, 0].set_ylabel('物件數量')
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
else:
    axes[0, 0].text(0.5, 0.5, '無重複交易', ha='center', va='center', 
                   transform=axes[0, 0].transAxes, fontsize=12)
    axes[0, 0].set_title('重複交易次數分布 (無資料)', fontsize=14)

# 2. 去重前後縣市交易量比較
top_cities = list(city_impact.keys())
before_counts = [city_impact[city]['before'] for city in top_cities]
after_counts = [city_impact[city]['after'] for city in top_cities]

x = np.arange(len(top_cities))
width = 0.35

bars1 = axes[0, 1].bar(x - width/2, before_counts, width, label='去重前', color='lightcoral')
bars2 = axes[0, 1].bar(x + width/2, after_counts, width, label='去重後', color='lightgreen')

axes[0, 1].set_title('主要縣市去重前後交易量比較', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('縣市')
axes[0, 1].set_ylabel('交易筆數')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(top_cities, rotation=45, ha='right')
axes[0, 1].legend()

# 3. 價格影響分析
price_metrics = ['平均總價', '平均單價']
before_prices = [transaction_df['交易總價'].mean(), transaction_df['建物單價'].mean()]
after_prices = [clean_transaction_df['交易總價'].mean(), clean_transaction_df['建物單價'].mean()]

x = np.arange(len(price_metrics))
bars1 = axes[1, 0].bar(x - width/2, before_prices, width, label='去重前', color='orange')
bars2 = axes[1, 0].bar(x + width/2, after_prices, width, label='去重後', color='blue')

axes[1, 0].set_title('去重前後價格統計比較', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('價格指標')
axes[1, 0].set_ylabel('價格')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(price_metrics)
axes[1, 0].legend()

# 添加數值標籤
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    axes[1, 0].text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                   f'{before_prices[i]:.0f}', ha='center', va='bottom')
    axes[1, 0].text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                   f'{after_prices[i]:.0f}', ha='center', va='bottom')

# 4. 資料品質問題分布
if consistency_issues:
    issue_names = list(consistency_issues.keys())
    issue_counts = list(consistency_issues.values())
    
    bars = axes[1, 1].bar(range(len(issue_names)), issue_counts, color='red', alpha=0.7)
    axes[1, 1].set_title('資料品質問題分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('問題類型')
    axes[1, 1].set_ylabel('問題筆數')
    axes[1, 1].set_xticks(range(len(issue_names)))
    axes[1, 1].set_xticklabels(issue_names, rotation=45, ha='right')
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
else:
    axes[1, 1].text(0.5, 0.5, '無品質問題', ha='center', va='center', 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('資料品質問題分布 (無問題)', fontsize=14)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. 結果儲存與驗證

# %%
# 儲存處理結果
print("💾 儲存重複交易處理結果...")

# 1. 儲存完整的交易資料（包含去重標記）
enhanced_transaction_df = transaction_df[[
    '備查編號', '縣市', '行政區', '坐落街道', '樓層', '交易日期', '交易年季',
    '交易總價', '建物單價', '總面積_數值', '解約情形', '物件唯一ID',
    '是否重複交易', '是否有效交易', '無效原因', '重複交易次數'
]].copy()

enhanced_transaction_df.to_csv('../data/processed/03_enhanced_transactions.csv', 
                              index=False, encoding='utf-8-sig')
print("✅ 完整交易資料已儲存至: ../data/processed/03_enhanced_transactions.csv")

# 2. 儲存乾淨的資料集（僅有效交易）
clean_transaction_df.to_csv('../data/processed/03_clean_transactions.csv', 
                           index=False, encoding='utf-8-sig')
print("✅ 乾淨交易資料已儲存至: ../data/processed/03_clean_transactions.csv")

# 3. 儲存重複交易分析結果
if not valid_results_df.empty:
    duplicate_analysis_summary = valid_results_df[[
        'property_id', 'total_transactions', 'normal_count', 'cancelled_count',
        'is_valid', 'selection_reason', 'duplicate_count'
    ]].copy()
    
    duplicate_analysis_summary.to_csv('../data/processed/03_duplicate_analysis.csv', 
                                     index=False, encoding='utf-8-sig')
    print("✅ 重複交易分析結果已儲存至: ../data/processed/03_duplicate_analysis.csv")

# %%
# 生成處理總結報告
processing_summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'original_transactions': len(transaction_df),
    'unique_properties': transaction_df['物件唯一ID'].nunique(),
    'duplicate_properties': len(duplicate_properties) if len(duplicate_properties) > 0 else 0,
    'duplicate_transactions': len(transaction_df[transaction_df['是否重複交易']]),
    'valid_transactions': transaction_df['是否有效交易'].sum(),
    'invalid_transactions': (~transaction_df['是否有效交易']).sum(),
    'data_retention_rate': transaction_df['是否有效交易'].sum() / len(transaction_df) * 100,
    'avg_price_change': clean_transaction_df['交易總價'].mean() - transaction_df['交易總價'].mean(),
    'avg_unit_price_change': clean_transaction_df['建物單價'].mean() - transaction_df['建物單價'].mean(),
    'cancellation_rate_before': transaction_df['解約情形'].notna().sum() / len(transaction_df) * 100,
    'cancellation_rate_after': clean_transaction_df['解約情形'].notna().sum() / len(clean_transaction_df) * 100
}

# 轉換為DataFrame並儲存
summary_df = pd.DataFrame([processing_summary])
summary_df.to_csv('../data/processed/03_processing_summary.csv', 
                 index=False, encoding='utf-8-sig')
print("✅ 處理總結已儲存至: ../data/processed/03_processing_summary.csv")

# %%
# 驗證處理結果
print("\n🔍 處理結果驗證")
print("=" * 50)

# 1. 基本數量驗證
print("1️⃣ 基本數量驗證:")
print(f"   原始交易筆數: {len(transaction_df):,}")
print(f"   有效交易筆數: {len(clean_transaction_df):,}")
print(f"   移除筆數: {len(transaction_df) - len(clean_transaction_df):,}")
print(f"   資料保留率: {len(clean_transaction_df)/len(transaction_df)*100:.2f}%")

# 2. 唯一性驗證
print(f"\n2️⃣ 唯一性驗證:")
remaining_duplicates = clean_transaction_df['物件唯一ID'].value_counts()
remaining_duplicate_count = (remaining_duplicates > 1).sum()
print(f"   處理後剩餘重複物件: {remaining_duplicate_count} 個")

if remaining_duplicate_count > 0:
    print("   ⚠️ 仍存在重複交易，需要進一步檢查")
    print("   剩餘重複案例:")
    for property_id, count in remaining_duplicates[remaining_duplicates > 1].head().items():
        print(f"      {property_id}: {count} 筆")
else:
    print("   ✅ 所有重複交易已正確處理")

# 3. 解約率影響驗證
print(f"\n3️⃣ 解約率影響驗證:")
print(f"   處理前解約率: {transaction_df['解約情形'].notna().sum() / len(transaction_df) * 100:.3f}%")
print(f"   處理後解約率: {clean_transaction_df['解約情形'].notna().sum() / len(clean_transaction_df) * 100:.3f}%")
print(f"   解約率變化: {(clean_transaction_df['解約情形'].notna().sum() / len(clean_transaction_df) - transaction_df['解約情形'].notna().sum() / len(transaction_df)) * 100:+.3f}%")

# %% [markdown]
# ## 10. 分析總結與建議

# %%
# 重複交易處理分析總結
print("📋 重複交易處理分析總結")
print("=" * 80)

print("1️⃣ 處理成果:")
print(f"   ✅ 成功識別 {len(duplicate_properties) if len(duplicate_properties) > 0 else 0:,} 個重複交易物件")
print(f"   ✅ 處理 {len(transaction_df) - len(clean_transaction_df):,} 筆重複交易")
print(f"   ✅ 資料保留率達 {len(clean_transaction_df)/len(transaction_df)*100:.2f}%")
print(f"   ✅ 去重後唯一物件比例: {clean_transaction_df['物件唯一ID'].nunique()/len(clean_transaction_df)*100:.2f}%")

print(f"\n2️⃣ 品質提升:")
if len(duplicate_properties) > 0:
    avg_duplicates_per_property = len(transaction_df[transaction_df['是否重複交易']]) / len(duplicate_properties)
    print(f"   📊 平均每個重複物件減少 {avg_duplicates_per_property-1:.1f} 筆冗餘交易")

price_change_pct = (clean_transaction_df['交易總價'].mean() - transaction_df['交易總價'].mean()) / transaction_df['交易總價'].mean() * 100
unit_price_change_pct = (clean_transaction_df['建物單價'].mean() - transaction_df['建物單價'].mean()) / transaction_df['建物單價'].mean() * 100

print(f"   💰 平均交易總價變化: {price_change_pct:+.2f}%")
print(f"   🏠 平均建物單價變化: {unit_price_change_pct:+.2f}%")

print(f"\n3️⃣ 主要發現:")
if 'city_impact' in locals() and city_impact:
    highest_removal_city = max(city_impact.items(), key=lambda x: x[1]['removal_rate'])
    print(f"   🗺️ 重複交易比例最高縣市: {highest_removal_city[0]} ({highest_removal_city[1]['removal_rate']:.2f}%)")

if not valid_results_df.empty:
    all_cancelled_count = len(valid_results_df[~valid_results_df['is_valid']])
    print(f"   ⚠️ 全部解約物件: {all_cancelled_count} 個")

print(f"\n4️⃣ 後續建議:")
print("   📝 建議定期執行重複交易檢查機制")
print("   🔍 建立自動化異常值檢測流程")
print("   📊 持續監控資料品質指標")

if remaining_duplicate_count > 0:
    print("   ⚠️ 建議進一步調查剩餘重複案例的原因")

# 品質問題建議
if consistency_issues:
    total_issues = sum(consistency_issues.values())
    issue_rate = total_issues / len(clean_transaction_df) * 100
    if issue_rate > 5:
        print(f"   🚨 資料品質問題比例較高 ({issue_rate:.1f}%)，建議加強資料驗證")

print(f"\n5️⃣ 下一步工作:")
print("   🎯 進行社區級去化率計算 (Notebook 4)")
print("   📈 建立三層級市場分析架構")
print("   🚨 實作風險評估與預警機制")

# %% [markdown]
# ## 11. 下一步工作重點
# 
# ### ✅ 已完成項目:
# 1. 物件唯一ID建立邏輯實作
# 2. 重複交易識別與分組機制
# 3. 有效交易判斷規則實作
# 4. 去重處理結果驗證
# 5. 重複交易模式與影響分析
# 6. 資料品質評估與改善
# 
# ### 🔄 待進行項目:
# 1. **Notebook 4**: 社區級去化率分析
#    - 建案基本資訊匹配
#    - 去化率計算（毛/淨/調整）
#    - 去化動態分析（速度/加速度）
# 
# 2. **Notebook 5**: 行政區級聚合分析
#    - 區域解約風險聚合
#    - 區域去化效率排名
#    - 區域風險等級評估
# 
# ### 🎯 關鍵成果:
# 1. 資料保留率 {len(clean_transaction_df)/len(transaction_df)*100:.2f}% 符合品質要求
# 2. 重複交易處理邏輯運作正常
# 3. 為後續三層級分析建立了可靠的資料基礎
# 4. 資料品質問題已識別並標記

print("\n" + "="*80)
print("🎉 Notebook 3 - 重複交易識別與處理完成！")
print("📝 請繼續執行 Notebook 4 進行社區級去化率分析")
print("="*80)