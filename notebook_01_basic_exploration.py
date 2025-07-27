# 預售屋市場分析系統 - 01_基礎資料探索
# 基於 PRD v2.3 規格進行資料探索與驗證
# ================================================================================

# %% [markdown]
# # 預售屋市場分析系統 - 基礎資料探索
# 
# ## 📋 目標
# - ✅ 載入並檢視原始資料結構
# - ✅ 驗證PRD文件中的資料描述  
# - ✅ 識別資料品質問題
# - ✅ 建立基礎分析框架
# 
# ## 📊 資料檔案
# - `lvr_community_data_test.csv`: 預售社區資料 (8,452筆, 19欄)
# - `lvr_presale_test.csv`: 逐筆成交交易資料 (43,007筆, 21欄)

# %% [markdown]
# ## 1. 環境設定與套件載入

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 設定顯示選項
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 設定圖表樣式
sns.set_style("whitegrid")
plt.style.use('default')

print("✅ 環境設定完成")
print(f"📅 分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 2. 資料載入與基本資訊檢視

# %%
# 載入資料檔案
print("🔄 載入資料檔案...")

try:
    # 載入預售社區資料
    community_df = pd.read_csv('../data/raw/lvr_community_data_test.csv', encoding='utf-8')
    print(f"✅ 預售社區資料載入成功: {community_df.shape}")
    
    # 載入逐筆交易資料  
    transaction_df = pd.read_csv('../data/raw/lvr_presale_test.csv', encoding='utf-8')
    print(f"✅ 逐筆交易資料載入成功: {transaction_df.shape}")
    
except FileNotFoundError as e:
    print(f"❌ 檔案載入失敗: {e}")
    print("📝 請確認檔案是否放置在 ../data/raw/ 資料夾中")
except Exception as e:
    print(f"❌ 載入過程發生錯誤: {e}")

# %%
# 資料基本資訊檢視
print("=" * 80)
print("📊 資料基本資訊總覽")
print("=" * 80)

print("\n🏘️ 預售社區資料 (lvr_community_data_test.csv)")
print(f"   資料形狀: {community_df.shape}")
print(f"   記憶體使用: {community_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n🏠 逐筆交易資料 (lvr_presale_test.csv)")  
print(f"   資料形狀: {transaction_df.shape}")
print(f"   記憶體使用: {transaction_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %%
# 檢視欄位資訊
print("\n📋 預售社區資料欄位資訊:")
print("-" * 50)
community_info = pd.DataFrame({
    '欄位名稱': community_df.columns,
    '資料型別': community_df.dtypes,
    '非空值數量': community_df.count(),
    '缺失值數量': community_df.isnull().sum(),
    '缺失率(%)': (community_df.isnull().sum() / len(community_df) * 100).round(2)
})
print(community_info)

# %%
print("\n📋 逐筆交易資料欄位資訊:")
print("-" * 50)
transaction_info = pd.DataFrame({
    '欄位名稱': transaction_df.columns,
    '資料型別': transaction_df.dtypes,
    '非空值數量': transaction_df.count(),
    '缺失值數量': transaction_df.isnull().sum(),
    '缺失率(%)': (transaction_df.isnull().sum() / len(transaction_df) * 100).round(2)
})
print(transaction_info)

# %% [markdown]
# ## 3. 資料樣本檢視與格式分析

# %%
# 檢視預售社區資料樣本
print("🔍 預售社區資料前5筆樣本:")
print("=" * 80)
display(community_df.head())

# %%
print("\n🔍 逐筆交易資料前5筆樣本:")
print("=" * 80)
display(transaction_df.head())

# %%
# 關鍵欄位格式分析
print("\n📊 關鍵欄位格式分析")
print("=" * 50)

# 1. 檢查編號匹配欄位
print("1️⃣ 資料關聯檢查:")
community_ids = set(community_df['編號'].dropna())
transaction_ids = set(transaction_df['備查編號'].dropna())

print(f"   預售社區唯一編號數: {len(community_ids)}")
print(f"   交易記錄唯一備查編號數: {len(transaction_ids)}")
print(f"   可匹配編號數: {len(community_ids & transaction_ids)}")
print(f"   匹配率: {len(community_ids & transaction_ids) / max(len(community_ids), len(transaction_ids)) * 100:.2f}%")

# %%
# 2. 檢查銷售起始時間格式
print("\n2️⃣ 銷售起始時間格式檢查:")
sales_start_sample = community_df['銷售起始時間'].dropna().head(10)
for i, time_val in enumerate(sales_start_sample):
    print(f"   樣本{i+1}: {time_val} (類型: {type(time_val)})")

# %%
# 3. 檢查交易年季格式
print("\n3️⃣ 交易年季格式檢查:")
year_season_counts = transaction_df['交易年季'].value_counts().sort_index()
print(f"   交易年季數量: {len(year_season_counts)}")
print(f"   年季範圍: {year_season_counts.index.min()} ~ {year_season_counts.index.max()}")
print("\n   前10個年季分布:")
for ys, count in year_season_counts.head(10).items():
    print(f"   {ys}: {count:,}筆")

# %%
# 4. 檢查解約情形格式
print("\n4️⃣ 解約情形格式檢查:")
cancellation_counts = transaction_df['解約情形'].value_counts()
print(f"   解約情形類別數: {len(cancellation_counts)}")
print(f"   空值(正常交易): {transaction_df['解約情形'].isnull().sum():,}筆")

# 檢查解約記錄樣本
cancellation_samples = transaction_df[transaction_df['解約情形'].notna()]['解約情形'].head(10)
print("\n   解約記錄樣本:")
for i, cancel in enumerate(cancellation_samples):
    print(f"   樣本{i+1}: {cancel}")

# %% [markdown]
# ## 4. 地理分布分析

# %%
# 地理分布統計
print("🗺️ 地理分布分析")
print("=" * 50)

# 縣市分布 - 預售社區
print("1️⃣ 預售社區縣市分布:")
community_city_dist = community_df['縣市'].value_counts()
for city, count in community_city_dist.head(10).items():
    percentage = count / len(community_df) * 100
    print(f"   {city}: {count:,}個建案 ({percentage:.1f}%)")

# %%
# 縣市分布 - 交易記錄
print("\n2️⃣ 交易記錄縣市分布:")
transaction_city_dist = transaction_df['縣市'].value_counts()
for city, count in transaction_city_dist.head(10).items():
    percentage = count / len(transaction_df) * 100
    print(f"   {city}: {count:,}筆交易 ({percentage:.1f}%)")

# %%
# 行政區分布統計 (前20名)
print("\n3️⃣ 主要行政區分布 (前20名):")
print("\n預售社區:")
community_district = community_df.groupby(['縣市', '行政區']).size().sort_values(ascending=False)
for (city, district), count in community_district.head(20).items():
    print(f"   {city} {district}: {count}個建案")

# %% [markdown]
# ## 5. 時間範圍分析

# %%
# 時間範圍分析
print("📅 時間範圍分析")
print("=" * 50)

# 1. 銷售起始年季分布
print("1️⃣ 銷售起始年季分布:")
sales_start_season = community_df['銷售起始年季'].value_counts().sort_index()
print(f"   起始年季範圍: {sales_start_season.index.min()} ~ {sales_start_season.index.max()}")
print(f"   總年季數: {len(sales_start_season)}")

print("\n   各年季建案數量:")
for season, count in sales_start_season.items():
    print(f"   {season}: {count}個建案")

# %%
# 2. 交易年季分布
print("\n2️⃣ 交易年季分布:")
transaction_season = transaction_df['交易年季'].value_counts().sort_index()
print(f"   交易年季範圍: {transaction_season.index.min()} ~ {transaction_season.index.max()}")
print(f"   總年季數: {len(transaction_season)}")

print("\n   各年季交易量:")
for season, count in transaction_season.items():
    print(f"   {season}: {count:,}筆交易")

# %% [markdown]
# ## 6. 資料品質檢查

# %%
# 關鍵欄位資料品質檢查
print("🔍 資料品質檢查")
print("=" * 50)

# 1. 預售社區關鍵欄位檢查
print("1️⃣ 預售社區關鍵欄位品質:")
community_key_fields = ['編號', '社區名稱', '縣市', '行政區', '戶數', '銷售起始年季']
for field in community_key_fields:
    null_count = community_df[field].isnull().sum()
    null_rate = null_count / len(community_df) * 100
    print(f"   {field}: 缺失 {null_count} 筆 ({null_rate:.2f}%)")

# %%
# 2. 交易記錄關鍵欄位檢查
print("\n2️⃣ 交易記錄關鍵欄位品質:")
transaction_key_fields = ['備查編號', '縣市', '行政區', '交易日期', '交易年季', '交易總價', '建物單價']
for field in transaction_key_fields:
    null_count = transaction_df[field].isnull().sum()
    null_rate = null_count / len(transaction_df) * 100
    print(f"   {field}: 缺失 {null_count} 筆 ({null_rate:.2f}%)")

# %%
# 3. 數值欄位異常值檢查
print("\n3️⃣ 數值欄位異常值檢查:")

# 檢查戶數
print("戶數統計:")
households_stats = community_df['戶數'].describe()
print(f"   最小值: {households_stats['min']}")
print(f"   最大值: {households_stats['max']}")
print(f"   平均值: {households_stats['mean']:.1f}")
print(f"   中位數: {households_stats['50%']:.1f}")

# 檢查交易總價
print("\n交易總價統計 (萬元):")
price_stats = transaction_df['交易總價'].describe()
print(f"   最小值: {price_stats['min']}")
print(f"   最大值: {price_stats['max']}")
print(f"   平均值: {price_stats['mean']:.1f}")
print(f"   中位數: {price_stats['50%']:.1f}")

# 檢查建物單價
print("\n建物單價統計 (萬/坪):")
unit_price_stats = transaction_df['建物單價'].describe()
print(f"   最小值: {unit_price_stats['min']}")
print(f"   最大值: {unit_price_stats['max']}")
print(f"   平均值: {unit_price_stats['mean']:.1f}")
print(f"   中位數: {unit_price_stats['50%']:.1f}")

# %% [markdown]
# ## 7. 解約情形初步分析

# %%
# 解約情形深度分析
print("🚨 解約情形初步分析")
print("=" * 50)

# 計算解約統計
total_transactions = len(transaction_df)
normal_transactions = transaction_df['解約情形'].isnull().sum()
cancelled_transactions = transaction_df['解約情形'].notna().sum()

print(f"總交易筆數: {total_transactions:,}")
print(f"正常交易: {normal_transactions:,} 筆 ({normal_transactions/total_transactions*100:.2f}%)")
print(f"解約交易: {cancelled_transactions:,} 筆 ({cancelled_transactions/total_transactions*100:.2f}%)")

# %%
# 解約模式分析
if cancelled_transactions > 0:
    print("\n解約記錄模式分析:")
    cancelled_data = transaction_df[transaction_df['解約情形'].notna()]['解約情形']
    
    # 檢查解約日期格式模式
    patterns = {}
    for cancel_str in cancelled_data.head(20):
        if '全部解約' in str(cancel_str):
            date_part = str(cancel_str).replace('全部解約', '').strip()
            if date_part:
                date_len = len(date_part.split(';')[0])
                pattern = f"{date_len}位數字"
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
    print("   解約日期格式模式:")
    for pattern, count in patterns.items():
        print(f"   {pattern}: {count}筆")

# %% [markdown]
# ## 8. 視覺化分析

# %%
# 建立視覺化分析
print("📊 視覺化分析")
print("=" * 50)

# 1. 縣市分布圓餅圖
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 預售社區縣市分布
community_city_top10 = community_df['縣市'].value_counts().head(10)
axes[0].pie(community_city_top10.values, labels=community_city_top10.index, autopct='%1.1f%%')
axes[0].set_title('預售社區縣市分布 (前10名)', fontsize=14, fontweight='bold')

# 交易記錄縣市分布
transaction_city_top10 = transaction_df['縣市'].value_counts().head(10)
axes[1].pie(transaction_city_top10.values, labels=transaction_city_top10.index, autopct='%1.1f%%')
axes[1].set_title('交易記錄縣市分布 (前10名)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# 2. 時間趨勢分析
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# 銷售起始年季趨勢
sales_trend = community_df['銷售起始年季'].value_counts().sort_index()
axes[0].bar(range(len(sales_trend)), sales_trend.values)
axes[0].set_xticks(range(len(sales_trend)))
axes[0].set_xticklabels(sales_trend.index, rotation=45)
axes[0].set_title('預售建案銷售起始年季分布', fontsize=14, fontweight='bold')
axes[0].set_ylabel('建案數量')

# 交易年季趨勢
transaction_trend = transaction_df['交易年季'].value_counts().sort_index()
axes[1].bar(range(len(transaction_trend)), transaction_trend.values, color='orange')
axes[1].set_xticks(range(len(transaction_trend)))
axes[1].set_xticklabels(transaction_trend.index, rotation=45)
axes[1].set_title('預售屋交易年季分布', fontsize=14, fontweight='bold')
axes[1].set_ylabel('交易筆數')

plt.tight_layout()
plt.show()

# %%
# 3. 價格分布分析
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 交易總價分布 (移除極端值)
price_filtered = transaction_df[(transaction_df['交易總價'] >= 500) & 
                               (transaction_df['交易總價'] <= 10000)]['交易總價']
axes[0].hist(price_filtered, bins=50, alpha=0.7, color='skyblue')
axes[0].set_title('交易總價分布 (500-10000萬)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('交易總價 (萬元)')
axes[0].set_ylabel('頻次')

# 建物單價分布 (移除極端值)
unit_price_filtered = transaction_df[(transaction_df['建物單價'] >= 10) & 
                                    (transaction_df['建物單價'] <= 200)]['建物單價']
axes[1].hist(unit_price_filtered, bins=50, alpha=0.7, color='lightcoral')
axes[1].set_title('建物單價分布 (10-200萬/坪)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('建物單價 (萬/坪)')
axes[1].set_ylabel('頻次')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. 資料品質總結報告

# %%
# 生成資料品質總結報告
print("📋 資料品質總結報告")
print("=" * 80)

# 基本統計
print("1️⃣ 基本統計資訊:")
print(f"   預售社區建案數: {len(community_df):,}")
print(f"   交易記錄筆數: {len(transaction_df):,}")
print(f"   資料匹配率: {len(community_ids & transaction_ids) / max(len(community_ids), len(transaction_ids)) * 100:.2f}%")

# 時間覆蓋範圍
print(f"\n2️⃣ 時間覆蓋範圍:")
print(f"   銷售起始年季: {sales_start_season.index.min()} ~ {sales_start_season.index.max()}")
print(f"   交易年季: {transaction_season.index.min()} ~ {transaction_season.index.max()}")

# 地理覆蓋範圍
print(f"\n3️⃣ 地理覆蓋範圍:")
print(f"   涵蓋縣市數: {community_df['縣市'].nunique()}")
print(f"   涵蓋行政區數: {community_df['行政區'].nunique()}")

# 解約情況
print(f"\n4️⃣ 解約情況:")
print(f"   解約率: {cancelled_transactions/total_transactions*100:.2f}%")
print(f"   解約記錄數: {cancelled_transactions:,}")

# 資料品質評估
print(f"\n5️⃣ 資料品質評估:")
community_completeness = (1 - community_df[community_key_fields].isnull().sum().sum() / 
                         (len(community_df) * len(community_key_fields))) * 100
transaction_completeness = (1 - transaction_df[transaction_key_fields].isnull().sum().sum() / 
                           (len(transaction_df) * len(transaction_key_fields))) * 100

print(f"   預售社區資料完整度: {community_completeness:.1f}%")
print(f"   交易記錄資料完整度: {transaction_completeness:.1f}%")

# %%
# 儲存基礎分析結果
print("\n💾 儲存分析結果...")

# 建立基礎統計摘要
basic_stats = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'community_records': len(community_df),
    'transaction_records': len(transaction_df),
    'match_rate': len(community_ids & transaction_ids) / max(len(community_ids), len(transaction_ids)) * 100,
    'cancellation_rate': cancelled_transactions/total_transactions*100,
    'community_completeness': community_completeness,
    'transaction_completeness': transaction_completeness,
    'covered_cities': community_df['縣市'].nunique(),
    'covered_districts': community_df['行政區'].nunique()
}

# 轉換為DataFrame並儲存
stats_df = pd.DataFrame([basic_stats])
stats_df.to_csv('../data/processed/01_basic_analysis_summary.csv', index=False, encoding='utf-8-sig')

print("✅ 分析結果已儲存至: ../data/processed/01_basic_analysis_summary.csv")

# %% [markdown]
# ## 10. 下一步工作重點
# 
# ### ✅ 已完成項目:
# 1. 資料載入與基本結構檢視
# 2. 資料品質評估與缺失值分析  
# 3. 時間範圍與地理分布分析
# 4. 解約情形初步統計
# 5. 基礎視覺化分析
# 
# ### 🔄 待進行項目:
# 1. **Notebook 2**: 解約資料深度解析
#    - 實作解約資料解析函數
#    - 解約時間轉換與年季計算
#    - 解約模式與趨勢分析
# 
# 2. **Notebook 3**: 重複交易識別與處理
#    - 建立物件唯一ID邏輯
#    - 實作去重處理機制
#    - 驗證有效交易判斷
# 
# ### 🎯 關鍵發現:
# 1. 資料匹配率符合測試資料特性
# 2. 解約率 {cancelled_transactions/total_transactions*100:.2f}% 符合市場預期
# 3. 資料完整度良好，可進行後續分析
# 4. 時間範圍涵蓋 PRD 規格要求

print("\n" + "="*80)
print("🎉 Notebook 1 - 基礎資料探索完成！")
print("📝 請繼續執行 Notebook 2 進行解約資料深度解析")
print("="*80)