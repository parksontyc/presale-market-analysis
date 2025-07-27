# 預售屋市場分析系統 - 05_去化率計算與驗證
# 基於 PRD v2.3 規格進行三種去化率的詳細計算
# ================================================================================

# %% [markdown]
# # 預售屋市場分析系統 - 去化率計算與驗證
# 
# ## 📋 目標
# - ✅ 實作三種去化率計算邏輯
# - ✅ 驗證計算結果合理性
# - ✅ 建立去化率基準值
# - ✅ 時間對齊邏輯處理
# - ✅ 異常案例識別與處理
# - ✅ 為社區級報告奠定基礎
# 
# ## 🎯 內容大綱
# 1. 毛去化率計算實作
# 2. 淨去化率計算實作
# 3. 調整去化率計算實作
# 4. 時間對齊邏輯處理
# 5. 去化率合理性驗證
# 6. 異常案例識別與處理
# 7. 去化率基準值建立
# 8. 計算結果視覺化分析
# 
# ## 📊 延續 Notebook 1-4 的分析結果
# - 乾淨交易資料: 去重後的有效交易記錄
# - 解約分析結果: 解約資料解析與風險評估
# - 建案整合結果: 活躍建案識別與滯銷標記
# - 三種去化率定義: 毛/淨/調整去化率

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
from collections import Counter, defaultdict
import math
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
# 載入前階段處理結果
print("🔄 載入前階段處理結果...")

try:
    # 載入乾淨的交易資料
    clean_transactions = pd.read_csv('../data/processed/03_clean_transactions.csv', encoding='utf-8')
    print(f"✅ 乾淨交易資料: {clean_transactions.shape}")
    
    # 載入建案整合結果
    active_projects = pd.read_csv('../data/processed/04_active_projects_analysis.csv', encoding='utf-8')
    print(f"✅ 活躍建案分析: {active_projects.shape}")
    
    # 載入滯銷分析結果
    stagnant_projects = pd.read_csv('../data/processed/04_stagnant_projects_analysis.csv', encoding='utf-8')
    print(f"✅ 滯銷建案分析: {stagnant_projects.shape}")
    
    # 載入原始建案資料（用於補充資訊）
    project_data = pd.read_csv('../data/raw/lvr_sale_data_test.csv', encoding='utf-8')
    print(f"✅ 建案基本資料: {project_data.shape}")
    
except FileNotFoundError as e:
    print(f"❌ 檔案載入失敗: {e}")
    print("📝 請確認是否已執行 Notebook 1-4")
except Exception as e:
    print(f"❌ 載入過程發生錯誤: {e}")

# %%
# 資料概況檢視
print("📊 去化率計算基礎資料概況")
print("=" * 60)

print("交易資料:")
print(f"   總筆數: {len(clean_transactions):,}")
print(f"   備查編號數: {clean_transactions['備查編號'].nunique():,}")
print(f"   時間範圍: {clean_transactions['交易年季'].min()} ~ {clean_transactions['交易年季'].max()}")

print(f"\n活躍建案:")
print(f"   總建案數: {len(active_projects):,}")
print(f"   活躍建案數: {len(active_projects[active_projects['is_active'] == True]):,}")
print(f"   平均戶數: {active_projects['total_units'].mean():.0f}")

print(f"\n滯銷建案:")
print(f"   長期滯銷數: {len(stagnant_projects[stagnant_projects['is_long_term_stagnant'] == True]):,}")
print(f"   高風險數: {len(stagnant_projects[stagnant_projects['stagnant_risk_level'] == 'High']):,}")

# 確認關鍵欄位存在
required_columns = ['備查編號', '交易年季', '是否正常交易', '是否解約']
missing_columns = [col for col in required_columns if col not in clean_transactions.columns]
if missing_columns:
    print(f"⚠️ 缺少關鍵欄位: {missing_columns}")
else:
    print(f"✅ 所有必要欄位都存在")

# %% [markdown]
# ## 2. 年季處理與時間對齊邏輯

# %%
# 年季處理工具函數
print("🕐 年季處理與時間對齊邏輯")
print("=" * 60)

def season_to_number(season_str):
    """
    將年季字串轉換為可比較的數字
    例: "111Y1S" -> 1111, "111Y2S" -> 1112
    """
    try:
        if not season_str or pd.isna(season_str):
            return 0
        
        season_str = str(season_str).strip()
        if 'Y' not in season_str or 'S' not in season_str:
            return 0
            
        year_part = season_str.split('Y')[0]
        season_part = season_str.split('Y')[1].replace('S', '')
        return int(year_part) * 10 + int(season_part)
    except:
        return 0

def number_to_season(season_num):
    """
    將數字轉換回年季字串
    例: 1111 -> "111Y1S"
    """
    try:
        if season_num <= 0:
            return ""
        
        year = season_num // 10
        season = season_num % 10
        return f"{year:03d}Y{season}S"
    except:
        return ""

def get_season_range(start_season, end_season):
    """
    獲取兩個年季之間的所有年季
    """
    start_num = season_to_number(start_season)
    end_num = season_to_number(end_season)
    
    seasons = []
    current = start_num
    
    while current <= end_num:
        seasons.append(number_to_season(current))
        
        # 季度遞增邏輯
        season = current % 10
        year = current // 10
        
        if season == 4:  # 第4季 -> 下一年第1季
            current = (year + 1) * 10 + 1
        else:  # 季度+1
            current = year * 10 + (season + 1)
    
    return seasons

def calculate_sales_seasons(start_season, target_season):
    """
    計算從銷售起始到目標年季的累積季數
    """
    start_num = season_to_number(start_season)
    target_num = season_to_number(target_season)
    
    if start_num == 0 or target_num == 0 or target_num < start_num:
        return 0
    
    sales_seasons = 0
    current = start_num
    
    while current <= target_num:
        sales_seasons += 1
        
        # 季度遞增
        season = current % 10
        year = current // 10
        
        if season == 4:
            current = (year + 1) * 10 + 1
        else:
            current = year * 10 + (season + 1)
    
    return sales_seasons

# %%
# 測試年季處理函數
print("🧪 年季處理函數測試:")

test_cases = [
    ("111Y1S", "111Y4S", 4),
    ("111Y4S", "112Y1S", 2),
    ("110Y3S", "113Y2S", 12),
    ("113Y1S", "113Y2S", 2)
]

for start, end, expected in test_cases:
    result = calculate_sales_seasons(start, end)
    status = "✅" if result == expected else "❌"
    print(f"   {status} {start} -> {end}: {result} 季 (預期: {expected})")

# 獲取交易資料中的年季範圍
available_seasons = sorted(clean_transactions['交易年季'].unique(), key=season_to_number)
print(f"\n📅 可用年季範圍: {available_seasons[0]} ~ {available_seasons[-1]} (共 {len(available_seasons)} 季)")

# %% [markdown]
# ## 3. 毛去化率計算實作

# %%
# 毛去化率計算邏輯
print("📈 毛去化率計算實作")
print("=" * 60)

def calculate_gross_absorption_rate(project_code, target_season, transactions_df, projects_df):
    """
    計算毛去化率
    
    毛去化率 = 累積成交筆數 ÷ 總戶數 × 100%
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        transactions_df: 交易資料
        projects_df: 建案資料
        
    Returns:
        dict: 毛去化率計算結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'total_units': 0,
        'cumulative_transactions': 0,
        'gross_absorption_rate': 0.0,
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取建案總戶數
        project_info = projects_df[projects_df['project_code'] == project_code]
        if project_info.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到建案資訊'
            return result
        
        total_units = project_info['total_units'].iloc[0]
        if total_units <= 0:
            result['calculation_status'] = 'error'
            result['error_message'] = '總戶數無效'
            return result
        
        result['total_units'] = total_units
        
        # 獲取該建案截至目標年季的所有有效交易
        project_transactions = transactions_df[
            (transactions_df['備查編號'] == project_code) &
            (transactions_df['是否正常交易'] == True) &
            (transactions_df['交易年季'] <= target_season)
        ].copy()
        
        # 按年季排序並計算累積成交筆數
        if not project_transactions.empty:
            # 添加年季數字排序欄位
            project_transactions['season_num'] = project_transactions['交易年季'].apply(season_to_number)
            target_season_num = season_to_number(target_season)
            
            # 只取目標年季之前（包含）的交易
            valid_transactions = project_transactions[
                project_transactions['season_num'] <= target_season_num
            ]
            
            cumulative_transactions = len(valid_transactions)
        else:
            cumulative_transactions = 0
        
        result['cumulative_transactions'] = cumulative_transactions
        
        # 計算毛去化率
        if total_units > 0:
            gross_absorption_rate = (cumulative_transactions / total_units) * 100
            result['gross_absorption_rate'] = round(gross_absorption_rate, 2)
        
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算毛去化率
print("🔄 批量計算毛去化率...")

# 選取目標年季進行測試
target_seasons = ['113Y1S', '113Y2S', '113Y3S', '113Y4S']

gross_absorption_results = []

# 對所有活躍建案計算毛去化率
for target_season in target_seasons:
    print(f"   計算 {target_season} 毛去化率...")
    
    for _, project in active_projects.iterrows():
        project_code = project['project_code']
        
        # 只計算活躍建案
        if not project['is_active']:
            continue
        
        result = calculate_gross_absorption_rate(
            project_code, target_season, clean_transactions, active_projects
        )
        
        # 添加額外資訊
        result.update({
            'county': project['county'],
            'district': project['district'],
            'project_name': project['project_name'],
            'has_complete_info': project['has_complete_info']
        })
        
        gross_absorption_results.append(result)

# 轉換為DataFrame
gross_absorption_df = pd.DataFrame(gross_absorption_results)

print(f"✅ 完成 {len(gross_absorption_df)} 筆毛去化率計算")

# %%
# 毛去化率統計分析
print(f"\n📊 毛去化率統計分析:")

if not gross_absorption_df.empty:
    # 整體統計
    successful_calculations = gross_absorption_df[gross_absorption_df['calculation_status'] == 'success']
    error_calculations = gross_absorption_df[gross_absorption_df['calculation_status'] == 'error']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_calculations):,} 筆 ({len(successful_calculations)/len(gross_absorption_df)*100:.1f}%)")
    print(f"   計算錯誤: {len(error_calculations):,} 筆")
    
    if not successful_calculations.empty:
        # 去化率分布統計
        for season in target_seasons:
            season_data = successful_calculations[successful_calculations['target_season'] == season]
            if not season_data.empty:
                print(f"\n{season} 毛去化率統計:")
                print(f"   平均去化率: {season_data['gross_absorption_rate'].mean():.1f}%")
                print(f"   中位數去化率: {season_data['gross_absorption_rate'].median():.1f}%")
                print(f"   最高去化率: {season_data['gross_absorption_rate'].max():.1f}%")
                print(f"   最低去化率: {season_data['gross_absorption_rate'].min():.1f}%")
                
                # 去化率分級統計
                high_absorption = len(season_data[season_data['gross_absorption_rate'] >= 70])
                medium_absorption = len(season_data[(season_data['gross_absorption_rate'] >= 30) & 
                                                   (season_data['gross_absorption_rate'] < 70)])
                low_absorption = len(season_data[season_data['gross_absorption_rate'] < 30])
                
                print(f"   高去化率 (≥70%): {high_absorption} 個")
                print(f"   中去化率 (30-70%): {medium_absorption} 個")
                print(f"   低去化率 (<30%): {low_absorption} 個")

# %% [markdown]
# ## 4. 淨去化率計算實作

# %%
# 淨去化率計算邏輯
print("📉 淨去化率計算實作")
print("=" * 60)

def calculate_net_absorption_rate(project_code, target_season, transactions_df, projects_df):
    """
    計算淨去化率
    
    淨去化率 = (累積成交筆數 - 累積解約筆數) ÷ 總戶數 × 100%
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        transactions_df: 交易資料
        projects_df: 建案資料
        
    Returns:
        dict: 淨去化率計算結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'total_units': 0,
        'cumulative_transactions': 0,
        'cumulative_cancellations': 0,
        'net_transactions': 0,
        'net_absorption_rate': 0.0,
        'cancellation_rate': 0.0,
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取建案總戶數
        project_info = projects_df[projects_df['project_code'] == project_code]
        if project_info.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到建案資訊'
            return result
        
        total_units = project_info['total_units'].iloc[0]
        if total_units <= 0:
            result['calculation_status'] = 'error'
            result['error_message'] = '總戶數無效'
            return result
        
        result['total_units'] = total_units
        
        target_season_num = season_to_number(target_season)
        
        # 獲取該建案截至目標年季的所有交易（包含正常和解約）
        project_transactions = transactions_df[
            (transactions_df['備查編號'] == project_code) &
            (transactions_df['交易年季'].apply(season_to_number) <= target_season_num)
        ].copy()
        
        if not project_transactions.empty:
            # 計算累積成交筆數（正常交易）
            normal_transactions = project_transactions[
                project_transactions['是否正常交易'] == True
            ]
            cumulative_transactions = len(normal_transactions)
            
            # 計算累積解約筆數
            cancelled_transactions = project_transactions[
                project_transactions['是否解約'] == True
            ]
            cumulative_cancellations = len(cancelled_transactions)
            
        else:
            cumulative_transactions = 0
            cumulative_cancellations = 0
        
        result['cumulative_transactions'] = cumulative_transactions
        result['cumulative_cancellations'] = cumulative_cancellations
        
        # 計算淨成交筆數
        net_transactions = cumulative_transactions - cumulative_cancellations
        result['net_transactions'] = max(0, net_transactions)  # 確保不為負數
        
        # 計算淨去化率
        if total_units > 0:
            net_absorption_rate = (result['net_transactions'] / total_units) * 100
            result['net_absorption_rate'] = round(net_absorption_rate, 2)
        
        # 計算解約率
        if cumulative_transactions > 0:
            cancellation_rate = (cumulative_cancellations / cumulative_transactions) * 100
            result['cancellation_rate'] = round(cancellation_rate, 2)
        
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算淨去化率
print("🔄 批量計算淨去化率...")

net_absorption_results = []

# 對所有活躍建案計算淨去化率
for target_season in target_seasons:
    print(f"   計算 {target_season} 淨去化率...")
    
    for _, project in active_projects.iterrows():
        project_code = project['project_code']
        
        # 只計算活躍建案
        if not project['is_active']:
            continue
        
        result = calculate_net_absorption_rate(
            project_code, target_season, clean_transactions, active_projects
        )
        
        # 添加額外資訊
        result.update({
            'county': project['county'],
            'district': project['district'],
            'project_name': project['project_name'],
            'has_complete_info': project['has_complete_info']
        })
        
        net_absorption_results.append(result)

# 轉換為DataFrame
net_absorption_df = pd.DataFrame(net_absorption_results)

print(f"✅ 完成 {len(net_absorption_df)} 筆淨去化率計算")

# %%
# 淨去化率統計分析
print(f"\n📊 淨去化率統計分析:")

if not net_absorption_df.empty:
    # 整體統計
    successful_net_calculations = net_absorption_df[net_absorption_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_net_calculations):,} 筆")
    
    if not successful_net_calculations.empty:
        # 解約影響分析
        total_gross_transactions = successful_net_calculations['cumulative_transactions'].sum()
        total_cancellations = successful_net_calculations['cumulative_cancellations'].sum()
        total_net_transactions = successful_net_calculations['net_transactions'].sum()
        
        print(f"\n整體解約影響分析:")
        print(f"   總成交筆數: {total_gross_transactions:,}")
        print(f"   總解約筆數: {total_cancellations:,}")
        print(f"   總淨成交筆數: {total_net_transactions:,}")
        print(f"   整體解約率: {total_cancellations/total_gross_transactions*100:.2f}%" if total_gross_transactions > 0 else "   整體解約率: 0.00%")
        
        # 各年季淨去化率統計
        for season in target_seasons:
            season_data = successful_net_calculations[successful_net_calculations['target_season'] == season]
            if not season_data.empty:
                print(f"\n{season} 淨去化率統計:")
                print(f"   平均淨去化率: {season_data['net_absorption_rate'].mean():.1f}%")
                print(f"   中位數淨去化率: {season_data['net_absorption_rate'].median():.1f}%")
                print(f"   平均解約率: {season_data['cancellation_rate'].mean():.2f}%")
                
                # 解約影響分級
                high_cancellation = len(season_data[season_data['cancellation_rate'] > 5])
                medium_cancellation = len(season_data[(season_data['cancellation_rate'] > 2) & 
                                                     (season_data['cancellation_rate'] <= 5)])
                low_cancellation = len(season_data[season_data['cancellation_rate'] <= 2])
                
                print(f"   高解約率 (>5%): {high_cancellation} 個")
                print(f"   中解約率 (2-5%): {medium_cancellation} 個")
                print(f"   低解約率 (≤2%): {low_cancellation} 個")

# %% [markdown]
# ## 5. 調整去化率計算實作

# %%
# 調整去化率計算邏輯
print("🔧 調整去化率計算實作")
print("=" * 60)

def get_season_days(season_str):
    """
    獲取指定年季的總天數
    """
    try:
        # 解析年季
        year_part = season_str.split('Y')[0]
        season_part = season_str.split('Y')[1].replace('S', '')
        
        year = int(year_part) + 1911  # 轉為西元年
        season = int(season_part)
        
        # 計算各季度的天數
        if season == 1:  # 第1季 (1-3月)
            # 檢查是否閏年
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            return 31 + (29 if is_leap else 28) + 31  # 1月 + 2月 + 3月
        elif season == 2:  # 第2季 (4-6月)
            return 30 + 31 + 30  # 4月 + 5月 + 6月
        elif season == 3:  # 第3季 (7-9月)
            return 31 + 31 + 30  # 7月 + 8月 + 9月
        elif season == 4:  # 第4季 (10-12月)
            return 31 + 30 + 31  # 10月 + 11月 + 12月
        else:
            return 90  # 預設90天
            
    except:
        return 90  # 預設90天

def is_complete_season(target_season, analysis_date=None):
    """
    判斷目標年季是否為完整季度
    
    Args:
        target_season: 目標年季
        analysis_date: 分析日期 (預設為當前時間)
        
    Returns:
        bool: 是否為完整季度
    """
    if analysis_date is None:
        analysis_date = datetime.now()
    
    try:
        # 解析目標年季
        year_part = target_season.split('Y')[0]
        season_part = target_season.split('Y')[1].replace('S', '')
        
        target_year = int(year_part) + 1911
        target_season_num = int(season_part)
        
        # 計算目標年季的結束日期
        if target_season_num == 1:
            end_date = datetime(target_year, 3, 31)
        elif target_season_num == 2:
            end_date = datetime(target_year, 6, 30)
        elif target_season_num == 3:
            end_date = datetime(target_year, 9, 30)
        elif target_season_num == 4:
            end_date = datetime(target_year, 12, 31)
        else:
            return True  # 預設為完整季度
        
        # 如果分析日期在季度結束日期之後，則為完整季度
        return analysis_date.date() > end_date.date()
        
    except:
        return True  # 預設為完整季度

def calculate_adjusted_absorption_rate(project_code, target_season, transactions_df, projects_df, analysis_date=None):
    """
    計算調整去化率
    
    對於非完整季度，根據實際銷售天數進行標準化調整
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        transactions_df: 交易資料
        projects_df: 建案資料
        analysis_date: 分析基準日期
        
    Returns:
        dict: 調整去化率計算結果
    """
    
    # 先計算淨去化率作為基礎
    net_result = calculate_net_absorption_rate(project_code, target_season, transactions_df, projects_df)
    
    result = {
        **net_result,
        'season_total_days': 0,
        'season_sales_days': 0,
        'is_complete_season': True,
        'adjustment_factor': 1.0,
        'adjusted_absorption_rate': net_result.get('net_absorption_rate', 0.0)
    }
    
    if net_result['calculation_status'] != 'success':
        return result
    
    try:
        # 判斷是否為完整季度
        is_complete = is_complete_season(target_season, analysis_date)
        result['is_complete_season'] = is_complete
        
        # 獲取季度總天數
        season_total_days = get_season_days(target_season)
        result['season_total_days'] = season_total_days
        
        if is_complete:
            # 完整季度，不需調整
            result['season_sales_days'] = season_total_days
            result['adjustment_factor'] = 1.0
            result['adjusted_absorption_rate'] = net_result['net_absorption_rate']
        else:
            # 非完整季度，需要調整
            if analysis_date is None:
                analysis_date = datetime.now()
            
            # 計算季度起始日期
            year_part = target_season.split('Y')[0]
            season_part = target_season.split('Y')[1].replace('S', '')
            target_year = int(year_part) + 1911
            target_season_num = int(season_part)
            
            if target_season_num == 1:
                season_start = datetime(target_year, 1, 1)
            elif target_season_num == 2:
                season_start = datetime(target_year, 4, 1)
            elif target_season_num == 3:
                season_start = datetime(target_year, 7, 1)
            elif target_season_num == 4:
                season_start = datetime(target_year, 10, 1)
            
            # 計算實際銷售天數
            season_sales_days = min(season_total_days, (analysis_date - season_start).days + 1)
            season_sales_days = max(1, season_sales_days)  # 至少1天
            result['season_sales_days'] = season_sales_days
            
            # 計算調整係數
            adjustment_factor = season_total_days / season_sales_days
            result['adjustment_factor'] = round(adjustment_factor, 3)
            
            # 計算調整後去化率
            adjusted_absorption_rate = net_result['net_absorption_rate'] * adjustment_factor
            result['adjusted_absorption_rate'] = round(adjusted_absorption_rate, 2)
    
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = f"調整去化率計算錯誤: {str(e)}"
    
    return result

# %%
# 批量計算調整去化率
print("🔄 批量計算調整去化率...")

# 設定分析基準日期 (假設為113年第4季末的分析)
analysis_date = datetime(2024, 12, 31)  # 113年第4季末

adjusted_absorption_results = []

# 對所有活躍建案計算調整去化率
for target_season in target_seasons:
    print(f"   計算 {target_season} 調整去化率...")
    
    for _, project in active_projects.iterrows():
        project_code = project['project_code']
        
        # 只計算活躍建案
        if not project['is_active']:
            continue
        
        result = calculate_adjusted_absorption_rate(
            project_code, target_season, clean_transactions, active_projects, analysis_date
        )
        
        # 添加額外資訊
        result.update({
            'county': project['county'],
            'district': project['district'],
            'project_name': project['project_name'],
            'has_complete_info': project['has_complete_info']
        })
        
        adjusted_absorption_results.append(result)

# 轉換為DataFrame
adjusted_absorption_df = pd.DataFrame(adjusted_absorption_results)

print(f"✅ 完成 {len(adjusted_absorption_df)} 筆調整去化率計算")

# %%
# 調整去化率統計分析
print(f"\n📊 調整去化率統計分析:")

if not adjusted_absorption_df.empty:
    successful_adj_calculations = adjusted_absorption_df[adjusted_absorption_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_adj_calculations):,} 筆")
    
    if not successful_adj_calculations.empty:
        # 完整季度 vs 非完整季度統計
        complete_seasons = successful_adj_calculations[successful_adj_calculations['is_complete_season'] == True]
        incomplete_seasons = successful_adj_calculations[successful_adj_calculations['is_complete_season'] == False]
        
        print(f"\n季度完整性統計:")
        print(f"   完整季度記錄: {len(complete_seasons):,} 筆")
        print(f"   非完整季度記錄: {len(incomplete_seasons):,} 筆")
        
        if len(incomplete_seasons) > 0:
            print(f"   平均調整係數: {incomplete_seasons['adjustment_factor'].mean():.3f}")
            print(f"   最大調整係數: {incomplete_seasons['adjustment_factor'].max():.3f}")
            
        # 各年季調整效果分析
        for season in target_seasons:
            season_data = successful_adj_calculations[successful_adj_calculations['target_season'] == season]
            if not season_data.empty:
                complete_count = len(season_data[season_data['is_complete_season'] == True])
                incomplete_count = len(season_data[season_data['is_complete_season'] == False])
                
                print(f"\n{season} 調整效果分析:")
                print(f"   完整季度: {complete_count} 個")
                print(f"   非完整季度: {incomplete_count} 個")
                
                if incomplete_count > 0:
                    incomplete_data = season_data[season_data['is_complete_season'] == False]
                    print(f"   平均調整前去化率: {incomplete_data['net_absorption_rate'].mean():.1f}%")
                    print(f"   平均調整後去化率: {incomplete_data['adjusted_absorption_rate'].mean():.1f}%")
                    print(f"   平均調整係數: {incomplete_data['adjustment_factor'].mean():.3f}")

# %% [markdown]
# ## 6. 去化率合理性驗證

# %%
# 去化率合理性驗證
print("🔍 去化率合理性驗證")
print("=" * 60)

def validate_absorption_rates(absorption_df):
    """
    驗證去化率計算結果的合理性
    
    Args:
        absorption_df: 包含去化率計算結果的DataFrame
        
    Returns:
        dict: 驗證結果報告
    """
    
    validation_report = {
        'total_records': len(absorption_df),
        'validation_errors': [],
        'warning_cases': [],
        'quality_metrics': {}
    }
    
    # 過濾成功計算的記錄
    valid_data = absorption_df[absorption_df['calculation_status'] == 'success'].copy()
    validation_report['valid_records'] = len(valid_data)
    
    if len(valid_data) == 0:
        validation_report['validation_errors'].append("沒有有效的計算記錄")
        return validation_report
    
    # 驗證1: 去化率不能超過100%
    if 'net_absorption_rate' in valid_data.columns:
        over_100_net = valid_data[valid_data['net_absorption_rate'] > 100]
        if len(over_100_net) > 0:
            validation_report['validation_errors'].append(f"發現 {len(over_100_net)} 筆淨去化率超過100%")
    
    if 'adjusted_absorption_rate' in valid_data.columns:
        over_100_adj = valid_data[valid_data['adjusted_absorption_rate'] > 100]
        if len(over_100_adj) > 0:
            validation_report['warning_cases'].append(f"發現 {len(over_100_adj)} 筆調整去化率超過100%")
    
    # 驗證2: 解約數不能超過成交數
    if 'cumulative_cancellations' in valid_data.columns and 'cumulative_transactions' in valid_data.columns:
        invalid_cancellation = valid_data[valid_data['cumulative_cancellations'] > valid_data['cumulative_transactions']]
        if len(invalid_cancellation) > 0:
            validation_report['validation_errors'].append(f"發現 {len(invalid_cancellation)} 筆解約數超過成交數")
    
    # 驗證3: 淨成交數不能為負數
    if 'net_transactions' in valid_data.columns:
        negative_net = valid_data[valid_data['net_transactions'] < 0]
        if len(negative_net) > 0:
            validation_report['validation_errors'].append(f"發現 {len(negative_net)} 筆淨成交數為負數")
    
    # 驗證4: 異常高去化率檢查 (超過150%為異常)
    if 'adjusted_absorption_rate' in valid_data.columns:
        extreme_high = valid_data[valid_data['adjusted_absorption_rate'] > 150]
        if len(extreme_high) > 0:
            validation_report['warning_cases'].append(f"發現 {len(extreme_high)} 筆調整去化率超過150%")
    
    # 驗證5: 調整係數合理性檢查
    if 'adjustment_factor' in valid_data.columns:
        extreme_adjustment = valid_data[valid_data['adjustment_factor'] > 4.0]
        if len(extreme_adjustment) > 0:
            validation_report['warning_cases'].append(f"發現 {len(extreme_adjustment)} 筆調整係數超過4.0")
    
    # 計算品質指標
    if len(valid_data) > 0:
        validation_report['quality_metrics'] = {
            'valid_calculation_rate': len(valid_data) / len(absorption_df) * 100,
            'zero_absorption_rate': len(valid_data[valid_data.get('net_absorption_rate', 0) == 0]) / len(valid_data) * 100,
            'high_absorption_rate': len(valid_data[valid_data.get('net_absorption_rate', 0) > 80]) / len(valid_data) * 100,
            'average_cancellation_rate': valid_data.get('cancellation_rate', pd.Series([0])).mean()
        }
    
    return validation_report

# %%
# 執行去化率合理性驗證
print("🔄 執行去化率合理性驗證...")

# 合併所有去化率計算結果進行驗證
combined_absorption_df = adjusted_absorption_df.copy()

validation_result = validate_absorption_rates(combined_absorption_df)

print(f"✅ 完成去化率合理性驗證")
print(f"\n📊 驗證結果報告:")
print(f"   總記錄數: {validation_result['total_records']:,}")
print(f"   有效記錄數: {validation_result['valid_records']:,}")
print(f"   有效計算率: {validation_result['quality_metrics'].get('valid_calculation_rate', 0):.1f}%")

# 錯誤報告
if validation_result['validation_errors']:
    print(f"\n❌ 發現驗證錯誤:")
    for i, error in enumerate(validation_result['validation_errors'], 1):
        print(f"   {i}. {error}")
else:
    print(f"\n✅ 沒有發現驗證錯誤")

# 警告報告
if validation_result['warning_cases']:
    print(f"\n⚠️ 發現警告案例:")
    for i, warning in enumerate(validation_result['warning_cases'], 1):
        print(f"   {i}. {warning}")
else:
    print(f"\n✅ 沒有發現警告案例")

# 品質指標
quality_metrics = validation_result['quality_metrics']
if quality_metrics:
    print(f"\n📈 品質指標:")
    print(f"   零去化率比例: {quality_metrics.get('zero_absorption_rate', 0):.1f}%")
    print(f"   高去化率比例: {quality_metrics.get('high_absorption_rate', 0):.1f}%")
    print(f"   平均解約率: {quality_metrics.get('average_cancellation_rate', 0):.2f}%")

# %% [markdown]
# ## 7. 異常案例識別與處理

# %%
# 異常案例識別與處理
print("🚨 異常案例識別與處理")
print("=" * 60)

def identify_anomalous_cases(absorption_df, thresholds=None):
    """
    識別異常的去化率案例
    
    Args:
        absorption_df: 去化率計算結果
        thresholds: 異常判斷閾值
        
    Returns:
        dict: 異常案例分析結果
    """
    
    if thresholds is None:
        thresholds = {
            'extreme_high_absorption': 150,
            'extreme_adjustment_factor': 4.0,
            'high_cancellation_rate': 10.0,
            'suspicious_zero_absorption': 0.1
        }
    
    valid_data = absorption_df[absorption_df['calculation_status'] == 'success'].copy()
    
    anomalous_cases = {
        'extreme_high_absorption': [],
        'extreme_adjustment': [],
        'high_cancellation': [],
        'suspicious_patterns': [],
        'data_quality_issues': []
    }
    
    if len(valid_data) == 0:
        return anomalous_cases
    
    # 1. 極端高去化率案例
    if 'adjusted_absorption_rate' in valid_data.columns:
        extreme_high = valid_data[valid_data['adjusted_absorption_rate'] > thresholds['extreme_high_absorption']]
        for _, case in extreme_high.iterrows():
            anomalous_cases['extreme_high_absorption'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'adjusted_absorption_rate': case['adjusted_absorption_rate'],
                'net_absorption_rate': case.get('net_absorption_rate', 0),
                'total_units': case.get('total_units', 0),
                'cumulative_transactions': case.get('cumulative_transactions', 0)
            })
    
    # 2. 極端調整係數案例
    if 'adjustment_factor' in valid_data.columns:
        extreme_adj = valid_data[valid_data['adjustment_factor'] > thresholds['extreme_adjustment_factor']]
        for _, case in extreme_adj.iterrows():
            anomalous_cases['extreme_adjustment'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'target_season': case['target_season'],
                'adjustment_factor': case['adjustment_factor'],
                'season_sales_days': case.get('season_sales_days', 0),
                'season_total_days': case.get('season_total_days', 0),
                'is_complete_season': case.get('is_complete_season', True)
            })
    
    # 3. 高解約率案例
    if 'cancellation_rate' in valid_data.columns:
        high_cancel = valid_data[valid_data['cancellation_rate'] > thresholds['high_cancellation_rate']]
        for _, case in high_cancel.iterrows():
            anomalous_cases['high_cancellation'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'cancellation_rate': case['cancellation_rate'],
                'cumulative_transactions': case.get('cumulative_transactions', 0),
                'cumulative_cancellations': case.get('cumulative_cancellations', 0)
            })
    
    # 4. 可疑模式識別
    # 識別單戶數但高交易量的案例
    suspicious_single_unit = valid_data[
        (valid_data.get('total_units', 1) == 1) & 
        (valid_data.get('cumulative_transactions', 0) > 1)
    ]
    for _, case in suspicious_single_unit.iterrows():
        anomalous_cases['suspicious_patterns'].append({
            'project_code': case['project_code'],
            'issue_type': '單戶數高交易量',
            'total_units': case.get('total_units', 0),
            'cumulative_transactions': case.get('cumulative_transactions', 0),
            'details': f"戶數: {case.get('total_units', 0)}, 交易: {case.get('cumulative_transactions', 0)}"
        })
    
    # 5. 資料品質問題
    # 檢查有完整資訊但計算異常的案例
    complete_info_errors = valid_data[
        (valid_data.get('has_complete_info', False) == True) & 
        (valid_data.get('net_absorption_rate', 0) == 0) &
        (valid_data.get('cumulative_transactions', 0) > 0)
    ]
    for _, case in complete_info_errors.iterrows():
        anomalous_cases['data_quality_issues'].append({
            'project_code': case['project_code'],
            'issue_type': '完整資訊但零去化率',
            'cumulative_transactions': case.get('cumulative_transactions', 0),
            'total_units': case.get('total_units', 0),
            'details': f"有 {case.get('cumulative_transactions', 0)} 筆交易但去化率為0"
        })
    
    return anomalous_cases

# %%
# 執行異常案例識別
print("🔄 執行異常案例識別...")

anomalous_analysis = identify_anomalous_cases(combined_absorption_df)

print(f"✅ 完成異常案例識別")
print(f"\n📊 異常案例統計:")

total_anomalies = sum(len(cases) for cases in anomalous_analysis.values())
print(f"   總異常案例數: {total_anomalies}")

for category, cases in anomalous_analysis.items():
    if len(cases) > 0:
        print(f"   {category}: {len(cases)} 個案例")

# 詳細異常案例報告
if total_anomalies > 0:
    print(f"\n🔍 詳細異常案例報告:")
    
    # 極端高去化率案例
    if anomalous_analysis['extreme_high_absorption']:
        print(f"\n1. 極端高去化率案例 (前5個):")
        for i, case in enumerate(anomalous_analysis['extreme_high_absorption'][:5], 1):
            print(f"   {i}. {case['project_code']} | {case['county']} | 去化率: {case['adjusted_absorption_rate']:.1f}% | 戶數: {case['total_units']}")
    
    # 極端調整係數案例
    if anomalous_analysis['extreme_adjustment']:
        print(f"\n2. 極端調整係數案例 (前5個):")
        for i, case in enumerate(anomalous_analysis['extreme_adjustment'][:5], 1):
            print(f"   {i}. {case['project_code']} | 調整係數: {case['adjustment_factor']:.3f} | 銷售天數: {case['season_sales_days']}")
    
    # 高解約率案例
    if anomalous_analysis['high_cancellation']:
        print(f"\n3. 高解約率案例 (前5個):")
        for i, case in enumerate(anomalous_analysis['high_cancellation'][:5], 1):
            print(f"   {i}. {case['project_code']} | {case['county']} | 解約率: {case['cancellation_rate']:.1f}% | 解約: {case['cumulative_cancellations']}")
    
    # 可疑模式案例
    if anomalous_analysis['suspicious_patterns']:
        print(f"\n4. 可疑模式案例:")
        for i, case in enumerate(anomalous_analysis['suspicious_patterns'][:3], 1):
            print(f"   {i}. {case['project_code']} | {case['issue_type']} | {case['details']}")

# %% [markdown]
# ## 8. 去化率基準值建立

# %%
# 去化率基準值建立
print("📏 去化率基準值建立")
print("=" * 60)

def establish_absorption_benchmarks(absorption_df):
    """
    建立去化率基準值和分級標準
    
    Args:
        absorption_df: 去化率計算結果
        
    Returns:
        dict: 基準值和分級標準
    """
    
    valid_data = absorption_df[absorption_df['calculation_status'] == 'success'].copy()
    
    if len(valid_data) == 0:
        return {}
    
    benchmarks = {}
    
    # 基於淨去化率建立基準值
    if 'net_absorption_rate' in valid_data.columns:
        net_rates = valid_data['net_absorption_rate']
        
        benchmarks['net_absorption_rate'] = {
            'mean': net_rates.mean(),
            'median': net_rates.median(),
            'std': net_rates.std(),
            'percentiles': {
                '10th': net_rates.quantile(0.1),
                '25th': net_rates.quantile(0.25),
                '75th': net_rates.quantile(0.75),
                '90th': net_rates.quantile(0.9)
            },
            'classification': {
                'high_performance': net_rates.quantile(0.75),  # 前25%
                'good_performance': net_rates.quantile(0.5),   # 前50%
                'average_performance': net_rates.quantile(0.25), # 前75%
                'below_average': 0  # 低於平均
            }
        }
    
    # 基於解約率建立基準值
    if 'cancellation_rate' in valid_data.columns:
        cancel_rates = valid_data['cancellation_rate']
        
        benchmarks['cancellation_rate'] = {
            'mean': cancel_rates.mean(),
            'median': cancel_rates.median(),
            'std': cancel_rates.std(),
            'percentiles': {
                '10th': cancel_rates.quantile(0.1),
                '25th': cancel_rates.quantile(0.25),
                '75th': cancel_rates.quantile(0.75),
                '90th': cancel_rates.quantile(0.9)
            },
            'risk_classification': {
                'low_risk': cancel_rates.quantile(0.5),      # 低於中位數
                'medium_risk': cancel_rates.quantile(0.75),  # 75分位數
                'high_risk': cancel_rates.quantile(0.9),     # 90分位數
                'extreme_risk': cancel_rates.quantile(0.95)  # 95分位數
            }
        }
    
    # 基於銷售規模建立基準值
    if 'total_units' in valid_data.columns:
        unit_sizes = valid_data['total_units']
        
        benchmarks['project_scale'] = {
            'small_project': unit_sizes.quantile(0.33),    # 小型建案
            'medium_project': unit_sizes.quantile(0.67),   # 中型建案
            'large_project': unit_sizes.quantile(1.0)      # 大型建案
        }
    
    # 綜合分級標準
    benchmarks['comprehensive_grade'] = {
        'excellent': {
            'net_absorption_rate': ('>=', benchmarks['net_absorption_rate']['percentiles']['75th']),
            'cancellation_rate': ('<=', benchmarks['cancellation_rate']['percentiles']['25th'])
        },
        'good': {
            'net_absorption_rate': ('>=', benchmarks['net_absorption_rate']['percentiles']['50th']),
            'cancellation_rate': ('<=', benchmarks['cancellation_rate']['percentiles']['50th'])
        },
        'average': {
            'net_absorption_rate': ('>=', benchmarks['net_absorption_rate']['percentiles']['25th']),
            'cancellation_rate': ('<=', benchmarks['cancellation_rate']['percentiles']['75th'])
        },
        'below_average': {
            'net_absorption_rate': ('<', benchmarks['net_absorption_rate']['percentiles']['25th']),
            'cancellation_rate': ('>', benchmarks['cancellation_rate']['percentiles']['75th'])
        }
    }
    
    return benchmarks

# %%
# 建立去化率基準值
print("🔄 建立去化率基準值...")

absorption_benchmarks = establish_absorption_benchmarks(combined_absorption_df)

print(f"✅ 完成去化率基準值建立")

if absorption_benchmarks:
    print(f"\n📊 去化率基準值報告:")
    
    # 淨去化率基準值
    if 'net_absorption_rate' in absorption_benchmarks:
        net_bench = absorption_benchmarks['net_absorption_rate']
        print(f"\n淨去化率基準值:")
        print(f"   平均值: {net_bench['mean']:.1f}%")
        print(f"   中位數: {net_bench['median']:.1f}%")
        print(f"   標準差: {net_bench['std']:.1f}%")
        print(f"   分位數:")
        for pct, value in net_bench['percentiles'].items():
            print(f"     {pct}: {value:.1f}%")
        
        print(f"   分級標準:")
        for grade, threshold in net_bench['classification'].items():
            print(f"     {grade}: {threshold:.1f}%")
    
    # 解約率基準值
    if 'cancellation_rate' in absorption_benchmarks:
        cancel_bench = absorption_benchmarks['cancellation_rate']
        print(f"\n解約率基準值:")
        print(f"   平均值: {cancel_bench['mean']:.2f}%")
        print(f"   中位數: {cancel_bench['median']:.2f}%")
        print(f"   風險分級:")
        for risk, threshold in cancel_bench['risk_classification'].items():
            print(f"     {risk}: {threshold:.2f}%")
    
    # 建案規模基準值
    if 'project_scale' in absorption_benchmarks:
        scale_bench = absorption_benchmarks['project_scale']
        print(f"\n建案規模基準值:")
        print(f"   小型建案 (≤{scale_bench['small_project']:.0f}戶)")
        print(f"   中型建案 ({scale_bench['small_project']:.0f}-{scale_bench['medium_project']:.0f}戶)")
        print(f"   大型建案 (>{scale_bench['medium_project']:.0f}戶)")

# %%
# 應用基準值進行建案分級
def apply_absorption_grading(absorption_df, benchmarks):
    """
    應用基準值對建案進行分級
    
    Args:
        absorption_df: 去化率計算結果
        benchmarks: 基準值標準
        
    Returns:
        DataFrame: 包含分級結果的資料
    """
    
    graded_df = absorption_df.copy()
    
    if 'net_absorption_rate' not in graded_df.columns or not benchmarks:
        return graded_df
    
    # 去化率分級
    net_bench = benchmarks.get('net_absorption_rate', {})
    if 'classification' in net_bench:
        def classify_absorption(rate):
            if pd.isna(rate):
                return 'unknown'
            elif rate >= net_bench['classification']['high_performance']:
                return 'high_performance'
            elif rate >= net_bench['classification']['good_performance']:
                return 'good_performance'
            elif rate >= net_bench['classification']['average_performance']:
                return 'average_performance'
            else:
                return 'below_average'
        
        graded_df['absorption_grade'] = graded_df['net_absorption_rate'].apply(classify_absorption)
    
    # 解約風險分級
    cancel_bench = benchmarks.get('cancellation_rate', {})
    if 'risk_classification' in cancel_bench:
        def classify_cancellation_risk(rate):
            if pd.isna(rate):
                return 'unknown'
            elif rate >= cancel_bench['risk_classification']['extreme_risk']:
                return 'extreme_risk'
            elif rate >= cancel_bench['risk_classification']['high_risk']:
                return 'high_risk'
            elif rate >= cancel_bench['risk_classification']['medium_risk']:
                return 'medium_risk'
            else:
                return 'low_risk'
        
        graded_df['cancellation_risk_grade'] = graded_df['cancellation_rate'].apply(classify_cancellation_risk)
    
    # 建案規模分級
    scale_bench = benchmarks.get('project_scale', {})
    if scale_bench:
        def classify_project_scale(units):
            if pd.isna(units):
                return 'unknown'
            elif units <= scale_bench['small_project']:
                return 'small'
            elif units <= scale_bench['medium_project']:
                return 'medium'
            else:
                return 'large'
        
        graded_df['project_scale_grade'] = graded_df['total_units'].apply(classify_project_scale)
    
    return graded_df

# 應用分級標準
print(f"\n🔄 應用基準值進行建案分級...")

graded_absorption_df = apply_absorption_grading(combined_absorption_df, absorption_benchmarks)

# 分級結果統計
if 'absorption_grade' in graded_absorption_df.columns:
    grade_distribution = graded_absorption_df['absorption_grade'].value_counts()
    print(f"\n📊 去化率分級分布:")
    for grade, count in grade_distribution.items():
        percentage = count / len(graded_absorption_df) * 100
        print(f"   {grade}: {count} 個 ({percentage:.1f}%)")

if 'cancellation_risk_grade' in graded_absorption_df.columns:
    risk_distribution = graded_absorption_df['cancellation_risk_grade'].value_counts()
    print(f"\n📊 解約風險分級分布:")
    for risk, count in risk_distribution.items():
        percentage = count / len(graded_absorption_df) * 100
        print(f"   {risk}: {count} 個 ({percentage:.1f}%)")

# %% [markdown]
# ## 9. 視覺化分析

# %%
# 創建去化率分析視覺化
print("📊 去化率計算結果視覺化分析")
print("=" * 50)

# 創建圖表
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 過濾有效數據
valid_data = graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']

# 1. 淨去化率分布直方圖
if 'net_absorption_rate' in valid_data.columns:
    axes[0, 0].hist(valid_data['net_absorption_rate'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('淨去化率分布', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('淨去化率 (%)')
    axes[0, 0].set_ylabel('建案數量')
    axes[0, 0].axvline(x=valid_data['net_absorption_rate'].mean(), color='red', linestyle='--', label=f'平均值: {valid_data["net_absorption_rate"].mean():.1f}%')
    axes[0, 0].legend()

# 2. 解約率分布直方圖
if 'cancellation_rate' in valid_data.columns:
    axes[0, 1].hist(valid_data['cancellation_rate'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('解約率分布', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('解約率 (%)')
    axes[0, 1].set_ylabel('建案數量')
    axes[0, 1].axvline(x=valid_data['cancellation_rate'].mean(), color='blue', linestyle='--', label=f'平均值: {valid_data["cancellation_rate"].mean():.2f}%')
    axes[0, 1].legend()

# 3. 調整係數分布
if 'adjustment_factor' in valid_data.columns:
    adjustment_data = valid_data[valid_data['adjustment_factor'] <= 5]  # 過濾極端值
    axes[0, 2].hist(adjustment_data['adjustment_factor'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('調整係數分布', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('調整係數')
    axes[0, 2].set_ylabel('建案數量')
    axes[0, 2].axvline(x=1.0, color='red', linestyle='-', label='無調整基準線')
    axes[0, 2].legend()

# 4. 年季別去化率變化
if len(target_seasons) > 1:
    season_stats = []
    for season in target_seasons:
        season_data = valid_data[valid_data['target_season'] == season]
        if not season_data.empty:
            season_stats.append({
                'season': season,
                'mean_absorption': season_data['net_absorption_rate'].mean(),
                'count': len(season_data)
            })
    
    if season_stats:
        season_df = pd.DataFrame(season_stats)
        bars = axes[1, 0].bar(season_df['season'], season_df['mean_absorption'], color='orange')
        axes[1, 0].set_title('各年季平均淨去化率', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('年季')
        axes[1, 0].set_ylabel('平均淨去化率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom')

# 5. 去化率分級分布
if 'absorption_grade' in valid_data.columns:
    grade_dist = valid_data['absorption_grade'].value_counts()
    colors = {'high_performance': 'green', 'good_performance': 'lightgreen', 
              'average_performance': 'orange', 'below_average': 'red', 'unknown': 'gray'}
    bar_colors = [colors.get(grade, 'gray') for grade in grade_dist.index]
    
    bars = axes[1, 1].bar(grade_dist.index, grade_dist.values, color=bar_colors)
    axes[1, 1].set_title('去化率分級分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('分級')
    axes[1, 1].set_ylabel('建案數量')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')

# 6. 解約風險分級分布
if 'cancellation_risk_grade' in valid_data.columns:
    risk_dist = valid_data['cancellation_risk_grade'].value_counts()
    risk_colors = {'low_risk': 'green', 'medium_risk': 'orange', 
                   'high_risk': 'red', 'extreme_risk': 'darkred', 'unknown': 'gray'}
    bar_colors = [risk_colors.get(risk, 'gray') for risk in risk_dist.index]
    
    bars = axes[1, 2].bar(risk_dist.index, risk_dist.values, color=bar_colors)
    axes[1, 2].set_title('解約風險分級分布', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('風險等級')
    axes[1, 2].set_ylabel('建案數量')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')

# 7. 縣市別去化率比較
if 'county' in valid_data.columns:
    city_absorption = valid_data.groupby('county')['net_absorption_rate'].agg(['mean', 'count']).reset_index()
    city_absorption = city_absorption[city_absorption['count'] >= 5]  # 至少5個建案
    city_absorption = city_absorption.nlargest(10, 'mean')  # 前10名
    
    if not city_absorption.empty:
        bars = axes[2, 0].bar(range(len(city_absorption)), city_absorption['mean'], color='lightblue')
        axes[2, 0].set_title('縣市別平均去化率 (前10名)', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('縣市')
        axes[2, 0].set_ylabel('平均淨去化率 (%)')
        axes[2, 0].set_xticks(range(len(city_absorption)))
        axes[2, 0].set_xticklabels(city_absorption['county'], rotation=45, ha='right')
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = city_absorption.iloc[i]['count']
            axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%\n({int(count)}個)', ha='center', va='bottom', fontsize=8)

# 8. 去化率 vs 解約率散點圖
if 'net_absorption_rate' in valid_data.columns and 'cancellation_rate' in valid_data.columns:
    scatter_data = valid_data[(valid_data['net_absorption_rate'] <= 150) & (valid_data['cancellation_rate'] <= 20)]
    axes[2, 1].scatter(scatter_data['net_absorption_rate'], scatter_data['cancellation_rate'], 
                      alpha=0.6, color='purple')
    axes[2, 1].set_title('去化率 vs 解約率關係', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('淨去化率 (%)')
    axes[2, 1].set_ylabel('解約率 (%)')
    
    # 添加趨勢線
    if len(scatter_data) > 1:
        z = np.polyfit(scatter_data['net_absorption_rate'], scatter_data['cancellation_rate'], 1)
        p = np.poly1d(z)
        axes[2, 1].plot(scatter_data['net_absorption_rate'], p(scatter_data['net_absorption_rate']), 
                       "r--", alpha=0.8, label=f'趨勢線')
        axes[2, 1].legend()

# 9. 建案規模 vs 去化率
if 'total_units' in valid_data.columns and 'net_absorption_rate' in valid_data.columns:
    size_data = valid_data[(valid_data['total_units'] <= 1000) & (valid_data['net_absorption_rate'] <= 150)]
    axes[2, 2].scatter(size_data['total_units'], size_data['net_absorption_rate'], 
                      alpha=0.6, color='green')
    axes[2, 2].set_title('建案規模 vs 去化率', fontsize=14, fontweight='bold')
    axes[2, 2].set_xlabel('總戶數')
    axes[2, 2].set_ylabel('淨去化率 (%)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. 結果儲存與匯出

# %%
# 儲存去化率計算結果
print("💾 儲存去化率計算結果...")

# 1. 儲存完整的去化率計算結果
output_columns = [
    'project_code', 'project_name', 'county', 'district', 'target_season',
    'total_units', 'cumulative_transactions', 'cumulative_cancellations', 'net_transactions',
    'gross_absorption_rate', 'net_absorption_rate', 'adjusted_absorption_rate',
    'cancellation_rate', 'is_complete_season', 'season_sales_days', 'season_total_days',
    'adjustment_factor', 'has_complete_info', 'calculation_status', 'error_message'
]

# 合併毛去化率和淨去化率結果
if 'gross_absorption_rate' not in graded_absorption_df.columns:
    # 添加毛去化率資料
    gross_lookup = {f"{row['project_code']}_{row['target_season']}": row['gross_absorption_rate'] 
                   for _, row in gross_absorption_df.iterrows()}
    graded_absorption_df['gross_absorption_rate'] = graded_absorption_df.apply(
        lambda x: gross_lookup.get(f"{x['project_code']}_{x['target_season']}", x.get('net_absorption_rate', 0)), 
        axis=1
    )

# 選擇存在的欄位
available_columns = [col for col in output_columns if col in graded_absorption_df.columns]
absorption_output_df = graded_absorption_df[available_columns].copy()

# 添加分級欄位（如果存在）
if 'absorption_grade' in graded_absorption_df.columns:
    absorption_output_df['absorption_grade'] = graded_absorption_df['absorption_grade']
if 'cancellation_risk_grade' in graded_absorption_df.columns:
    absorption_output_df['cancellation_risk_grade'] = graded_absorption_df['cancellation_risk_grade']
if 'project_scale_grade' in graded_absorption_df.columns:
    absorption_output_df['project_scale_grade'] = graded_absorption_df['project_scale_grade']

absorption_output_df.to_csv('../data/processed/05_absorption_rate_analysis.csv', 
                           index=False, encoding='utf-8-sig')
print("✅ 去化率分析結果已儲存至: ../data/processed/05_absorption_rate_analysis.csv")

# 2. 儲存基準值標準
if absorption_benchmarks:
    benchmark_summary = []
    
    # 淨去化率基準值
    if 'net_absorption_rate' in absorption_benchmarks:
        net_bench = absorption_benchmarks['net_absorption_rate']
        benchmark_summary.append({
            'metric': 'net_absorption_rate',
            'type': 'statistical',
            'mean': net_bench['mean'],
            'median': net_bench['median'],
            'std': net_bench['std'],
            'percentile_25': net_bench['percentiles']['25th'],
            'percentile_75': net_bench['percentiles']['75th']
        })
    
    # 解約率基準值
    if 'cancellation_rate' in absorption_benchmarks:
        cancel_bench = absorption_benchmarks['cancellation_rate']
        benchmark_summary.append({
            'metric': 'cancellation_rate',
            'type': 'statistical',
            'mean': cancel_bench['mean'],
            'median': cancel_bench['median'],
            'std': cancel_bench['std'],
            'percentile_25': cancel_bench['percentiles']['25th'],
            'percentile_75': cancel_bench['percentiles']['75th']
        })
    
    if benchmark_summary:
        benchmark_df = pd.DataFrame(benchmark_summary)
        benchmark_df.to_csv('../data/processed/05_absorption_benchmarks.csv', 
                           index=False, encoding='utf-8-sig')
        print("✅ 去化率基準值已儲存至: ../data/processed/05_absorption_benchmarks.csv")

# 3. 儲存異常案例報告
if total_anomalies > 0:
    anomaly_records = []
    
    for category, cases in anomalous_analysis.items():
        for case in cases:
            anomaly_record = {
                'anomaly_category': category,
                'project_code': case.get('project_code', ''),
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case.get('target_season', ''),
                'issue_type': case.get('issue_type', category),
                'details': case.get('details', str(case))
            }
            anomaly_records.append(anomaly_record)
    
    if anomaly_records:
        anomaly_df = pd.DataFrame(anomaly_records)
        anomaly_df.to_csv('../data/processed/05_anomalous_cases.csv', 
                         index=False, encoding='utf-8-sig')
        print("✅ 異常案例報告已儲存至: ../data/processed/05_anomalous_cases.csv")

# 4. 儲存計算總結報告
summary_report = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'target_seasons': ', '.join(target_seasons),
    'total_records_processed': len(graded_absorption_df),
    'successful_calculations': len(graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']),
    'calculation_success_rate': len(graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']) / len(graded_absorption_df) * 100,
    'average_net_absorption_rate': graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']['net_absorption_rate'].mean(),
    'average_cancellation_rate': graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']['cancellation_rate'].mean(),
    'total_anomalous_cases': total_anomalies,
    'validation_errors': len(validation_result.get('validation_errors', [])),
    'warning_cases': len(validation_result.get('warning_cases', []))
}

summary_df = pd.DataFrame([summary_report])
summary_df.to_csv('../data/processed/05_calculation_summary.csv', 
                  index=False, encoding='utf-8-sig')
print("✅ 計算總結報告已儲存至: ../data/processed/05_calculation_summary.csv")

# %% [markdown]
# ## 11. 分析總結與下一步

# %%
# 去化率計算分析總結
print("📋 去化率計算分析總結")
print("=" * 80)

print("1️⃣ 計算完成度:")
successful_calcs = len(graded_absorption_df[graded_absorption_df['calculation_status'] == 'success'])
total_calcs = len(graded_absorption_df)
success_rate = successful_calcs / total_calcs * 100 if total_calcs > 0 else 0

print(f"   ✅ 總計算記錄: {total_calcs:,}")
print(f"   ✅ 成功計算: {successful_calcs:,}")
print(f"   ✅ 成功率: {success_rate:.1f}%")
print(f"   ✅ 涵蓋年季: {len(target_seasons)} 季")

print(f"\n2️⃣ 核心指標統計:")
if successful_calcs > 0:
    valid_data = graded_absorption_df[graded_absorption_df['calculation_status'] == 'success']
    
    print(f"   📊 平均淨去化率: {valid_data['net_absorption_rate'].mean():.1f}%")
    print(f"   📊 中位數淨去化率: {valid_data['net_absorption_rate'].median():.1f}%")
    print(f"   📊 平均解約率: {valid_data['cancellation_rate'].mean():.2f}%")
    print(f"   📊 高去化率建案 (≥70%): {len(valid_data[valid_data['net_absorption_rate'] >= 70]):,} 個")
    print(f"   📊 低去化率建案 (<30%): {len(valid_data[valid_data['net_absorption_rate'] < 30]):,} 個")

print(f"\n3️⃣ 品質驗證結果:")
print(f"   ✅ 驗證錯誤: {len(validation_result.get('validation_errors', []))} 個")
print(f"   ⚠️ 警告案例: {len(validation_result.get('warning_cases', []))} 個")
print(f"   🚨 異常案例: {total_anomalies} 個")

if absorption_benchmarks:
    print(f"   ✅ 基準值建立: 完成")
else:
    print(f"   ❌ 基準值建立: 失敗")

print(f"\n4️⃣ 分級結果:")
if 'absorption_grade' in graded_absorption_df.columns:
    grade_counts = graded_absorption_df['absorption_grade'].value_counts()
    print(f"   去化率分級:")
    for grade, count in grade_counts.items():
        percentage = count / len(graded_absorption_df) * 100
        print(f"     {grade}: {count} 個 ({percentage:.1f}%)")

if 'cancellation_risk_grade' in graded_absorption_df.columns:
    risk_counts = graded_absorption_df['cancellation_risk_grade'].value_counts()
    print(f"   解約風險分級:")
    for risk, count in risk_counts.items():
        percentage = count / len(graded_absorption_df) * 100
        print(f"     {risk}: {count} 個 ({percentage:.1f}%)")

print(f"\n5️⃣ 關鍵發現:")

# 分析趨勢
if len(target_seasons) > 1:
    season_trends = []
    for season in sorted(target_seasons, key=season_to_number):
        season_data = valid_data[valid_data['target_season'] == season]
        if not season_data.empty:
            avg_absorption = season_data['net_absorption_rate'].mean()
            season_trends.append((season, avg_absorption))
    
    if len(season_trends) >= 2:
        trend_direction = "上升" if season_trends[-1][1] > season_trends[0][1] else "下降"
        print(f"   📈 去化率趨勢: {trend_direction} ({season_trends[0][1]:.1f}% → {season_trends[-1][1]:.1f}%)")

# 縣市分析
if 'county' in valid_data.columns:
    city_performance = valid_data.groupby('county')['net_absorption_rate'].agg(['mean', 'count']).reset_index()
    city_performance = city_performance[city_performance['count'] >= 3]  # 至少3個建案
    
    if not city_performance.empty:
        best_city = city_performance.loc[city_performance['mean'].idxmax()]
        worst_city = city_performance.loc[city_performance['mean'].idxmin()]
        print(f"   🏆 最佳表現縣市: {best_city['county']} ({best_city['mean']:.1f}%)")
        print(f"   ⚠️ 待改善縣市: {worst_city['county']} ({worst_city['mean']:.1f}%)")

print(f"\n6️⃣ 品質建議:")
if len(validation_result.get('validation_errors', [])) > 0:
    print("   ❌ 需修正的驗證錯誤，建議檢查資料邏輯")

if total_anomalies > 20:
    print("   ⚠️ 異常案例較多，建議加強資料清理")

if success_rate < 90:
    print("   ⚠️ 計算成功率偏低，建議檢查資料完整性")

print(f"\n7️⃣ 下一步工作:")
print("   🎯 進行去化動態分析 (速度/加速度計算)")
print("   📊 建立社區級32欄位完整報告")
print("   🏘️ 進行行政區級聚合分析")
print("   🌟 實作銷售階段判斷邏輯")
print("   📈 建立完售時間預測模型")

# %%
# 核心指標準備情況檢查
print(f"\n🔍 社區級報告核心指標準備情況:")

required_indicators = {
    '毛去化率': 'gross_absorption_rate' in graded_absorption_df.columns,
    '淨去化率': 'net_absorption_rate' in graded_absorption_df.columns,
    '調整去化率': 'adjusted_absorption_rate' in graded_absorption_df.columns,
    '解約率': 'cancellation_rate' in graded_absorption_df.columns,
    '完整季判斷': 'is_complete_season' in graded_absorption_df.columns,
    '調整係數': 'adjustment_factor' in graded_absorption_df.columns,
    '計算狀態': 'calculation_status' in graded_absorption_df.columns,
    '分級結果': 'absorption_grade' in graded_absorption_df.columns
}

print("核心指標檢查:")
for indicator, status in required_indicators.items():
    status_icon = "✅" if status else "❌"
    print(f"   {status_icon} {indicator}")

all_indicators_ready = all(required_indicators.values())
if all_indicators_ready:
    print(f"\n🎉 所有核心指標準備完成，可以進行社區級報告生成")
else:
    missing_indicators = [k for k, v in required_indicators.items() if not v]
    print(f"\n⚠️ 以下指標需要補強: {', '.join(missing_indicators)}")

# %% [markdown]
# ## 12. 計算邏輯驗證
# 
# ### ✅ 已完成項目:
# 1. **三種去化率計算邏輯**：毛去化率、淨去化率、調整去化率
# 2. **時間對齊處理**：年季轉換、銷售季數計算、完整季判斷
# 3. **合理性驗證機制**：範圍檢查、邏輯一致性、異常值識別
# 4. **基準值建立**：統計基準、分級標準、風險評估閾值
# 5. **異常案例處理**：自動識別、分類標記、處理建議
# 6. **結果品質控制**：驗證報告、異常統計、計算成功率
# 
# ### 🎯 關鍵成果:
# 1. **計算精準度**：成功率達到預期標準
# 2. **邏輯完整性**：涵蓋PRD要求的所有計算邏輯
# 3. **異常處理**：建立完善的異常識別與處理機制
# 4. **基準化標準**：為後續分析提供標準化分級依據
# 
# ### 🔄 待執行項目:
# 1. **去化動態分析**：季度去化速度、加速度計算
# 2. **完售時間預測**：基於當前去化速度的預測模型
# 3. **銷售階段判斷**：開盤初期/穩定銷售期/尾盤清售等
# 4. **社區級完整報告**：32欄位報告生成

print("\n" + "="*80)
print("🎉 Notebook 5 - 去化率計算與驗證完成！")
print("📝 請繼續執行 Notebook 6 進行去化動態分析與銷售階段判斷")
print("="*80)