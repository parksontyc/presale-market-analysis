# 預售屋市場分析系統 - 06_去化動態分析與效率評級
# 基於 PRD v2.3 規格進行去化速度、加速度與完售預測分析
# ================================================================================

# %% [markdown]
# # 預售屋市場分析系統 - 去化動態分析與效率評級
# 
# ## 📋 目標
# - ✅ 實作季度去化速度計算邏輯
# - ✅ 建立去化加速度分析機制
# - ✅ 實作完售時間預測算法
# - ✅ 建立去化效率評級系統
# - ✅ 建立去化動態基準值標準
# - ✅ 識別異常去化模式
# - ✅ 為社區級報告提供動態指標
# 
# ## 🎯 內容大綱
# 1. 季度去化速度計算實作
# 2. 去化加速度分析邏輯
# 3. 預估完售時間算法
# 4. 去化效率評級邏輯
# 5. 去化動態趨勢分析
# 6. 異常去化模式識別
# 7. 去化動態基準值建立
# 8. 動態指標視覺化分析
# 
# ## 📊 延續 Notebook 1-5 的分析結果
# - 乾淨交易資料: 去重後的有效交易記錄
# - 解約分析結果: 解約資料解析與風險評估
# - 建案整合結果: 活躍建案識別與滯銷標記
# - 去化率計算結果: 毛/淨/調整去化率與分級

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
from scipy import stats
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
    # 載入去化率計算結果
    absorption_analysis = pd.read_csv('../data/processed/05_absorption_rate_analysis.csv', encoding='utf-8')
    print(f"✅ 去化率分析結果: {absorption_analysis.shape}")
    
    # 載入基準值資料
    absorption_benchmarks = pd.read_csv('../data/processed/05_absorption_benchmarks.csv', encoding='utf-8')
    print(f"✅ 去化率基準值: {absorption_benchmarks.shape}")
    
    # 載入乾淨的交易資料（用於時間序列分析）
    clean_transactions = pd.read_csv('../data/processed/03_clean_transactions.csv', encoding='utf-8')
    print(f"✅ 乾淨交易資料: {clean_transactions.shape}")
    
    # 載入活躍建案資料
    active_projects = pd.read_csv('../data/processed/04_active_projects_analysis.csv', encoding='utf-8')
    print(f"✅ 活躍建案分析: {active_projects.shape}")
    
except FileNotFoundError as e:
    print(f"❌ 檔案載入失敗: {e}")
    print("📝 請確認是否已執行 Notebook 1-5")
except Exception as e:
    print(f"❌ 載入過程發生錯誤: {e}")

# %%
# 資料概況檢視
print("📊 去化動態分析基礎資料概況")
print("=" * 60)

print("去化率分析結果:")
print(f"   總記錄數: {len(absorption_analysis):,}")
print(f"   成功計算數: {len(absorption_analysis[absorption_analysis['calculation_status'] == 'success']):,}")
print(f"   涵蓋建案數: {absorption_analysis['project_code'].nunique():,}")
print(f"   涵蓋年季數: {absorption_analysis['target_season'].nunique():,}")

valid_absorption = absorption_analysis[absorption_analysis['calculation_status'] == 'success']
if not valid_absorption.empty:
    print(f"   平均淨去化率: {valid_absorption['net_absorption_rate'].mean():.1f}%")
    print(f"   有去化動態數據的記錄: {len(valid_absorption[valid_absorption['net_absorption_rate'] > 0]):,}")

print(f"\n交易資料概況:")
print(f"   總交易筆數: {len(clean_transactions):,}")
print(f"   年季範圍: {clean_transactions['交易年季'].min()} ~ {clean_transactions['交易年季'].max()}")
print(f"   備查編號數: {clean_transactions['備查編號'].nunique():,}")

# 確認關鍵欄位存在
required_columns = ['project_code', 'target_season', 'net_absorption_rate', 'total_units']
missing_columns = [col for col in required_columns if col not in absorption_analysis.columns]
if missing_columns:
    print(f"⚠️ 缺少關鍵欄位: {missing_columns}")
else:
    print(f"✅ 所有必要欄位都存在")

# %% [markdown]
# ## 2. 年季處理工具函數擴展

# %%
# 年季處理工具函數（擴展版）
print("🕐 年季處理工具函數擴展")
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

def get_previous_season(current_season):
    """
    獲取前一個年季
    例: "111Y2S" -> "111Y1S", "111Y1S" -> "110Y4S"
    """
    try:
        current_num = season_to_number(current_season)
        if current_num <= 0:
            return ""
        
        year = current_num // 10
        season = current_num % 10
        
        if season == 1:  # 第1季 -> 前一年第4季
            prev_num = (year - 1) * 10 + 4
        else:  # 季度-1
            prev_num = year * 10 + (season - 1)
        
        return number_to_season(prev_num)
    except:
        return ""

def get_next_season(current_season):
    """
    獲取下一個年季
    """
    try:
        current_num = season_to_number(current_season)
        if current_num <= 0:
            return ""
        
        year = current_num // 10
        season = current_num % 10
        
        if season == 4:  # 第4季 -> 下一年第1季
            next_num = (year + 1) * 10 + 1
        else:  # 季度+1
            next_num = year * 10 + (season + 1)
        
        return number_to_season(next_num)
    except:
        return ""

def get_season_sequence(start_season, end_season):
    """
    獲取從開始到結束的所有年季序列
    """
    seasons = []
    current = start_season
    
    while season_to_number(current) <= season_to_number(end_season):
        seasons.append(current)
        current = get_next_season(current)
        
        if current == "" or len(seasons) > 100:  # 防止無限迴圈
            break
    
    return seasons

# %%
# 測試年季處理函數
print("🧪 年季處理函數測試:")

test_cases = [
    ("111Y2S", "111Y1S", "111Y3S"),
    ("111Y1S", "110Y4S", "111Y2S"),
    ("113Y4S", "113Y3S", "114Y1S"),
    ("110Y3S", ["110Y3S", "110Y4S", "111Y1S"], "112Y1S")  # 序列測試
]

for current, expected_prev, expected_next in test_cases[:3]:
    prev_result = get_previous_season(current)
    next_result = get_next_season(current)
    prev_status = "✅" if prev_result == expected_prev else "❌"
    next_status = "✅" if next_result == expected_next else "❌"
    print(f"   {prev_status} {expected_prev} <- {current} -> {next_result} {next_status}")

# 序列測試
seq_result = get_season_sequence("110Y3S", "111Y2S")
expected_seq = ["110Y3S", "110Y4S", "111Y1S", "111Y2S"]
seq_status = "✅" if seq_result == expected_seq else "❌"
print(f"   {seq_status} 序列測試: {seq_result}")

# %% [markdown]
# ## 3. 季度去化速度計算實作

# %%
# 季度去化速度計算邏輯
print("📈 季度去化速度計算實作")
print("=" * 60)

def calculate_quarterly_absorption_speed(project_code, target_season, absorption_df, method='net_units'):
    """
    計算季度去化速度
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        absorption_df: 去化率分析結果
        method: 計算方法 ('net_units' 或 'absorption_rate')
        
    Returns:
        dict: 季度去化速度計算結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'quarterly_absorption_speed': 0.0,
        'speed_calculation_method': method,
        'previous_season': '',
        'current_net_absorption_rate': 0.0,
        'previous_net_absorption_rate': 0.0,
        'total_units': 0,
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取當前季度資料
        current_data = absorption_df[
            (absorption_df['project_code'] == project_code) & 
            (absorption_df['target_season'] == target_season) &
            (absorption_df['calculation_status'] == 'success')
        ]
        
        if current_data.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到當前季度資料'
            return result
        
        current_row = current_data.iloc[0]
        result['current_net_absorption_rate'] = current_row['net_absorption_rate']
        result['total_units'] = current_row['total_units']
        
        # 獲取前一季度資料
        previous_season = get_previous_season(target_season)
        result['previous_season'] = previous_season
        
        if previous_season:
            previous_data = absorption_df[
                (absorption_df['project_code'] == project_code) & 
                (absorption_df['target_season'] == previous_season) &
                (absorption_df['calculation_status'] == 'success')
            ]
            
            if not previous_data.empty:
                previous_row = previous_data.iloc[0]
                result['previous_net_absorption_rate'] = previous_row['net_absorption_rate']
                
                # 計算去化速度
                if method == 'net_units':
                    # 方法1: 以實際戶數變化計算
                    current_net_units = (current_row['net_absorption_rate'] / 100) * current_row['total_units']
                    previous_net_units = (previous_row['net_absorption_rate'] / 100) * previous_row['total_units']
                    quarterly_speed = max(0, current_net_units - previous_net_units)  # 不允許負值
                    
                elif method == 'absorption_rate':
                    # 方法2: 以去化率變化計算
                    rate_change = current_row['net_absorption_rate'] - previous_row['net_absorption_rate']
                    quarterly_speed = max(0, (rate_change / 100) * current_row['total_units'])
                
                result['quarterly_absorption_speed'] = round(quarterly_speed, 2)
                
            else:
                # 前一季沒有資料，使用平均化方法估算
                if 'sales_seasons' in current_row.index or '銷售季數' in current_row.index:
                    sales_seasons = current_row.get('sales_seasons', current_row.get('銷售季數', 1))
                    if sales_seasons > 0:
                        current_net_units = (current_row['net_absorption_rate'] / 100) * current_row['total_units']
                        estimated_speed = current_net_units / sales_seasons
                        result['quarterly_absorption_speed'] = round(max(0, estimated_speed), 2)
        else:
            # 第一季，使用當前去化率估算
            current_net_units = (current_row['net_absorption_rate'] / 100) * current_row['total_units']
            result['quarterly_absorption_speed'] = round(max(0, current_net_units), 2)
    
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算季度去化速度
print("🔄 批量計算季度去化速度...")

# 獲取所有可用的年季並排序
available_seasons = sorted(absorption_analysis['target_season'].unique(), key=season_to_number)
print(f"   可用年季: {available_seasons}")

quarterly_speed_results = []

# 為每個建案的每個年季計算去化速度
for project_code in absorption_analysis['project_code'].unique():
    project_seasons = absorption_analysis[
        (absorption_analysis['project_code'] == project_code) & 
        (absorption_analysis['calculation_status'] == 'success')
    ]['target_season'].unique()
    
    # 按年季順序處理
    sorted_seasons = sorted(project_seasons, key=season_to_number)
    
    for season in sorted_seasons:
        result = calculate_quarterly_absorption_speed(
            project_code, season, absorption_analysis, method='net_units'
        )
        
        # 添加建案基本資訊
        project_info = absorption_analysis[
            (absorption_analysis['project_code'] == project_code) & 
            (absorption_analysis['target_season'] == season)
        ].iloc[0]
        
        result.update({
            'county': project_info.get('county', ''),
            'district': project_info.get('district', ''),
            'project_name': project_info.get('project_name', ''),
            'has_complete_info': project_info.get('has_complete_info', False)
        })
        
        quarterly_speed_results.append(result)

# 轉換為DataFrame
quarterly_speed_df = pd.DataFrame(quarterly_speed_results)

print(f"✅ 完成 {len(quarterly_speed_df)} 筆季度去化速度計算")

# %%
# 季度去化速度統計分析
print(f"\n📊 季度去化速度統計分析:")

if not quarterly_speed_df.empty:
    # 成功計算統計
    successful_speed_calcs = quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_speed_calcs):,} 筆")
    print(f"   計算成功率: {len(successful_speed_calcs)/len(quarterly_speed_df)*100:.1f}%")
    
    if not successful_speed_calcs.empty:
        # 去化速度分布統計
        speeds = successful_speed_calcs['quarterly_absorption_speed']
        
        print(f"\n去化速度分布統計:")
        print(f"   平均去化速度: {speeds.mean():.2f} 戶/季")
        print(f"   中位數去化速度: {speeds.median():.2f} 戶/季")
        print(f"   最高去化速度: {speeds.max():.2f} 戶/季")
        print(f"   標準差: {speeds.std():.2f} 戶/季")
        
        # 速度分級統計
        high_speed = len(speeds[speeds >= 5])
        medium_speed = len(speeds[(speeds >= 2) & (speeds < 5)])
        low_speed = len(speeds[(speeds >= 0.5) & (speeds < 2)])
        stagnant = len(speeds[speeds < 0.5])
        
        print(f"\n去化速度分級:")
        print(f"   高速去化 (≥5戶/季): {high_speed} 個 ({high_speed/len(speeds)*100:.1f}%)")
        print(f"   中速去化 (2-5戶/季): {medium_speed} 個 ({medium_speed/len(speeds)*100:.1f}%)")
        print(f"   低速去化 (0.5-2戶/季): {low_speed} 個 ({low_speed/len(speeds)*100:.1f}%)")
        print(f"   滯銷狀態 (<0.5戶/季): {stagnant} 個 ({stagnant/len(speeds)*100:.1f}%)")
        
        # 各年季速度趨勢
        print(f"\n各年季去化速度趨勢:")
        for season in available_seasons:
            season_data = successful_speed_calcs[successful_speed_calcs['target_season'] == season]
            if not season_data.empty:
                avg_speed = season_data['quarterly_absorption_speed'].mean()
                record_count = len(season_data)
                print(f"   {season}: 平均 {avg_speed:.2f} 戶/季 ({record_count} 個建案)")

# %% [markdown]
# ## 4. 去化加速度分析實作

# %%
# 去化加速度計算邏輯
print("🚀 去化加速度分析實作")
print("=" * 60)

def calculate_absorption_acceleration(project_code, target_season, speed_df):
    """
    計算去化加速度
    
    去化加速度 = (本季去化速度 - 上季去化速度) / 上季去化速度 × 100%
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        speed_df: 季度去化速度計算結果
        
    Returns:
        dict: 去化加速度計算結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'absorption_acceleration': 0.0,
        'current_speed': 0.0,
        'previous_speed': 0.0,
        'acceleration_status': 'stable',
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取當前季度速度
        current_data = speed_df[
            (speed_df['project_code'] == project_code) & 
            (speed_df['target_season'] == target_season) &
            (speed_df['calculation_status'] == 'success')
        ]
        
        if current_data.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到當前季度速度資料'
            return result
        
        current_speed = current_data.iloc[0]['quarterly_absorption_speed']
        result['current_speed'] = current_speed
        
        # 獲取前一季度速度
        previous_season = get_previous_season(target_season)
        
        if previous_season:
            previous_data = speed_df[
                (speed_df['project_code'] == project_code) & 
                (speed_df['target_season'] == previous_season) &
                (speed_df['calculation_status'] == 'success')
            ]
            
            if not previous_data.empty:
                previous_speed = previous_data.iloc[0]['quarterly_absorption_speed']
                result['previous_speed'] = previous_speed
                
                # 計算加速度
                if previous_speed > 0:
                    acceleration = ((current_speed - previous_speed) / previous_speed) * 100
                    result['absorption_acceleration'] = round(acceleration, 2)
                    
                    # 判斷加速度狀態
                    if acceleration > 20:
                        result['acceleration_status'] = 'strong_acceleration'
                    elif acceleration > 5:
                        result['acceleration_status'] = 'acceleration'
                    elif acceleration > -5:
                        result['acceleration_status'] = 'stable'
                    elif acceleration > -20:
                        result['acceleration_status'] = 'deceleration'
                    else:
                        result['acceleration_status'] = 'strong_deceleration'
                else:
                    # 前一季速度為0，特殊處理
                    if current_speed > 0:
                        result['acceleration_status'] = 'restart'
                        result['absorption_acceleration'] = 999.0  # 標記為重啟
                    else:
                        result['acceleration_status'] = 'stagnant'
                        result['absorption_acceleration'] = 0.0
            else:
                # 前一季沒有資料
                result['acceleration_status'] = 'initial'
                result['absorption_acceleration'] = 0.0
        else:
            # 第一季
            result['acceleration_status'] = 'initial'
            result['absorption_acceleration'] = 0.0
    
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算去化加速度
print("🔄 批量計算去化加速度...")

acceleration_results = []

# 對所有有速度資料的記錄計算加速度
for _, speed_row in quarterly_speed_df.iterrows():
    if speed_row['calculation_status'] != 'success':
        continue
    
    result = calculate_absorption_acceleration(
        speed_row['project_code'], 
        speed_row['target_season'], 
        quarterly_speed_df
    )
    
    # 添加基本資訊
    result.update({
        'county': speed_row['county'],
        'district': speed_row['district'],
        'project_name': speed_row['project_name'],
        'has_complete_info': speed_row['has_complete_info']
    })
    
    acceleration_results.append(result)

# 轉換為DataFrame
acceleration_df = pd.DataFrame(acceleration_results)

print(f"✅ 完成 {len(acceleration_df)} 筆去化加速度計算")

# %%
# 去化加速度統計分析
print(f"\n📊 去化加速度統計分析:")

if not acceleration_df.empty:
    successful_accel_calcs = acceleration_df[acceleration_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_accel_calcs):,} 筆")
    
    if not successful_accel_calcs.empty:
        # 加速度狀態分布
        status_distribution = successful_accel_calcs['acceleration_status'].value_counts()
        print(f"\n加速度狀態分布:")
        for status, count in status_distribution.items():
            percentage = count / len(successful_accel_calcs) * 100
            print(f"   {status}: {count} 個 ({percentage:.1f}%)")
        
        # 加速度數值統計（排除特殊值）
        numeric_acceleration = successful_accel_calcs[
            (successful_accel_calcs['absorption_acceleration'] != 999.0) &
            (successful_accel_calcs['absorption_acceleration'].abs() <= 200)  # 過濾極端值
        ]['absorption_acceleration']
        
        if not numeric_acceleration.empty:
            print(f"\n加速度數值統計:")
            print(f"   平均加速度: {numeric_acceleration.mean():.1f}%")
            print(f"   中位數加速度: {numeric_acceleration.median():.1f}%")
            print(f"   標準差: {numeric_acceleration.std():.1f}%")
            print(f"   最大加速度: {numeric_acceleration.max():.1f}%")
            print(f"   最小加速度: {numeric_acceleration.min():.1f}%")
        
        # 各年季加速度趨勢
        print(f"\n各年季加速度趨勢:")
        for season in available_seasons:
            season_data = successful_accel_calcs[successful_accel_calcs['target_season'] == season]
            if not season_data.empty and len(season_data) >= 5:  # 至少5個樣本
                season_numeric = season_data[
                    (season_data['absorption_acceleration'] != 999.0) &
                    (season_data['absorption_acceleration'].abs() <= 200)
                ]['absorption_acceleration']
                
                if not season_numeric.empty:
                    avg_acceleration = season_numeric.mean()
                    positive_count = len(season_numeric[season_numeric > 0])
                    total_count = len(season_numeric)
                    print(f"   {season}: 平均 {avg_acceleration:+.1f}% ({positive_count}/{total_count} 加速)")

# %% [markdown]
# ## 5. 預估完售時間計算實作

# %%
# 預估完售時間計算邏輯
print("⏰ 預估完售時間計算實作")
print("=" * 60)

def calculate_estimated_completion_time(project_code, target_season, absorption_df, speed_df, method='current_speed'):
    """
    計算預估完售時間
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        absorption_df: 去化率分析結果
        speed_df: 季度去化速度結果
        method: 預測方法 ('current_speed', 'average_speed', 'trend_based')
        
    Returns:
        dict: 預估完售時間計算結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'estimated_completion_seasons': 0,
        'estimated_completion_season': '',
        'prediction_method': method,
        'current_absorption_rate': 0.0,
        'remaining_absorption_rate': 0.0,
        'current_speed': 0.0,
        'prediction_confidence': 'low',
        'completion_status': 'unknown',
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取當前去化率資料
        current_absorption = absorption_df[
            (absorption_df['project_code'] == project_code) & 
            (absorption_df['target_season'] == target_season) &
            (absorption_df['calculation_status'] == 'success')
        ]
        
        if current_absorption.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到當前去化率資料'
            return result
        
        absorption_row = current_absorption.iloc[0]
        current_rate = absorption_row['net_absorption_rate']
        total_units = absorption_row['total_units']
        
        result['current_absorption_rate'] = current_rate
        result['remaining_absorption_rate'] = max(0, 100 - current_rate)
        
        # 檢查是否已完售
        if current_rate >= 100:
            result['completion_status'] = 'completed'
            result['estimated_completion_seasons'] = 0
            result['estimated_completion_season'] = target_season
            result['prediction_confidence'] = 'high'
            return result
        
        # 獲取當前去化速度
        current_speed_data = speed_df[
            (speed_df['project_code'] == project_code) & 
            (speed_df['target_season'] == target_season) &
            (speed_df['calculation_status'] == 'success')
        ]
        
        if current_speed_data.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到當前去化速度資料'
            return result
        
        current_speed = current_speed_data.iloc[0]['quarterly_absorption_speed']
        result['current_speed'] = current_speed
        
        # 計算剩餘戶數
        remaining_units = (result['remaining_absorption_rate'] / 100) * total_units
        
        if method == 'current_speed':
            # 方法1: 基於當前速度預測
            if current_speed > 0:
                estimated_seasons = math.ceil(remaining_units / current_speed)
                result['prediction_confidence'] = 'medium' if current_speed >= 1 else 'low'
            else:
                estimated_seasons = 999  # 無法預測
                result['completion_status'] = 'stagnant'
                result['prediction_confidence'] = 'very_low'
        
        elif method == 'average_speed':
            # 方法2: 基於平均速度預測
            project_speeds = speed_df[
                (speed_df['project_code'] == project_code) &
                (speed_df['calculation_status'] == 'success')
            ]['quarterly_absorption_speed']
            
            if not project_speeds.empty and project_speeds.mean() > 0:
                avg_speed = project_speeds.mean()
                estimated_seasons = math.ceil(remaining_units / avg_speed)
                result['prediction_confidence'] = 'high' if len(project_speeds) >= 3 else 'medium'
            else:
                estimated_seasons = 999
                result['completion_status'] = 'stagnant'
        
        elif method == 'trend_based':
            # 方法3: 基於趨勢預測（簡化版）
            project_speeds = speed_df[
                (speed_df['project_code'] == project_code) &
                (speed_df['calculation_status'] == 'success')
            ].sort_values('target_season', key=lambda x: x.map(season_to_number))
            
            if len(project_speeds) >= 2:
                recent_speeds = project_speeds['quarterly_absorption_speed'].tail(2).values
                if len(recent_speeds) == 2 and recent_speeds[0] > 0:
                    # 簡單趨勢預測
                    trend_speed = recent_speeds[1] + (recent_speeds[1] - recent_speeds[0])
                    trend_speed = max(0.1, trend_speed)  # 最低0.1戶/季
                    estimated_seasons = math.ceil(remaining_units / trend_speed)
                    result['prediction_confidence'] = 'medium'
                else:
                    estimated_seasons = 999
            else:
                # 回到當前速度方法
                if current_speed > 0:
                    estimated_seasons = math.ceil(remaining_units / current_speed)
                else:
                    estimated_seasons = 999
        
        # 設定預估完售季數
        result['estimated_completion_seasons'] = min(estimated_seasons, 999)
        
        # 計算預估完售年季
        if estimated_seasons < 999:
            current_season_num = season_to_number(target_season)
            estimated_season_num = current_season_num
            
            for _ in range(estimated_seasons):
                estimated_season_num = season_to_number(get_next_season(number_to_season(estimated_season_num)))
                if estimated_season_num == 0:
                    break
            
            result['estimated_completion_season'] = number_to_season(estimated_season_num)
            
            # 設定完售狀態
            if estimated_seasons <= 4:
                result['completion_status'] = 'fast_completion'
            elif estimated_seasons <= 8:
                result['completion_status'] = 'normal_completion'
            elif estimated_seasons <= 16:
                result['completion_status'] = 'slow_completion'
            else:
                result['completion_status'] = 'long_term_sales'
        else:
            result['completion_status'] = 'unpredictable'
            result['estimated_completion_season'] = ''
    
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算預估完售時間
print("🔄 批量計算預估完售時間...")

completion_results = []

# 對所有有效記錄計算預估完售時間
for _, absorption_row in absorption_analysis.iterrows():
    if absorption_row['calculation_status'] != 'success':
        continue
    
    # 使用三種方法計算
    methods = ['current_speed', 'average_speed', 'trend_based']
    
    for method in methods:
        result = calculate_estimated_completion_time(
            absorption_row['project_code'], 
            absorption_row['target_season'], 
            absorption_analysis,
            quarterly_speed_df,
            method=method
        )
        
        # 添加基本資訊
        result.update({
            'county': absorption_row.get('county', ''),
            'district': absorption_row.get('district', ''),
            'project_name': absorption_row.get('project_name', ''),
            'total_units': absorption_row.get('total_units', 0),
            'has_complete_info': absorption_row.get('has_complete_info', False)
        })
        
        completion_results.append(result)

# 轉換為DataFrame
completion_df = pd.DataFrame(completion_results)

print(f"✅ 完成 {len(completion_df)} 筆預估完售時間計算")

# %%
# 預估完售時間統計分析
print(f"\n📊 預估完售時間統計分析:")

if not completion_df.empty:
    successful_completion_calcs = completion_df[completion_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_completion_calcs):,} 筆")
    
    if not successful_completion_calcs.empty:
        # 各預測方法統計
        for method in ['current_speed', 'average_speed', 'trend_based']:
            method_data = successful_completion_calcs[successful_completion_calcs['prediction_method'] == method]
            
            if not method_data.empty:
                print(f"\n{method} 方法統計:")
                
                # 完售狀態分布
                status_dist = method_data['completion_status'].value_counts()
                for status, count in status_dist.items():
                    percentage = count / len(method_data) * 100
                    print(f"   {status}: {count} 個 ({percentage:.1f}%)")
                
                # 預估季數統計（排除999）
                valid_predictions = method_data[method_data['estimated_completion_seasons'] < 999]
                if not valid_predictions.empty:
                    avg_seasons = valid_predictions['estimated_completion_seasons'].mean()
                    median_seasons = valid_predictions['estimated_completion_seasons'].median()
                    print(f"   平均預估完售季數: {avg_seasons:.1f} 季")
                    print(f"   中位數預估完售季數: {median_seasons:.1f} 季")
                    
                    # 完售時間分級
                    fast_completion = len(valid_predictions[valid_predictions['estimated_completion_seasons'] <= 4])
                    normal_completion = len(valid_predictions[(valid_predictions['estimated_completion_seasons'] > 4) & 
                                                             (valid_predictions['estimated_completion_seasons'] <= 8)])
                    slow_completion = len(valid_predictions[valid_predictions['estimated_completion_seasons'] > 8])
                    
                    print(f"   快速完售 (≤4季): {fast_completion} 個")
                    print(f"   正常完售 (5-8季): {normal_completion} 個")
                    print(f"   緩慢完售 (>8季): {slow_completion} 個")

# %% [markdown]
# ## 6. 去化效率評級邏輯實作

# %%
# 去化效率評級邏輯
print("⭐ 去化效率評級邏輯實作")
print("=" * 60)

def evaluate_absorption_efficiency(project_code, target_season, absorption_df, speed_df, completion_df):
    """
    評估去化效率等級
    
    綜合考慮：去化率、去化速度、預估完售時間、銷售季數
    
    Args:
        project_code: 建案編號
        target_season: 目標年季
        absorption_df: 去化率分析結果
        speed_df: 去化速度結果
        completion_df: 完售預測結果
        
    Returns:
        dict: 去化效率評級結果
    """
    
    result = {
        'project_code': project_code,
        'target_season': target_season,
        'efficiency_grade': 'unknown',
        'efficiency_score': 0.0,
        'absorption_score': 0.0,
        'speed_score': 0.0,
        'completion_score': 0.0,
        'time_score': 0.0,
        'grade_emoji': '❓',
        'grade_description': '',
        'calculation_status': 'success',
        'error_message': ''
    }
    
    try:
        # 獲取去化率資料
        absorption_data = absorption_df[
            (absorption_df['project_code'] == project_code) & 
            (absorption_df['target_season'] == target_season) &
            (absorption_df['calculation_status'] == 'success')
        ]
        
        if absorption_data.empty:
            result['calculation_status'] = 'error'
            result['error_message'] = '找不到去化率資料'
            return result
        
        absorption_row = absorption_data.iloc[0]
        net_absorption_rate = absorption_row['net_absorption_rate']
        
        # 獲取去化速度資料
        speed_data = speed_df[
            (speed_df['project_code'] == project_code) & 
            (speed_df['target_season'] == target_season) &
            (speed_df['calculation_status'] == 'success')
        ]
        
        quarterly_speed = 0.0
        if not speed_data.empty:
            quarterly_speed = speed_data.iloc[0]['quarterly_absorption_speed']
        
        # 獲取預估完售時間（使用current_speed方法）
        completion_data = completion_df[
            (completion_df['project_code'] == project_code) & 
            (completion_df['target_season'] == target_season) &
            (completion_df['prediction_method'] == 'current_speed') &
            (completion_df['calculation_status'] == 'success')
        ]
        
        estimated_seasons = 999
        if not completion_data.empty:
            estimated_seasons = completion_data.iloc[0]['estimated_completion_seasons']
        
        # 計算銷售季數（從建案資料或推算）
        sales_seasons = 1
        if 'sales_seasons' in absorption_row.index:
            sales_seasons = max(1, absorption_row['sales_seasons'])
        elif 'start_season' in absorption_row.index:
            # 從銷售起始年季推算
            start_season = absorption_row['start_season']
            if start_season:
                sales_seasons = len(get_season_sequence(start_season, target_season))
        
        # 評分計算（總分100分）
        
        # 1. 去化率評分 (0-30分)
        if net_absorption_rate >= 100:
            absorption_score = 30  # 已完售
        elif net_absorption_rate >= 80:
            absorption_score = 25
        elif net_absorption_rate >= 60:
            absorption_score = 20
        elif net_absorption_rate >= 40:
            absorption_score = 15
        elif net_absorption_rate >= 20:
            absorption_score = 10
        else:
            absorption_score = max(0, net_absorption_rate / 20 * 10)
        
        # 2. 去化速度評分 (0-25分)
        if quarterly_speed >= 5:
            speed_score = 25
        elif quarterly_speed >= 3:
            speed_score = 20
        elif quarterly_speed >= 2:
            speed_score = 15
        elif quarterly_speed >= 1:
            speed_score = 10
        elif quarterly_speed >= 0.5:
            speed_score = 5
        else:
            speed_score = 0
        
        # 3. 預估完售時間評分 (0-25分)
        if net_absorption_rate >= 100:
            completion_score = 25  # 已完售
        elif estimated_seasons <= 4:
            completion_score = 25
        elif estimated_seasons <= 8:
            completion_score = 20
        elif estimated_seasons <= 12:
            completion_score = 15
        elif estimated_seasons <= 20:
            completion_score = 10
        elif estimated_seasons < 999:
            completion_score = 5
        else:
            completion_score = 0  # 無法預測
        
        # 4. 銷售時間效率評分 (0-20分)
        if net_absorption_rate >= 100:
            # 已完售建案根據銷售時間評分
            if sales_seasons <= 4:
                time_score = 20  # 快速完售
            elif sales_seasons <= 8:
                time_score = 15  # 正常完售
            elif sales_seasons <= 12:
                time_score = 10  # 較慢完售
            else:
                time_score = 5   # 緩慢完售
        else:
            # 未完售建案根據當前進度評分
            if sales_seasons <= 4:
                expected_rate = net_absorption_rate / sales_seasons * 4  # 推算4季應有進度
                time_score = min(20, max(0, expected_rate / 50 * 20))
            else:
                time_score = max(0, 20 - (sales_seasons - 4) * 2)
        
        # 總評分
        total_score = absorption_score + speed_score + completion_score + time_score
        
        result.update({
            'efficiency_score': round(total_score, 1),
            'absorption_score': round(absorption_score, 1),
            'speed_score': round(speed_score, 1),
            'completion_score': round(completion_score, 1),
            'time_score': round(time_score, 1)
        })
        
        # 等級判定
        if total_score >= 85:
            result.update({
                'efficiency_grade': 'excellent',
                'grade_emoji': '🚀',
                'grade_description': '高效去化'
            })
        elif total_score >= 70:
            result.update({
                'efficiency_grade': 'good',
                'grade_emoji': '⭐',
                'grade_description': '正常去化'
            })
        elif total_score >= 50:
            result.update({
                'efficiency_grade': 'average',
                'grade_emoji': '⚠️',
                'grade_description': '緩慢去化'
            })
        else:
            result.update({
                'efficiency_grade': 'poor',
                'grade_emoji': '🐌',
                'grade_description': '滯銷狀態'
            })
    
    except Exception as e:
        result['calculation_status'] = 'error'
        result['error_message'] = str(e)
    
    return result

# %%
# 批量計算去化效率評級
print("🔄 批量計算去化效率評級...")

efficiency_results = []

# 對所有有效記錄計算效率評級
unique_records = absorption_analysis[absorption_analysis['calculation_status'] == 'success'][
    ['project_code', 'target_season']
].drop_duplicates()

for _, record in unique_records.iterrows():
    result = evaluate_absorption_efficiency(
        record['project_code'],
        record['target_season'],
        absorption_analysis,
        quarterly_speed_df,
        completion_df
    )
    
    # 添加基本資訊
    absorption_info = absorption_analysis[
        (absorption_analysis['project_code'] == record['project_code']) & 
        (absorption_analysis['target_season'] == record['target_season'])
    ].iloc[0]
    
    result.update({
        'county': absorption_info.get('county', ''),
        'district': absorption_info.get('district', ''),
        'project_name': absorption_info.get('project_name', ''),
        'total_units': absorption_info.get('total_units', 0),
        'net_absorption_rate': absorption_info.get('net_absorption_rate', 0),
        'has_complete_info': absorption_info.get('has_complete_info', False)
    })
    
    efficiency_results.append(result)

# 轉換為DataFrame
efficiency_df = pd.DataFrame(efficiency_results)

print(f"✅ 完成 {len(efficiency_df)} 筆去化效率評級")

# %%
# 去化效率評級統計分析
print(f"\n📊 去化效率評級統計分析:")

if not efficiency_df.empty:
    successful_efficiency = efficiency_df[efficiency_df['calculation_status'] == 'success']
    
    print(f"計算結果統計:")
    print(f"   成功計算: {len(successful_efficiency):,} 筆")
    
    if not successful_efficiency.empty:
        # 效率等級分布
        grade_distribution = successful_efficiency['efficiency_grade'].value_counts()
        print(f"\n效率等級分布:")
        
        grade_order = ['excellent', 'good', 'average', 'poor']
        for grade in grade_order:
            if grade in grade_distribution.index:
                count = grade_distribution[grade]
                percentage = count / len(successful_efficiency) * 100
                # 取得對應的emoji
                sample_record = successful_efficiency[successful_efficiency['efficiency_grade'] == grade].iloc[0]
                emoji = sample_record['grade_emoji']
                description = sample_record['grade_description']
                print(f"   {emoji} {description} ({grade}): {count} 個 ({percentage:.1f}%)")
        
        # 評分統計
        print(f"\n評分統計:")
        print(f"   平均總分: {successful_efficiency['efficiency_score'].mean():.1f}")
        print(f"   中位數總分: {successful_efficiency['efficiency_score'].median():.1f}")
        print(f"   標準差: {successful_efficiency['efficiency_score'].std():.1f}")
        
        # 各分項評分平均
        print(f"\n各分項平均評分:")
        print(f"   去化率評分: {successful_efficiency['absorption_score'].mean():.1f}/30")
        print(f"   去化速度評分: {successful_efficiency['speed_score'].mean():.1f}/25")
        print(f"   完售預測評分: {successful_efficiency['completion_score'].mean():.1f}/25")
        print(f"   時間效率評分: {successful_efficiency['time_score'].mean():.1f}/20")
        
        # 縣市別效率分析
        if 'county' in successful_efficiency.columns:
            city_efficiency = successful_efficiency.groupby('county').agg({
                'efficiency_score': ['mean', 'count'],
                'efficiency_grade': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
            }).round(1)
            
            # 只顯示建案數≥3的縣市
            city_efficiency = city_efficiency[city_efficiency[('efficiency_score', 'count')] >= 3]
            
            if not city_efficiency.empty:
                print(f"\n縣市別效率表現 (建案數≥3):")
                city_efficiency = city_efficiency.sort_values(('efficiency_score', 'mean'), ascending=False)
                
                for county in city_efficiency.index[:10]:  # 前10名
                    avg_score = city_efficiency.loc[county, ('efficiency_score', 'mean')]
                    count = city_efficiency.loc[county, ('efficiency_score', 'count')]
                    mode_grade = city_efficiency.loc[county, ('efficiency_grade', '<lambda>')]
                    print(f"   {county}: {avg_score:.1f}分 ({int(count)}個建案) - 主要等級: {mode_grade}")

# %% [markdown]
# ## 7. 去化動態趨勢分析

# %%
# 去化動態趨勢分析
print("📈 去化動態趨勢分析")
print("=" * 60)

def analyze_absorption_trends(absorption_df, speed_df, efficiency_df, available_seasons):
    """
    分析整體去化動態趨勢
    
    Args:
        absorption_df: 去化率分析結果
        speed_df: 去化速度結果
        efficiency_df: 效率評級結果
        available_seasons: 可用年季列表
        
    Returns:
        dict: 趨勢分析結果
    """
    
    trend_analysis = {
        'seasonal_trends': {},
        'market_momentum': {},
        'efficiency_trends': {},
        'speed_distribution_trends': {},
        'regional_trends': {}
    }
    
    try:
        # 1. 各季度趨勢分析
        for season in available_seasons:
            season_absorption = absorption_df[
                (absorption_df['target_season'] == season) &
                (absorption_df['calculation_status'] == 'success')
            ]
            
            season_speed = speed_df[
                (speed_df['target_season'] == season) &
                (speed_df['calculation_status'] == 'success')
            ]
            
            season_efficiency = efficiency_df[
                (efficiency_df['target_season'] == season) &
                (efficiency_df['calculation_status'] == 'success')
            ]
            
            if not season_absorption.empty:
                trend_analysis['seasonal_trends'][season] = {
                    'project_count': len(season_absorption),
                    'avg_absorption_rate': season_absorption['net_absorption_rate'].mean(),
                    'median_absorption_rate': season_absorption['net_absorption_rate'].median(),
                    'high_absorption_count': len(season_absorption[season_absorption['net_absorption_rate'] >= 70]),
                    'completed_count': len(season_absorption[season_absorption['net_absorption_rate'] >= 100])
                }
                
                if not season_speed.empty:
                    trend_analysis['seasonal_trends'][season].update({
                        'avg_speed': season_speed['quarterly_absorption_speed'].mean(),
                        'high_speed_count': len(season_speed[season_speed['quarterly_absorption_speed'] >= 3])
                    })
                
                if not season_efficiency.empty:
                    grade_dist = season_efficiency['efficiency_grade'].value_counts()
                    trend_analysis['seasonal_trends'][season].update({
                        'excellent_count': grade_dist.get('excellent', 0),
                        'good_count': grade_dist.get('good', 0),
                        'average_count': grade_dist.get('average', 0),
                        'poor_count': grade_dist.get('poor', 0)
                    })
        
        # 2. 市場動能分析
        if len(available_seasons) >= 2:
            recent_seasons = available_seasons[-2:]  # 最近兩季
            
            for i, season in enumerate(recent_seasons):
                season_data = trend_analysis['seasonal_trends'].get(season, {})
                if season_data:
                    trend_analysis['market_momentum'][season] = {
                        'market_activity_level': 'high' if season_data['project_count'] >= 50 else 'medium' if season_data['project_count'] >= 20 else 'low',
                        'completion_momentum': 'strong' if season_data.get('completed_count', 0) >= 5 else 'moderate' if season_data.get('completed_count', 0) >= 2 else 'weak',
                        'speed_momentum': 'accelerating' if season_data.get('avg_speed', 0) >= 2.5 else 'stable' if season_data.get('avg_speed', 0) >= 1.5 else 'slowing'
                    }
        
        # 3. 效率趨勢分析
        efficiency_trend_data = []
        for season in available_seasons:
            season_efficiency = efficiency_df[
                (efficiency_df['target_season'] == season) &
                (efficiency_df['calculation_status'] == 'success')
            ]
            if not season_efficiency.empty:
                avg_efficiency_score = season_efficiency['efficiency_score'].mean()
                excellent_ratio = len(season_efficiency[season_efficiency['efficiency_grade'] == 'excellent']) / len(season_efficiency) * 100
                efficiency_trend_data.append({
                    'season': season,
                    'avg_score': avg_efficiency_score,
                    'excellent_ratio': excellent_ratio
                })
        
        if len(efficiency_trend_data) >= 2:
            # 計算趨勢方向
            score_trend = efficiency_trend_data[-1]['avg_score'] - efficiency_trend_data[-2]['avg_score']
            ratio_trend = efficiency_trend_data[-1]['excellent_ratio'] - efficiency_trend_data[-2]['excellent_ratio']
            
            trend_analysis['efficiency_trends'] = {
                'score_trend': 'improving' if score_trend > 2 else 'declining' if score_trend < -2 else 'stable',
                'excellence_trend': 'increasing' if ratio_trend > 5 else 'decreasing' if ratio_trend < -5 else 'stable',
                'trend_data': efficiency_trend_data
            }
        
        # 4. 速度分布趨勢
        for season in available_seasons:
            season_speed = speed_df[
                (speed_df['target_season'] == season) &
                (speed_df['calculation_status'] == 'success')
            ]
            
            if not season_speed.empty:
                speeds = season_speed['quarterly_absorption_speed']
                trend_analysis['speed_distribution_trends'][season] = {
                    'high_speed_ratio': len(speeds[speeds >= 3]) / len(speeds) * 100,
                    'medium_speed_ratio': len(speeds[(speeds >= 1) & (speeds < 3)]) / len(speeds) * 100,
                    'low_speed_ratio': len(speeds[speeds < 1]) / len(speeds) * 100,
                    'avg_speed': speeds.mean(),
                    'speed_variance': speeds.var()
                }
        
        # 5. 區域趨勢分析（簡化版）
        if 'county' in absorption_df.columns:
            for county in absorption_df['county'].unique():
                county_data = absorption_df[
                    (absorption_df['county'] == county) &
                    (absorption_df['calculation_status'] == 'success')
                ]
                
                if len(county_data) >= 10:  # 至少10個記錄
                    avg_absorption = county_data['net_absorption_rate'].mean()
                    high_absorption_ratio = len(county_data[county_data['net_absorption_rate'] >= 70]) / len(county_data) * 100
                    
                    trend_analysis['regional_trends'][county] = {
                        'avg_absorption_rate': avg_absorption,
                        'high_performance_ratio': high_absorption_ratio,
                        'project_count': len(county_data),
                        'performance_level': 'high' if avg_absorption >= 60 else 'medium' if avg_absorption >= 40 else 'low'
                    }
    
    except Exception as e:
        print(f"❌ 趨勢分析錯誤: {e}")
    
    return trend_analysis

# %%
# 執行去化動態趨勢分析
print("🔄 執行去化動態趨勢分析...")

trend_analysis_result = analyze_absorption_trends(
    absorption_analysis, 
    quarterly_speed_df, 
    efficiency_df, 
    available_seasons
)

print(f"✅ 完成去化動態趨勢分析")

# 趨勢分析結果展示
if trend_analysis_result:
    print(f"\n📊 去化動態趨勢分析結果:")
    
    # 季度趨勢
    print(f"\n1. 季度趨勢分析:")
    for season, data in trend_analysis_result['seasonal_trends'].items():
        print(f"   {season}: {data['project_count']}個建案, 平均去化率{data['avg_absorption_rate']:.1f}%, 完售{data['completed_count']}個")
    
    # 市場動能
    if trend_analysis_result['market_momentum']:
        print(f"\n2. 市場動能分析:")
        for season, momentum in trend_analysis_result['market_momentum'].items():
            print(f"   {season}: 活動水準-{momentum['market_activity_level']}, 完售動能-{momentum['completion_momentum']}, 速度動能-{momentum['speed_momentum']}")
    
    # 效率趨勢
    if trend_analysis_result['efficiency_trends']:
        efficiency_trends = trend_analysis_result['efficiency_trends']
        print(f"\n3. 效率趨勢:")
        print(f"   評分趨勢: {efficiency_trends['score_trend']}")
        print(f"   優秀比例趨勢: {efficiency_trends['excellence_trend']}")
    
    # 區域表現
    if trend_analysis_result['regional_trends']:
        print(f"\n4. 區域表現前5名:")
        regional_sorted = sorted(
            trend_analysis_result['regional_trends'].items(),
            key=lambda x: x[1]['avg_absorption_rate'],
            reverse=True
        )[:5]
        
        for county, data in regional_sorted:
            print(f"   {county}: 平均去化率{data['avg_absorption_rate']:.1f}%, 高表現比例{data['high_performance_ratio']:.1f}%")

# %% [markdown]
# ## 8. 異常去化模式識別

# %%
# 異常去化模式識別
print("🚨 異常去化模式識別")
print("=" * 60)

def identify_abnormal_absorption_patterns(absorption_df, speed_df, efficiency_df, acceleration_df):
    """
    識別異常的去化模式
    
    Args:
        absorption_df: 去化率分析結果
        speed_df: 去化速度結果
        efficiency_df: 效率評級結果
        acceleration_df: 加速度分析結果
        
    Returns:
        dict: 異常模式分析結果
    """
    
    abnormal_patterns = {
        'extreme_speed_variations': [],
        'inconsistent_performance': [],
        'stagnation_patterns': [],
        'acceleration_anomalies': [],
        'efficiency_mismatches': [],
        'suspicious_completions': []
    }
    
    try:
        # 合併所有資料進行分析
        combined_df = absorption_df.merge(
            speed_df[['project_code', 'target_season', 'quarterly_absorption_speed']],
            on=['project_code', 'target_season'],
            how='left'
        )
        
        combined_df = combined_df.merge(
            efficiency_df[['project_code', 'target_season', 'efficiency_grade', 'efficiency_score']],
            on=['project_code', 'target_season'],
            how='left'
        )
        
        combined_df = combined_df.merge(
            acceleration_df[['project_code', 'target_season', 'absorption_acceleration', 'acceleration_status']],
            on=['project_code', 'target_season'],
            how='left'
        )
        
        valid_data = combined_df[combined_df['calculation_status'] == 'success']
        
        # 1. 極端速度變化識別
        for project_code in valid_data['project_code'].unique():
            project_data = valid_data[valid_data['project_code'] == project_code].sort_values(
                'target_season', key=lambda x: x.map(season_to_number)
            )
            
            if len(project_data) >= 2:
                speeds = project_data['quarterly_absorption_speed'].fillna(0)
                if len(speeds) >= 2:
                    speed_changes = speeds.diff().abs()
                    extreme_changes = speed_changes[speed_changes > 5]  # 變化超過5戶/季
                    
                    if not extreme_changes.empty:
                        abnormal_patterns['extreme_speed_variations'].append({
                            'project_code': project_code,
                            'project_name': project_data.iloc[0].get('project_name', ''),
                            'county': project_data.iloc[0].get('county', ''),
                            'max_speed_change': extreme_changes.max(),
                            'seasons_with_extreme_change': len(extreme_changes),
                            'pattern_type': 'extreme_speed_variation'
                        })
        
        # 2. 不一致表現識別（高去化率但低效率評級）
        inconsistent_cases = valid_data[
            (valid_data['net_absorption_rate'] >= 70) &
            (valid_data['efficiency_grade'].isin(['poor', 'average']))
        ]
        
        for _, case in inconsistent_cases.iterrows():
            abnormal_patterns['inconsistent_performance'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'net_absorption_rate': case['net_absorption_rate'],
                'efficiency_grade': case['efficiency_grade'],
                'efficiency_score': case.get('efficiency_score', 0),
                'pattern_type': 'high_absorption_low_efficiency'
            })
        
        # 3. 停滯模式識別（多季速度接近零）
        for project_code in valid_data['project_code'].unique():
            project_data = valid_data[valid_data['project_code'] == project_code].sort_values(
                'target_season', key=lambda x: x.map(season_to_number)
            )
            
            if len(project_data) >= 3:
                low_speed_seasons = len(project_data[project_data['quarterly_absorption_speed'] < 0.5])
                if low_speed_seasons >= 3:
                    latest_absorption = project_data.iloc[-1]['net_absorption_rate']
                    if latest_absorption < 80:  # 未接近完售
                        abnormal_patterns['stagnation_patterns'].append({
                            'project_code': project_code,
                            'project_name': project_data.iloc[0].get('project_name', ''),
                            'county': project_data.iloc[0].get('county', ''),
                            'stagnant_seasons': low_speed_seasons,
                            'current_absorption_rate': latest_absorption,
                            'avg_speed': project_data['quarterly_absorption_speed'].mean(),
                            'pattern_type': 'long_term_stagnation'
                        })
        
        # 4. 加速度異常識別
        extreme_acceleration = valid_data[
            (valid_data['absorption_acceleration'].abs() > 100) &
            (valid_data['absorption_acceleration'] != 999.0)
        ]
        
        for _, case in extreme_acceleration.iterrows():
            abnormal_patterns['acceleration_anomalies'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'absorption_acceleration': case['absorption_acceleration'],
                'acceleration_status': case.get('acceleration_status', ''),
                'current_speed': case.get('current_speed', 0),
                'previous_speed': case.get('previous_speed', 0),
                'pattern_type': 'extreme_acceleration'
            })
        
        # 5. 效率錯配識別（低去化率但高效率評級）
        efficiency_mismatches = valid_data[
            (valid_data['net_absorption_rate'] < 30) &
            (valid_data['efficiency_grade'].isin(['excellent', 'good']))
        ]
        
        for _, case in efficiency_mismatches.iterrows():
            abnormal_patterns['efficiency_mismatches'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'net_absorption_rate': case['net_absorption_rate'],
                'efficiency_grade': case['efficiency_grade'],
                'efficiency_score': case.get('efficiency_score', 0),
                'pattern_type': 'low_absorption_high_efficiency'
            })
        
        # 6. 可疑完售識別（極短時間內完售）
        suspicious_completions = valid_data[
            (valid_data['net_absorption_rate'] >= 100) &
            (valid_data['quarterly_absorption_speed'] > 20)  # 單季超過20戶
        ]
        
        for _, case in suspicious_completions.iterrows():
            abnormal_patterns['suspicious_completions'].append({
                'project_code': case['project_code'],
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case['target_season'],
                'total_units': case.get('total_units', 0),
                'quarterly_absorption_speed': case['quarterly_absorption_speed'],
                'net_absorption_rate': case['net_absorption_rate'],
                'pattern_type': 'rapid_completion'
            })
    
    except Exception as e:
        print(f"❌ 異常模式識別錯誤: {e}")
    
    return abnormal_patterns

# %%
# 執行異常去化模式識別
print("🔄 執行異常去化模式識別...")

abnormal_patterns = identify_abnormal_absorption_patterns(
    absorption_analysis,
    quarterly_speed_df,
    efficiency_df,
    acceleration_df
)

print(f"✅ 完成異常去化模式識別")

# 異常模式統計報告
total_abnormal_cases = sum(len(patterns) for patterns in abnormal_patterns.values())
print(f"\n📊 異常模式識別結果:")
print(f"   總異常案例數: {total_abnormal_cases}")

for pattern_type, cases in abnormal_patterns.items():
    if len(cases) > 0:
        print(f"   {pattern_type}: {len(cases)} 個案例")

# 詳細異常案例報告
if total_abnormal_cases > 0:
    print(f"\n🔍 詳細異常案例報告:")
    
    # 極端速度變化案例
    if abnormal_patterns['extreme_speed_variations']:
        print(f"\n1. 極端速度變化案例 (前3個):")
        for i, case in enumerate(abnormal_patterns['extreme_speed_variations'][:3], 1):
            print(f"   {i}. {case['project_code']} | {case['county']} | 最大變化: {case['max_speed_change']:.1f}戶/季")
    
    # 停滯模式案例
    if abnormal_patterns['stagnation_patterns']:
        print(f"\n2. 長期停滯案例 (前3個):")
        for i, case in enumerate(abnormal_patterns['stagnation_patterns'][:3], 1):
            print(f"   {i}. {case['project_code']} | {case['county']} | 停滯{case['stagnant_seasons']}季 | 去化率{case['current_absorption_rate']:.1f}%")
    
    # 不一致表現案例
    if abnormal_patterns['inconsistent_performance']:
        print(f"\n3. 表現不一致案例 (前3個):")
        for i, case in enumerate(abnormal_patterns['inconsistent_performance'][:3], 1):
            print(f"   {i}. {case['project_code']} | 去化率{case['net_absorption_rate']:.1f}% | 效率等級{case['efficiency_grade']}")
    
    # 可疑完售案例
    if abnormal_patterns['suspicious_completions']:
        print(f"\n4. 可疑快速完售案例:")
        for i, case in enumerate(abnormal_patterns['suspicious_completions'][:3], 1):
            print(f"   {i}. {case['project_code']} | {case['county']} | 速度{case['quarterly_absorption_speed']:.1f}戶/季 | 戶數{case['total_units']}")

# %% [markdown]
# ## 9. 去化動態基準值建立

# %%
# 去化動態基準值建立
print("📏 去化動態基準值建立")
print("=" * 60)

def establish_absorption_dynamics_benchmarks(speed_df, acceleration_df, efficiency_df, completion_df):
    """
    建立去化動態相關的基準值和分級標準
    
    Args:
        speed_df: 去化速度結果
        acceleration_df: 加速度分析結果
        efficiency_df: 效率評級結果
        completion_df: 完售預測結果
        
    Returns:
        dict: 動態基準值和分級標準
    """
    
    dynamics_benchmarks = {}
    
    try:
        # 1. 去化速度基準值
        valid_speeds = speed_df[speed_df['calculation_status'] == 'success']['quarterly_absorption_speed']
        if not valid_speeds.empty:
            dynamics_benchmarks['absorption_speed'] = {
                'mean': valid_speeds.mean(),
                'median': valid_speeds.median(),
                'std': valid_speeds.std(),
                'percentiles': {
                    '10th': valid_speeds.quantile(0.1),
                    '25th': valid_speeds.quantile(0.25),
                    '50th': valid_speeds.quantile(0.5),
                    '75th': valid_speeds.quantile(0.75),
                    '90th': valid_speeds.quantile(0.9)
                },
                'classification': {
                    'high_speed': valid_speeds.quantile(0.8),      # 前20%
                    'medium_speed': valid_speeds.quantile(0.5),    # 前50%
                    'low_speed': valid_speeds.quantile(0.2),       # 前80%
                    'stagnant': 0.5  # 固定閾值
                },
                'grade_thresholds': {
                    'excellent': 5.0,     # 優秀: ≥5戶/季
                    'good': 3.0,          # 良好: 3-5戶/季
                    'average': 1.5,       # 一般: 1.5-3戶/季
                    'poor': 0.5          # 不佳: <1.5戶/季
                }
            }
        
        # 2. 去化加速度基準值
        valid_accelerations = acceleration_df[
            (acceleration_df['calculation_status'] == 'success') &
            (acceleration_df['absorption_acceleration'] != 999.0) &
            (acceleration_df['absorption_acceleration'].abs() <= 200)
        ]['absorption_acceleration']
        
        if not valid_accelerations.empty:
            dynamics_benchmarks['absorption_acceleration'] = {
                'mean': valid_accelerations.mean(),
                'median': valid_accelerations.median(),
                'std': valid_accelerations.std(),
                'percentiles': {
                    '10th': valid_accelerations.quantile(0.1),
                    '25th': valid_accelerations.quantile(0.25),
                    '75th': valid_accelerations.quantile(0.75),
                    '90th': valid_accelerations.quantile(0.9)
                },
                'classification': {
                    'strong_acceleration': 30,      # 強加速: >30%
                    'acceleration': 10,             # 加速: 10-30%
                    'stable': -10,                  # 穩定: -10%~10%
                    'deceleration': -30,            # 減速: -30%~-10%
                    'strong_deceleration': -50      # 強減速: <-30%
                }
            }
        
        # 3. 效率評分基準值
        valid_efficiency = efficiency_df[efficiency_df['calculation_status'] == 'success']['efficiency_score']
        if not valid_efficiency.empty:
            dynamics_benchmarks['efficiency_score'] = {
                'mean': valid_efficiency.mean(),
                'median': valid_efficiency.median(),
                'std': valid_efficiency.std(),
                'percentiles': {
                    '10th': valid_efficiency.quantile(0.1),
                    '25th': valid_efficiency.quantile(0.25),
                    '75th': valid_efficiency.quantile(0.75),
                    '90th': valid_efficiency.quantile(0.9)
                },
                'grade_boundaries': {
                    'excellent': 85,    # 優秀: ≥85分
                    'good': 70,         # 良好: 70-84分
                    'average': 50,      # 一般: 50-69分
                    'poor': 0          # 不佳: <50分
                }
            }
        
        # 4. 完售預測基準值
        valid_predictions = completion_df[
            (completion_df['calculation_status'] == 'success') &
            (completion_df['estimated_completion_seasons'] < 999) &
            (completion_df['prediction_method'] == 'current_speed')
        ]['estimated_completion_seasons']
        
        if not valid_predictions.empty:
            dynamics_benchmarks['completion_prediction'] = {
                'mean': valid_predictions.mean(),
                'median': valid_predictions.median(),
                'std': valid_predictions.std(),
                'percentiles': {
                    '25th': valid_predictions.quantile(0.25),
                    '50th': valid_predictions.quantile(0.5),
                    '75th': valid_predictions.quantile(0.75),
                    '90th': valid_predictions.quantile(0.9)
                },
                'completion_categories': {
                    'fast_completion': 4,      # 快速完售: ≤4季
                    'normal_completion': 8,    # 正常完售: 5-8季
                    'slow_completion': 16,     # 緩慢完售: 9-16季
                    'long_term_sales': 32      # 長期銷售: >16季
                }
            }
        
        # 5. 綜合動態表現分級
        dynamics_benchmarks['comprehensive_dynamics_grade'] = {
            'excellent': {
                'speed_requirement': dynamics_benchmarks['absorption_speed']['grade_thresholds']['excellent'],
                'efficiency_requirement': dynamics_benchmarks['efficiency_score']['grade_boundaries']['excellent'],
                'acceleration_preference': 'positive',
                'completion_expectation': dynamics_benchmarks['completion_prediction']['completion_categories']['fast_completion']
            },
            'good': {
                'speed_requirement': dynamics_benchmarks['absorption_speed']['grade_thresholds']['good'],
                'efficiency_requirement': dynamics_benchmarks['efficiency_score']['grade_boundaries']['good'],
                'acceleration_preference': 'stable_or_positive',
                'completion_expectation': dynamics_benchmarks['completion_prediction']['completion_categories']['normal_completion']
            },
            'average': {
                'speed_requirement': dynamics_benchmarks['absorption_speed']['grade_thresholds']['average'],
                'efficiency_requirement': dynamics_benchmarks['efficiency_score']['grade_boundaries']['average'],
                'acceleration_preference': 'any',
                'completion_expectation': dynamics_benchmarks['completion_prediction']['completion_categories']['slow_completion']
            },
            'poor': {
                'speed_requirement': 0,
                'efficiency_requirement': 0,
                'acceleration_preference': 'any',
                'completion_expectation': 999
            }
        }
        
        # 6. 市場基準比較
        dynamics_benchmarks['market_standards'] = {
            'industry_speed_standard': 2.5,      # 業界標準去化速度
            'healthy_acceleration_range': (-10, 20),  # 健康加速度範圍
            'target_efficiency_score': 75,       # 目標效率分數
            'ideal_completion_seasons': 6,       # 理想完售季數
            'warning_stagnation_threshold': 0.3  # 停滯警告閾值
        }
    
    except Exception as e:
        print(f"❌ 基準值建立錯誤: {e}")
    
    return dynamics_benchmarks

# %%
# 建立去化動態基準值
print("🔄 建立去化動態基準值...")

dynamics_benchmarks = establish_absorption_dynamics_benchmarks(
    quarterly_speed_df,
    acceleration_df,
    efficiency_df,
    completion_df
)

print(f"✅ 完成去化動態基準值建立")

if dynamics_benchmarks:
    print(f"\n📊 去化動態基準值報告:")
    
    # 去化速度基準值
    if 'absorption_speed' in dynamics_benchmarks:
        speed_bench = dynamics_benchmarks['absorption_speed']
        print(f"\n1. 去化速度基準值:")
        print(f"   平均值: {speed_bench['mean']:.2f} 戶/季")
        print(f"   中位數: {speed_bench['median']:.2f} 戶/季")
        print(f"   分級標準:")
        for grade, threshold in speed_bench['grade_thresholds'].items():
            print(f"     {grade}: {threshold:.1f} 戶/季")
    
    # 效率評分基準值
    if 'efficiency_score' in dynamics_benchmarks:
        eff_bench = dynamics_benchmarks['efficiency_score']
        print(f"\n2. 效率評分基準值:")
        print(f"   平均分: {eff_bench['mean']:.1f}")
        print(f"   中位數: {eff_bench['median']:.1f}")
        print(f"   分級邊界:")
        for grade, boundary in eff_bench['grade_boundaries'].items():
            print(f"     {grade}: {boundary}分")
    
    # 完售預測基準值
    if 'completion_prediction' in dynamics_benchmarks:
        comp_bench = dynamics_benchmarks['completion_prediction']
        print(f"\n3. 完售預測基準值:")
        print(f"   平均預測: {comp_bench['mean']:.1f} 季")
        print(f"   中位數預測: {comp_bench['median']:.1f} 季")
        print(f"   完售分類:")
        for category, threshold in comp_bench['completion_categories'].items():
            print(f"     {category}: {threshold}季")
    
    # 市場標準
    if 'market_standards' in dynamics_benchmarks:
        market_std = dynamics_benchmarks['market_standards']
        print(f"\n4. 市場標準:")
        print(f"   業界速度標準: {market_std['industry_speed_standard']} 戶/季")
        print(f"   目標效率分數: {market_std['target_efficiency_score']} 分")
        print(f"   理想完售季數: {market_std['ideal_completion_seasons']} 季")

# %% [markdown]
# ## 10. 視覺化分析

# %%
# 創建去化動態分析視覺化
print("📊 去化動態分析視覺化")
print("=" * 50)

# 創建圖表
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 過濾有效數據
valid_speed_data = quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']
valid_efficiency_data = efficiency_df[efficiency_df['calculation_status'] == 'success']
valid_acceleration_data = acceleration_df[
    (acceleration_df['calculation_status'] == 'success') &
    (acceleration_df['absorption_acceleration'] != 999.0) &
    (acceleration_df['absorption_acceleration'].abs() <= 200)
]

# 1. 去化速度分布直方圖
if not valid_speed_data.empty:
    speeds = valid_speed_data['quarterly_absorption_speed']
    axes[0, 0].hist(speeds[speeds <= 10], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('季度去化速度分布', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('去化速度 (戶/季)')
    axes[0, 0].set_ylabel('建案數量')
    axes[0, 0].axvline(x=speeds.mean(), color='red', linestyle='--', label=f'平均值: {speeds.mean():.2f}')
    axes[0, 0].axvline(x=speeds.median(), color='green', linestyle='--', label=f'中位數: {speeds.median():.2f}')
    axes[0, 0].legend()

# 2. 效率評級分布
if not valid_efficiency_data.empty:
    grade_dist = valid_efficiency_data['efficiency_grade'].value_counts()
    colors = {'excellent': 'green', 'good': 'lightgreen', 'average': 'orange', 'poor': 'red'}
    bar_colors = [colors.get(grade, 'gray') for grade in grade_dist.index]
    
    bars = axes[0, 1].bar(grade_dist.index, grade_dist.values, color=bar_colors)
    axes[0, 1].set_title('去化效率評級分布', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('效率等級')
    axes[0, 1].set_ylabel('建案數量')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')

# 3. 加速度分布
if not valid_acceleration_data.empty:
    accelerations = valid_acceleration_data['absorption_acceleration']
    axes[0, 2].hist(accelerations, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_title('去化加速度分布', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('加速度 (%)')
    axes[0, 2].set_ylabel('建案數量')
    axes[0, 2].axvline(x=0, color='black', linestyle='-', alpha=0.5, label='穩定基準線')
    axes[0, 2].axvline(x=accelerations.mean(), color='red', linestyle='--', label=f'平均值: {accelerations.mean():.1f}%')
    axes[0, 2].legend()

# 4. 年季別去化速度趨勢
if len(available_seasons) > 1:
    speed_trends = []
    for season in available_seasons:
        season_data = valid_speed_data[valid_speed_data['target_season'] == season]
        if not season_data.empty:
            speed_trends.append({
                'season': season,
                'avg_speed': season_data['quarterly_absorption_speed'].mean(),
                'count': len(season_data)
            })
    
    if speed_trends:
        trend_df = pd.DataFrame(speed_trends)
        line = axes[1, 0].plot(range(len(trend_df)), trend_df['avg_speed'], marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('各年季平均去化速度趨勢', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('年季')
        axes[1, 0].set_ylabel('平均去化速度 (戶/季)')
        axes[1, 0].set_xticks(range(len(trend_df)))
        axes[1, 0].set_xticklabels(trend_df['season'], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

# 5. 效率分數分布
if not valid_efficiency_data.empty:
    scores = valid_efficiency_data['efficiency_score']
    axes[1, 1].hist(scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('效率評分分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('效率評分')
    axes[1, 1].set_ylabel('建案數量')
    axes[1, 1].axvline(x=scores.mean(), color='red', linestyle='--', label=f'平均分: {scores.mean():.1f}')
    axes[1, 1].axvline(x=75, color='orange', linestyle='--', label='目標分數: 75')
    axes[1, 1].legend()

# 6. 去化速度 vs 效率評分散點圖
if not valid_speed_data.empty and not valid_efficiency_data.empty:
    # 合併速度和效率資料
    speed_efficiency = valid_speed_data.merge(
        valid_efficiency_data[['project_code', 'target_season', 'efficiency_score']],
        on=['project_code', 'target_season'],
        how='inner'
    )
    
    if not speed_efficiency.empty:
        # 過濾極端值
        scatter_data = speed_efficiency[
            (speed_efficiency['quarterly_absorption_speed'] <= 15) &
            (speed_efficiency['efficiency_score'] <= 100)
        ]
        
        axes[1, 2].scatter(scatter_data['quarterly_absorption_speed'], scatter_data['efficiency_score'], 
                          alpha=0.6, color='purple')
        axes[1, 2].set_title('去化速度 vs 效率評分', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('季度去化速度 (戶/季)')
        axes[1, 2].set_ylabel('效率評分')
        
        # 添加趨勢線
        if len(scatter_data) > 5:
            z = np.polyfit(scatter_data['quarterly_absorption_speed'], scatter_data['efficiency_score'], 1)
            p = np.poly1d(z)
            axes[1, 2].plot(scatter_data['quarterly_absorption_speed'], 
                           p(scatter_data['quarterly_absorption_speed']), 
                           "r--", alpha=0.8, label='趨勢線')
            axes[1, 2].legend()

# 7. 完售預測時間分布
completion_current_method = completion_df[
    (completion_df['calculation_status'] == 'success') &
    (completion_df['prediction_method'] == 'current_speed') &
    (completion_df['estimated_completion_seasons'] < 50)
]

if not completion_current_method.empty:
    completion_seasons = completion_current_method['estimated_completion_seasons']
    axes[2, 0].hist(completion_seasons, bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[2, 0].set_title('預估完售時間分布', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('預估完售季數')
    axes[2, 0].set_ylabel('建案數量')
    axes[2, 0].axvline(x=completion_seasons.mean(), color='red', linestyle='--', 
                      label=f'平均: {completion_seasons.mean():.1f}季')
    axes[2, 0].axvline(x=8, color='orange', linestyle='--', label='正常完售: 8季')
    axes[2, 0].legend()

# 8. 加速度狀態分布
if not valid_acceleration_data.empty:
    accel_status_dist = valid_acceleration_data['acceleration_status'].value_counts()
    status_colors = {
        'strong_acceleration': 'darkgreen',
        'acceleration': 'lightgreen',
        'stable': 'gray',
        'deceleration': 'orange',
        'strong_deceleration': 'red',
        'initial': 'blue',
        'restart': 'purple',
        'stagnant': 'darkred'
    }
    bar_colors = [status_colors.get(status, 'gray') for status in accel_status_dist.index]
    
    bars = axes[2, 1].bar(range(len(accel_status_dist)), accel_status_dist.values, color=bar_colors)
    axes[2, 1].set_title('加速度狀態分布', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('加速度狀態')
    axes[2, 1].set_ylabel('建案數量')
    axes[2, 1].set_xticks(range(len(accel_status_dist)))
    axes[2, 1].set_xticklabels(accel_status_dist.index, rotation=45, ha='right')
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)

# 9. 縣市別平均去化速度比較
if 'county' in valid_speed_data.columns:
    city_speed = valid_speed_data.groupby('county')['quarterly_absorption_speed'].agg(['mean', 'count']).reset_index()
    city_speed = city_speed[city_speed['count'] >= 5]  # 至少5個建案
    city_speed = city_speed.nlargest(8, 'mean')  # 前8名
    
    if not city_speed.empty:
        bars = axes[2, 2].bar(range(len(city_speed)), city_speed['mean'], color='lightblue')
        axes[2, 2].set_title('縣市別平均去化速度 (前8名)', fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('縣市')
        axes[2, 2].set_ylabel('平均去化速度 (戶/季)')
        axes[2, 2].set_xticks(range(len(city_speed)))
        axes[2, 2].set_xticklabels(city_speed['county'], rotation=45, ha='right')
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = city_speed.iloc[i]['count']
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}\n({int(count)}個)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. 結果儲存與匯出

# %%
# 儲存去化動態分析結果
print("💾 儲存去化動態分析結果...")

# 1. 儲存季度去化速度結果
speed_output_columns = [
    'project_code', 'project_name', 'county', 'district', 'target_season',
    'quarterly_absorption_speed', 'speed_calculation_method', 'previous_season',
    'current_net_absorption_rate', 'previous_net_absorption_rate', 'total_units',
    'has_complete_info', 'calculation_status', 'error_message'
]

available_speed_columns = [col for col in speed_output_columns if col in quarterly_speed_df.columns]
speed_output_df = quarterly_speed_df[available_speed_columns].copy()

speed_output_df.to_csv('../data/processed/06_quarterly_absorption_speed.csv', 
                      index=False, encoding='utf-8-sig')
print("✅ 季度去化速度結果已儲存至: ../data/processed/06_quarterly_absorption_speed.csv")

# 2. 儲存去化加速度結果
acceleration_output_columns = [
    'project_code', 'project_name', 'county', 'district', 'target_season',
    'absorption_acceleration', 'current_speed', 'previous_speed', 'acceleration_status',
    'has_complete_info', 'calculation_status', 'error_message'
]

available_accel_columns = [col for col in acceleration_output_columns if col in acceleration_df.columns]
acceleration_output_df = acceleration_df[available_accel_columns].copy()

acceleration_output_df.to_csv('../data/processed/06_absorption_acceleration.csv', 
                             index=False, encoding='utf-8-sig')
print("✅ 去化加速度結果已儲存至: ../data/processed/06_absorption_acceleration.csv")

# 3. 儲存完售預測結果
completion_output_columns = [
    'project_code', 'project_name', 'county', 'district', 'target_season',
    'estimated_completion_seasons', 'estimated_completion_season', 'prediction_method',
    'current_absorption_rate', 'remaining_absorption_rate', 'current_speed',
    'prediction_confidence', 'completion_status', 'total_units',
    'has_complete_info', 'calculation_status', 'error_message'
]

available_completion_columns = [col for col in completion_output_columns if col in completion_df.columns]
completion_output_df = completion_df[available_completion_columns].copy()

completion_output_df.to_csv('../data/processed/06_completion_prediction.csv', 
                           index=False, encoding='utf-8-sig')
print("✅ 完售預測結果已儲存至: ../data/processed/06_completion_prediction.csv")

# 4. 儲存效率評級結果
efficiency_output_columns = [
    'project_code', 'project_name', 'county', 'district', 'target_season',
    'efficiency_grade', 'efficiency_score', 'absorption_score', 'speed_score',
    'completion_score', 'time_score', 'grade_emoji', 'grade_description',
    'net_absorption_rate', 'total_units', 'has_complete_info',
    'calculation_status', 'error_message'
]

available_efficiency_columns = [col for col in efficiency_output_columns if col in efficiency_df.columns]
efficiency_output_df = efficiency_df[available_efficiency_columns].copy()

efficiency_output_df.to_csv('../data/processed/06_absorption_efficiency.csv', 
                           index=False, encoding='utf-8-sig')
print("✅ 去化效率評級結果已儲存至: ../data/processed/06_absorption_efficiency.csv")

# 5. 儲存動態基準值
if dynamics_benchmarks:
    benchmark_records = []
    
    # 速度基準值
    if 'absorption_speed' in dynamics_benchmarks:
        speed_bench = dynamics_benchmarks['absorption_speed']
        benchmark_records.append({
            'metric_category': 'absorption_speed',
            'metric_name': 'quarterly_absorption_speed',
            'mean_value': speed_bench['mean'],
            'median_value': speed_bench['median'],
            'std_value': speed_bench['std'],
            'percentile_25': speed_bench['percentiles']['25th'],
            'percentile_75': speed_bench['percentiles']['75th'],
            'excellent_threshold': speed_bench['grade_thresholds']['excellent'],
            'good_threshold': speed_bench['grade_thresholds']['good'],
            'average_threshold': speed_bench['grade_thresholds']['average'],
            'poor_threshold': speed_bench['grade_thresholds']['poor']
        })
    
    # 效率基準值
    if 'efficiency_score' in dynamics_benchmarks:
        eff_bench = dynamics_benchmarks['efficiency_score']
        benchmark_records.append({
            'metric_category': 'efficiency_score',
            'metric_name': 'absorption_efficiency_score',
            'mean_value': eff_bench['mean'],
            'median_value': eff_bench['median'],
            'std_value': eff_bench['std'],
            'percentile_25': eff_bench['percentiles']['25th'],
            'percentile_75': eff_bench['percentiles']['75th'],
            'excellent_threshold': eff_bench['grade_boundaries']['excellent'],
            'good_threshold': eff_bench['grade_boundaries']['good'],
            'average_threshold': eff_bench['grade_boundaries']['average'],
            'poor_threshold': eff_bench['grade_boundaries']['poor']
        })
    
    if benchmark_records:
        benchmark_df = pd.DataFrame(benchmark_records)
        benchmark_df.to_csv('../data/processed/06_dynamics_benchmarks.csv', 
                           index=False, encoding='utf-8-sig')
        print("✅ 去化動態基準值已儲存至: ../data/processed/06_dynamics_benchmarks.csv")

# 6. 儲存異常模式報告
if total_abnormal_cases > 0:
    abnormal_records = []
    
    for pattern_type, cases in abnormal_patterns.items():
        for case in cases:
            abnormal_record = {
                'pattern_category': pattern_type,
                'project_code': case.get('project_code', ''),
                'project_name': case.get('project_name', ''),
                'county': case.get('county', ''),
                'target_season': case.get('target_season', ''),
                'pattern_type': case.get('pattern_type', pattern_type),
                'severity': 'high' if pattern_type in ['stagnation_patterns', 'extreme_speed_variations'] else 'medium',
                'description': str(case),
                'key_metrics': '; '.join([f"{k}: {v}" for k, v in case.items() 
                                        if k not in ['project_code', 'project_name', 'county', 'target_season', 'pattern_type']])
            }
            abnormal_records.append(abnormal_record)
    
    if abnormal_records:
        abnormal_df = pd.DataFrame(abnormal_records)
        abnormal_df.to_csv('../data/processed/06_abnormal_patterns.csv', 
                          index=False, encoding='utf-8-sig')
        print("✅ 異常模式報告已儲存至: ../data/processed/06_abnormal_patterns.csv")

# 7. 儲存趨勢分析結果
if trend_analysis_result:
    trend_records = []
    
    # 季度趨勢
    for season, data in trend_analysis_result['seasonal_trends'].items():
        trend_records.append({
            'analysis_type': 'seasonal_trend',
            'period': season,
            'project_count': data['project_count'],
            'avg_absorption_rate': data['avg_absorption_rate'],
            'median_absorption_rate': data['median_absorption_rate'],
            'high_absorption_count': data['high_absorption_count'],
            'completed_count': data['completed_count'],
            'avg_speed': data.get('avg_speed', 0),
            'high_speed_count': data.get('high_speed_count', 0),
            'excellent_count': data.get('excellent_count', 0),
            'good_count': data.get('good_count', 0)
        })
    
    if trend_records:
        trend_df = pd.DataFrame(trend_records)
        trend_df.to_csv('../data/processed/06_trend_analysis.csv', 
                       index=False, encoding='utf-8-sig')
        print("✅ 趨勢分析結果已儲存至: ../data/processed/06_trend_analysis.csv")

# 8. 儲存綜合分析摘要
summary_report = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'target_seasons': ', '.join(available_seasons),
    'total_speed_calculations': len(quarterly_speed_df),
    'successful_speed_calculations': len(quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']),
    'speed_calculation_success_rate': len(quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']) / len(quarterly_speed_df) * 100 if len(quarterly_speed_df) > 0 else 0,
    'total_efficiency_evaluations': len(efficiency_df),
    'successful_efficiency_evaluations': len(efficiency_df[efficiency_df['calculation_status'] == 'success']),
    'total_completion_predictions': len(completion_df),
    'successful_completion_predictions': len(completion_df[completion_df['calculation_status'] == 'success']),
    'total_abnormal_patterns': total_abnormal_cases,
    'avg_absorption_speed': quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']['quarterly_absorption_speed'].mean() if not quarterly_speed_df.empty else 0,
    'avg_efficiency_score': efficiency_df[efficiency_df['calculation_status'] == 'success']['efficiency_score'].mean() if not efficiency_df.empty else 0,
    'excellent_efficiency_ratio': len(efficiency_df[efficiency_df['efficiency_grade'] == 'excellent']) / len(efficiency_df) * 100 if not efficiency_df.empty else 0,
    'high_speed_projects_ratio': len(quarterly_speed_df[quarterly_speed_df['quarterly_absorption_speed'] >= 3]) / len(quarterly_speed_df) * 100 if not quarterly_speed_df.empty else 0
}

summary_df = pd.DataFrame([summary_report])
summary_df.to_csv('../data/processed/06_dynamics_summary.csv', 
                  index=False, encoding='utf-8-sig')
print("✅ 綜合分析摘要已儲存至: ../data/processed/06_dynamics_summary.csv")

# %% [markdown]
# ## 12. 分析總結與下一步

# %%
# 去化動態分析總結
print("📋 去化動態分析總結")
print("=" * 80)

print("1️⃣ 計算完成度:")
successful_speed = len(quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success'])
total_speed = len(quarterly_speed_df)
speed_success_rate = successful_speed / total_speed * 100 if total_speed > 0 else 0

successful_efficiency = len(efficiency_df[efficiency_df['calculation_status'] == 'success'])
total_efficiency = len(efficiency_df)
efficiency_success_rate = successful_efficiency / total_efficiency * 100 if total_efficiency > 0 else 0

print(f"   ✅ 去化速度計算: {successful_speed:,}/{total_speed:,} ({speed_success_rate:.1f}%)")
print(f"   ✅ 效率評級計算: {successful_efficiency:,}/{total_efficiency:,} ({efficiency_success_rate:.1f}%)")
print(f"   ✅ 加速度計算: {len(acceleration_df[acceleration_df['calculation_status'] == 'success']):,}")
print(f"   ✅ 完售預測計算: {len(completion_df[completion_df['calculation_status'] == 'success']):,}")

print(f"\n2️⃣ 核心動態指標統計:")
if successful_speed > 0:
    valid_speeds = quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']
    
    print(f"   📊 平均去化速度: {valid_speeds['quarterly_absorption_speed'].mean():.2f} 戶/季")
    print(f"   📊 中位數去化速度: {valid_speeds['quarterly_absorption_speed'].median():.2f} 戶/季")
    
    high_speed_count = len(valid_speeds[valid_speeds['quarterly_absorption_speed'] >= 3])
    stagnant_count = len(valid_speeds[valid_speeds['quarterly_absorption_speed'] < 0.5])
    
    print(f"   📊 高速去化建案 (≥3戶/季): {high_speed_count:,} 個 ({high_speed_count/len(valid_speeds)*100:.1f}%)")
    print(f"   📊 滯銷建案 (<0.5戶/季): {stagnant_count:,} 個 ({stagnant_count/len(valid_speeds)*100:.1f}%)")

if successful_efficiency > 0:
    valid_efficiency = efficiency_df[efficiency_df['calculation_status'] == 'success']
    
    print(f"   📊 平均效率評分: {valid_efficiency['efficiency_score'].mean():.1f}分")
    
    excellent_count = len(valid_efficiency[valid_efficiency['efficiency_grade'] == 'excellent'])
    poor_count = len(valid_efficiency[valid_efficiency['efficiency_grade'] == 'poor'])
    
    print(f"   📊 優秀效率建案: {excellent_count:,} 個 ({excellent_count/len(valid_efficiency)*100:.1f}%)")
    print(f"   📊 不佳效率建案: {poor_count:,} 個 ({poor_count/len(valid_efficiency)*100:.1f}%)")

print(f"\n3️⃣ 預測與趨勢分析:")
valid_completion = completion_df[
    (completion_df['calculation_status'] == 'success') &
    (completion_df['prediction_method'] == 'current_speed') &
    (completion_df['estimated_completion_seasons'] < 999)
]

if not valid_completion.empty:
    avg_completion_seasons = valid_completion['estimated_completion_seasons'].mean()
    fast_completion_count = len(valid_completion[valid_completion['estimated_completion_seasons'] <= 4])
    
    print(f"   ⏰ 平均預估完售時間: {avg_completion_seasons:.1f} 季")
    print(f"   🚀 快速完售建案 (≤4季): {fast_completion_count:,} 個")

if len(available_seasons) >= 2:
    print(f"   📈 趨勢分析涵蓋: {len(available_seasons)} 個年季")
    print(f"   📊 市場動能評估: 完成")

print(f"\n4️⃣ 基準值與分級標準:")
if dynamics_benchmarks:
    print(f"   ✅ 去化動態基準值建立: 完成")
    
    if 'absorption_speed' in dynamics_benchmarks:
        speed_bench = dynamics_benchmarks['absorption_speed']
        print(f"   📏 速度優秀標準: ≥{speed_bench['grade_thresholds']['excellent']:.1f} 戶/季")
        print(f"   📏 速度良好標準: ≥{speed_bench['grade_thresholds']['good']:.1f} 戶/季")
    
    if 'efficiency_score' in dynamics_benchmarks:
        eff_bench = dynamics_benchmarks['efficiency_score']
        print(f"   📏 效率優秀標準: ≥{eff_bench['grade_boundaries']['excellent']} 分")
        print(f"   📏 效率良好標準: ≥{eff_bench['grade_boundaries']['good']} 分")
else:
    print(f"   ❌ 基準值建立: 失敗")

print(f"\n5️⃣ 異常模式識別:")
print(f"   🚨 總異常案例數: {total_abnormal_cases}")

if total_abnormal_cases > 0:
    for pattern_type, cases in abnormal_patterns.items():
        if len(cases) > 0:
            print(f"   ⚠️ {pattern_type}: {len(cases)} 個")

print(f"\n6️⃣ 關鍵發現:")

# 速度趨勢分析
if len(available_seasons) > 1 and not valid_speeds.empty:
    season_speed_trends = []
    for season in available_seasons:
        season_data = valid_speeds[valid_speeds['target_season'] == season]
        if not season_data.empty:
            avg_speed = season_data['quarterly_absorption_speed'].mean()
            season_speed_trends.append((season, avg_speed))
    
    if len(season_speed_trends) >= 2:
        trend_direction = "上升" if season_speed_trends[-1][1] > season_speed_trends[0][1] else "下降"
        print(f"   📈 去化速度趨勢: {trend_direction} ({season_speed_trends[0][1]:.2f} → {season_speed_trends[-1][1]:.2f} 戶/季)")

# 效率分析
if not valid_efficiency.empty:
    excellent_ratio = len(valid_efficiency[valid_efficiency['efficiency_grade'] == 'excellent']) / len(valid_efficiency) * 100
    print(f"   ⭐ 優秀效率建案比例: {excellent_ratio:.1f}%")

# 縣市表現
if 'county' in valid_speeds.columns:
    city_performance = valid_speeds.groupby('county')['quarterly_absorption_speed'].agg(['mean', 'count']).reset_index()
    city_performance = city_performance[city_performance['count'] >= 3]
    
    if not city_performance.empty:
        best_city = city_performance.loc[city_performance['mean'].idxmax()]
        print(f"   🏆 最佳速度表現縣市: {best_city['county']} ({best_city['mean']:.2f} 戶/季)")

print(f"\n7️⃣ 品質與準確性評估:")

if speed_success_rate >= 90:
    print(f"   ✅ 去化速度計算品質: 優秀")
elif speed_success_rate >= 80:
    print(f"   ⚠️ 去化速度計算品質: 良好")
else:
    print(f"   ❌ 去化速度計算品質: 需改善")

if total_abnormal_cases > 0:
    abnormal_ratio = total_abnormal_cases / (successful_speed + successful_efficiency) * 100
    if abnormal_ratio > 10:
        print(f"   ⚠️ 異常案例比例偏高: {abnormal_ratio:.1f}%")
    else:
        print(f"   ✅ 異常案例比例正常: {abnormal_ratio:.1f}%")

print(f"\n8️⃣ 下一步工作:")
print("   🎯 整合所有動態指標至社區級報告")
print("   📊 進行行政區級動態聚合分析")
print("   🏘️ 建立縣市級動態趨勢分析")
print("   🌟 實作銷售階段動態判斷邏輯")
print("   📈 建立動態預警監控系統")
print("   🔮 開發市場預測模型")

# %%
# 動態指標準備情況檢查
print(f"\n🔍 社區級報告動態指標準備情況:")

required_dynamics_indicators = {
    '季度去化速度': len(quarterly_speed_df[quarterly_speed_df['calculation_status'] == 'success']) > 0,
    '去化加速度': len(acceleration_df[acceleration_df['calculation_status'] == 'success']) > 0,
    '預估完售季數': len(completion_df[completion_df['calculation_status'] == 'success']) > 0,
    '去化效率評級': len(efficiency_df[efficiency_df['calculation_status'] == 'success']) > 0,
    '動態基準值': bool(dynamics_benchmarks),
    '異常模式識別': total_abnormal_cases >= 0,
    '趨勢分析': bool(trend_analysis_result),
    '效率分數計算': 'efficiency_score' in efficiency_df.columns
}

print("動態指標檢查:")
for indicator, status in required_dynamics_indicators.items():
    status_icon = "✅" if status else "❌"
    print(f"   {status_icon} {indicator}")

all_dynamics_ready = all(required_dynamics_indicators.values())
if all_dynamics_ready:
    print(f"\n🎉 所有動態指標準備完成，可以進行完整社區級報告生成")
else:
    missing_indicators = [k for k, v in required_dynamics_indicators.items() if not v]
    print(f"\n⚠️ 以下動態指標需要補強: {', '.join(missing_indicators)}")

# 檢查與前階段資料的整合準備度
integration_readiness = {
    '去化率資料對接': 'net_absorption_rate' in absorption_analysis.columns,
    '建案基本資料對接': 'project_code' in active_projects.columns,
    '時間序列對接': len(available_seasons) > 0,
    '地理資訊對接': 'county' in quarterly_speed_df.columns,
    '品質標記對接': 'calculation_status' in quarterly_speed_df.columns
}

print(f"\n🔗 與前階段資料整合準備度:")
for aspect, status in integration_readiness.items():
    status_icon = "✅" if status else "❌"
    print(f"   {status_icon} {aspect}")

# %% [markdown]
# ## 13. 核心算法驗證與品質檢查
# 
# ### ✅ 已完成核心功能:
# 1. **季度去化速度計算**：基於淨去化率變化的精確計算
# 2. **去化加速度分析**：速度變化率計算與狀態分類
# 3. **預估完售時間算法**：三種預測方法的綜合評估
# 4. **去化效率評級系統**：四維度評分的綜合分級
# 5. **動態趨勢分析**：時間序列與區域比較分析
# 6. **異常模式識別**：六大類異常模式的自動識別
# 7. **動態基準值建立**：統計基準與市場標準的建立
# 
# ### 🎯 關鍵演算法創新:
# 1. **多方法完售預測**：current_speed/average_speed/trend_based三種方法
# 2. **綜合效率評分**：去化率(30分)+速度(25分)+完售預測(25分)+時間效率(20分)
# 3. **動態加速度計算**：考慮重啟、停滯、初期等特殊狀態
# 4. **異常模式自動識別**：六大維度的智能異常檢測
# 
# ### 🔄 待執行項目:
# 1. **社區級32欄位報告整合**：所有靜態與動態指標的完整整合
# 2. **銷售階段動態判斷**：基於動態指標的階段智能識別
# 3. **行政區級動態聚合**：區域層級的動態指標聚合邏輯
# 4. **三層級動態風險整合**：跨層級的動態風險評估體系

print("\n" + "="*80)
print("🎉 Notebook 6 - 去化動態分析與效率評級完成！")
print("📝 請繼續執行 Notebook 7 進行銷售階段判斷與社區級完整報告生成")
print("="*80)