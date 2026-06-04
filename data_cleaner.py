"""
数据清洗模块 — 缺失值检测处理 + 异常值检测处理
"""
import pandas as pd
import numpy as np


def analyze_quality(df):
    """
    分析数据质量，返回每列的缺失值和异常值报告。
    返回格式：
    {
        'total_rows': int,
        'total_cols': int,
        'columns': [
            {
                'name': '列名',
                'dtype': 'int64/float64/object/...',
                'missing_count': 0,
                'missing_pct': 0.0,
                'outlier_count': 0,      # 仅数值列有
                'outlier_pct': 0.0,       # 仅数值列有
                'outlier_method': 'iqr',  # 检测方法
                'outlier_bounds': [low, high],  # 正常范围
                'outlier_indices': [...]   # 异常值所在行索引
            },
            ...
        ]
    }
    """
    report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'columns': []
    }

    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_pct': round(float(df[col].isna().sum() / len(df) * 100), 2)
        }

        # 仅对真正的数值列做异常值检测（排除 bool 类型）
        is_real_numeric = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        if is_real_numeric:
            outliers = _detect_outliers_iqr(df[col])
            col_info['outlier_count'] = len(outliers)
            col_info['outlier_pct'] = round(len(outliers) / len(df) * 100, 2)
            col_info['outlier_method'] = 'iqr'
            col_info['outlier_bounds'] = _get_iqr_bounds(df[col])
            col_info['outlier_indices'] = outliers
            # 返回异常行的完整数据（所有列），方便前端展示
            outlier_rows = []
            for idx in outliers:
                outlier_rows.append({
                    'index': int(idx),
                    'values': {c: _safe_json_val(df.loc[idx, c]) for c in df.columns}
                })
            col_info['outlier_rows'] = outlier_rows
        else:
            col_info['outlier_count'] = 0
            col_info['outlier_pct'] = 0.0
            col_info['outlier_method'] = ''
            col_info['outlier_bounds'] = []
            col_info['outlier_indices'] = []
            col_info['outlier_rows'] = []

        report['columns'].append(col_info)

    return report


def apply_cleaning(df, config):
    """
    根据配置执行清洗，返回 (清洗后DataFrame, 执行报告)。

    config 格式：
    {
        'missing_strategy': {
            '全局': 'drop' | 'fill_mean' | 'fill_median' | 'fill_mode' | 'fill_zero' | 'ignore',
            '列名A': 'fill_mean',   # 覆盖全局策略
            '列名B': 'drop',
        },
        'outlier_strategy': {
            '全局': 'ignore' | 'remove' | 'cap',
            'method': 'iqr' | 'zscore',
            'iqr_multiplier': 1.5,
            'zscore_threshold': 3,
            '列名A': 'remove',
        }
    }
    """
    report_lines = []
    total_before = len(df)
    df = df.copy()

    # -------- 1. 缺失值处理 --------
    miss_cfg = config.get('missing_strategy', {})
    global_miss = miss_cfg.get('全局', 'ignore')

    for col in df.columns:
        strategy = miss_cfg.get(col, global_miss)
        missing_before = int(df[col].isna().sum())
        if missing_before == 0 or strategy == 'ignore':
            continue

        if strategy == 'drop':
            df = df[df[col].notna()]
            report_lines.append(f'列 [{col}] 缺失值删除：移除 {missing_before} 行')
        elif strategy == 'fill_mean' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
            report_lines.append(f'列 [{col}] 缺失值填均值：填充 {missing_before} 个')
        elif strategy == 'fill_median' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            report_lines.append(f'列 [{col}] 缺失值填中位数：填充 {missing_before} 个')
        elif strategy == 'fill_mode':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
            report_lines.append(f'列 [{col}] 缺失值填众数：填充 {missing_before} 个')
        elif strategy == 'fill_zero':
            df[col] = df[col].fillna(0)
            report_lines.append(f'列 [{col}] 缺失值填0：填充 {missing_before} 个')
        elif strategy == 'fill_ffill':
            df[col] = df[col].ffill()
            report_lines.append(f'列 [{col}] 缺失值前向填充：填充 {missing_before} 个')
        elif strategy == 'fill_bfill':
            df[col] = df[col].bfill()
            report_lines.append(f'列 [{col}] 缺失值后向填充：填充 {missing_before} 个')

    # -------- 2. 异常值处理 --------
    outlier_cfg = config.get('outlier_strategy', {})
    global_outlier = outlier_cfg.get('全局', 'ignore')
    method = outlier_cfg.get('method', 'iqr')
    iqr_mult = outlier_cfg.get('iqr_multiplier', 1.5)
    z_thresh = outlier_cfg.get('zscore_threshold', 3)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        strategy = outlier_cfg.get(col, global_outlier)
        if strategy == 'ignore':
            continue

        if method == 'iqr':
            low, high = _get_iqr_bounds(df[col], iqr_mult)
            outliers = df[col].apply(lambda x: x < low or x > high)
        else:
            z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
            outliers = z.abs() > z_thresh
            low, high = None, None

        n_out = outliers.sum()
        if n_out == 0:
            continue

        if strategy == 'remove':
            df = df[~outliers]
            report_lines.append(f'列 [{col}] 异常值删除：移除 {n_out} 行')
        elif strategy == 'cap':
            if method == 'iqr':
                df[col] = df[col].clip(lower=low, upper=high)
            else:
                mean, std = df[col].mean(), df[col].std(ddof=0)
                df[col] = df[col].clip(lower=mean - z_thresh * std, upper=mean + z_thresh * std)
            report_lines.append(f'列 [{col}] 异常值截断：处理 {n_out} 个（超出范围的替换为边界值）')

    # -------- 3. 汇总报告 --------
    total_after = len(df)
    dropped_rows = total_before - total_after

    return df, {
        'rows_before': total_before,
        'rows_after': total_after,
        'rows_dropped': dropped_rows,
        'details': report_lines
    }


# ========== 内部工具函数 ==========

def _get_iqr_bounds(series, multiplier=1.5):
    """计算 IQR 正常范围 [下界, 上界]，保证返回值是合法 Python float"""
    clean = series.dropna()
    if len(clean) < 2:
        # 数据太少，无法计算分位数
        return 0.0, 0.0

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    # 如果 IQR 为 0（所有值相同），稍微放宽范围
    if iqr == 0:
        low = q1 - abs(q1) * 0.01 if q1 != 0 else -0.01
        high = q1 + abs(q1) * 0.01 if q1 != 0 else 0.01
    else:
        low = q1 - multiplier * iqr
        high = q3 + multiplier * iqr

    # 确保不是 NaN / Inf
    import math
    if math.isnan(low) or math.isinf(low):
        low = 0.0
    if math.isnan(high) or math.isinf(high):
        high = 0.0

    return round(low, 6), round(high, 6)


def _detect_outliers_iqr(series, multiplier=1.5):
    """用 IQR 方法检测异常值，返回异常值所在的行索引列表（Python int）"""
    low, high = _get_iqr_bounds(series, multiplier)
    outlier_mask = (series < low) | (series > high)
    # 排除 NaN 行（NaN 比较结果始终为 False，但显式处理更安全）
    outlier_mask = outlier_mask & series.notna()
    return [int(i) for i in series[outlier_mask].index.tolist()]


def _safe_json_val(val):
    """将单个值转为 JSON 安全的 Python 类型"""
    import math
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if isinstance(val, (pd.Timestamp,)):
            return str(val)
        if isinstance(val, (pd.Timedelta,)):
            return str(val)
        if isinstance(val, (int, float, str, bool, type(None))):
            if isinstance(val, float):
                return round(val, 6)
            return val
        if pd.isna(val):
            return None
        return str(val)
    except Exception:
        return None
