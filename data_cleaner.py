"""
clean.py — 任务B：数据清洗
==========================
- 缺失值处理（删除 / 均值 / 中位数 / 众数 / 前向 / 后向填充）
- 异常值检测（IQR / Z-score）
- 数据质量分析 + 自动清洗
- 返回清洗后的数据预览
"""

import traceback

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request

clean_bp = Blueprint('clean', __name__)


# ===========================================================================
# 内部工具函数
# ===========================================================================

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


def _get_iqr_bounds(series, multiplier=1.5):
    """计算 IQR 正常范围 [下界, 上界]"""
    import math
    clean_vals = series.dropna()
    if len(clean_vals) < 2:
        return 0.0, 0.0

    q1 = float(clean_vals.quantile(0.25))
    q3 = float(clean_vals.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        low = q1 - abs(q1) * 0.01 if q1 != 0 else -0.01
        high = q1 + abs(q1) * 0.01 if q1 != 0 else 0.01
    else:
        low = q1 - multiplier * iqr
        high = q3 + multiplier * iqr

    if math.isnan(low) or math.isinf(low):
        low = 0.0
    if math.isnan(high) or math.isinf(high):
        high = 0.0

    return round(low, 6), round(high, 6)


def _detect_outliers_iqr(series, multiplier=1.5):
    """用 IQR 方法检测异常值，返回异常值所在的行索引列表"""
    low, high = _get_iqr_bounds(series, multiplier)
    outlier_mask = (series < low) | (series > high)
    outlier_mask = outlier_mask & series.notna()
    return [int(i) for i in series[outlier_mask].index.tolist()]


# ===========================================================================
# 数据质量分析
# ===========================================================================

def analyze_quality(df):
    """
    分析数据质量，返回每列的缺失值和异常值报告。
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

        is_real_numeric = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        if is_real_numeric:
            outliers = _detect_outliers_iqr(df[col])
            col_info['outlier_count'] = len(outliers)
            col_info['outlier_pct'] = round(len(outliers) / len(df) * 100, 2)
            col_info['outlier_method'] = 'iqr'
            col_info['outlier_bounds'] = _get_iqr_bounds(df[col])
            col_info['outlier_indices'] = outliers
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


# ===========================================================================
# 自动清洗
# ===========================================================================

def apply_cleaning(df, config):
    """
    根据配置执行清洗，返回 (清洗后DataFrame, 执行报告)。

    config 格式：
    {
        'missing_strategy': {
            '全局': 'drop' | 'fill_mean' | 'fill_median' | 'fill_mode' | 'fill_zero' | 'ignore',
            '列名A': 'fill_mean',
        },
        'outlier_strategy': {
            '全局': 'ignore' | 'remove' | 'cap',
            'method': 'iqr' | 'zscore',
            'iqr_multiplier': 1.5,
            'zscore_threshold': 3,
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
            report_lines.append(f'列 [{col}] 异常值截断：处理 {n_out} 个')

    total_after = len(df)
    dropped_rows = total_before - total_after

    return df, {
        'rows_before': total_before,
        'rows_after': total_after,
        'rows_dropped': dropped_rows,
        'details': report_lines
    }


# ===========================================================================
# Blueprint 路由
# ===========================================================================

@clean_bp.route('/clean', methods=['POST'])
def clean():
    """基础清洗：删除缺失行 / 均值填充 / 中位数填充"""
    df = current_app.config.get('CURRENT_DF')
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json
    method = data.get('method', 'drop')
    columns = data.get('columns', df.columns.tolist())

    df_copy = df.copy()

    if method == 'drop':
        df_copy = df_copy[columns].dropna()
    elif method == 'fill_mean':
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ('float64', 'int64'):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif method == 'fill_median':
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ('float64', 'int64'):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    current_app.config['CURRENT_DF'] = df_copy

    return jsonify({
        'code': 200,
        'data': {
            'columns': df_copy.columns.tolist(),
            'rows': df_copy.head(10).values.tolist(),
            'shape': list(df_copy.shape),
            'null_count': int(df_copy.isnull().sum().sum())
        },
        'msg': f'清洗完成，{df_copy.shape[0]} 行 × {df_copy.shape[1]} 列'
    })


@clean_bp.route('/data_quality', methods=['GET'])
def data_quality():
    """数据质量分析：缺失值 + 异常值检测报告"""
    df = current_app.config.get('CURRENT_DF')
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    try:
        report = analyze_quality(df)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': f'质量检测异常: {str(e)}'}), 500

    return jsonify({'code': 200, 'data': report})


@clean_bp.route('/auto_clean', methods=['POST'])
def auto_clean():
    """增强自动清洗：支持多种缺失值/异常值策略"""
    df = current_app.config.get('CURRENT_DF')
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    config = request.json
    before_shape = df.shape

    try:
        df, result = apply_cleaning(df, config)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': f'清洗出错: {str(e)}'}), 500

    current_app.config['CURRENT_DF'] = df

    after_shape = df.shape
    details = '\n'.join(result['details']) if result['details'] else '无需处理'

    return jsonify({
        'code': 200,
        'data': {
            'columns': df.columns.tolist(),
            'rows': df.head(10).values.tolist(),
            'shape': list(after_shape),
            'report': {
                'rows_before': result['rows_before'],
                'rows_after': result['rows_after'],
                'rows_dropped': result['rows_dropped'],
                'details': details
            }
        },
        'msg': f'自动清洗完成：{before_shape[0]}行 → {after_shape[0]}行（移除 {result["rows_dropped"]} 行）'
    })


@clean_bp.route('/clean.html')
def go_clean():
    return render_template('clean.html')
