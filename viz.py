from flask import Blueprint, request, jsonify, render_template, current_app
import pandas as pd
import numpy as np

viz_bp = Blueprint('viz', __name__)


def _series_to_list(s):
    # datetime -> ISO string, numpy types -> python native
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    return [None if (isinstance(v, float) and pd.isna(v)) else (v.tolist() if hasattr(v, 'tolist') else v)
            for v in s]


@viz_bp.route('/get_data', methods=['POST'])
def get_data():
    current_df = current_app.config.get('CURRENT_DF')
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json or {}
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    chart_type = (data.get('chart_type') or 'scatter').lower()
    max_points = int(data.get('max_points', 2000) or 2000)
    force_full = bool(data.get('force_full', False))

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'}), 400

    df = current_df[[x_col, y_col]].dropna().copy()
    original_n = len(df)

    # 默认上限，避免用户请求过多数据
    max_points = min(max_points, 20000)

    downsampled = False
    df_out = df

    if not force_full and original_n > max_points:
        downsampled = True
        try:
            # 对于折线图/散点图，更好的策略是按 X 排序后均匀抽取索引，保留趋势和顺序
            x_series = df[x_col]
            x_num = pd.to_numeric(x_series, errors='coerce')
            x_dt = pd.to_datetime(x_series, errors='coerce')

            is_numeric = x_num.notna().any()
            is_datetime = x_dt.notna().any()

            if is_numeric or is_datetime:
                # 按解析后的 x 排序，均匀取点
                if is_datetime:
                    df['_sort_x'] = x_dt
                else:
                    df['_sort_x'] = x_num

                df = df.sort_values('_sort_x').reset_index(drop=True)
                indices = np.linspace(0, len(df) - 1, max_points).astype(int)
                df_out = df.iloc[indices][[x_col, y_col]].reset_index(drop=True)
                # 清理临时列（如果存在）
                df_out = df_out.drop(columns=['_sort_x'], errors='ignore')
            else:
                # 非数值/时间型：折线按原序均匀抽样，散点随机采样，柱状取 top
                if chart_type == 'line':
                    df = df.reset_index(drop=True)
                    indices = np.linspace(0, len(df) - 1, max_points).astype(int)
                    df_out = df.iloc[indices][[x_col, y_col]].reset_index(drop=True)
                elif chart_type == 'scatter':
                    df_out = df.sample(n=max_points, random_state=42)[[x_col, y_col]].reset_index(drop=True)
                else:
                    top = df[x_col].value_counts().nlargest(max_points).index
                    df_out = df[df[x_col].isin(top)].groupby(x_col)[y_col].mean().reset_index()
        except Exception:
            # 回退到随机采样
            df_out = df.sample(n=max_points, random_state=42)[[x_col, y_col]].reset_index(drop=True)

    # 准备返回数据
    x_series = df_out[x_col]
    y_series = df_out[y_col]

    # 格式化
    x_values = _series_to_list(pd.to_datetime(x_series) if pd.api.types.is_datetime64_any_dtype(x_series) else x_series)
    y_values = _series_to_list(y_series)

    meta = {
        'original_points': int(original_n),
        'returned_points': int(len(x_values)),
        'downsampled': bool(downsampled)
    }

    return jsonify({
        'code': 200,
        'data': {
            'x_values': x_values,
            'y_values': y_values,
            'meta': meta
        }
    })


@viz_bp.route('/view.html')
def go_view():
    return render_template('view.html')
