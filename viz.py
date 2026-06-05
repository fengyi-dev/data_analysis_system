from flask import Blueprint, request, jsonify, render_template, current_app
import pandas as pd
import numpy as np

viz_bp = Blueprint('viz', __name__)


def _series_to_list(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    return [None if pd.isna(value) else value for value in series.tolist()]


def _guess_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    if pd.to_datetime(series, errors='coerce').notna().any():
        return 'datetime'
    if pd.to_numeric(series, errors='coerce').notna().any():
        return 'numeric'
    return 'category'


def _bin_series(series, max_bins=30):
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_numeric(series, errors='coerce').dropna()
        if values.empty:
            return [], np.array([], dtype=int)
        if values.nunique() <= max_bins:
            labels = values.astype(str).unique().tolist()
            codes = pd.Categorical(values.astype(str), categories=labels).codes
            return labels, codes

        edges = np.linspace(values.min(), values.max(), max_bins + 1)
        labels = [str(round((edges[i] + edges[i + 1]) / 2, 3)) for i in range(max_bins)]
        codes = pd.cut(values, bins=edges, include_lowest=True, labels=False).astype(int).to_numpy()
        return labels, codes

    values = series.astype(str)
    labels = values.value_counts().nlargest(max_bins).index.tolist()
    label_map = {label: index for index, label in enumerate(labels)}
    codes = values.map(label_map).fillna(-1).astype(int).to_numpy()
    return labels, codes


def _build_heatmap_response(df, x_col, y_col, value_col=None):
    columns = [x_col, y_col] + ([value_col] if value_col in df.columns else [])
    df = df[columns].dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return {'x_labels': [], 'y_labels': [], 'heatmap_data': [], 'min_value': 0, 'max_value': 0, 'value_label': value_col or 'count'}

    x_labels, x_codes = _bin_series(df[x_col])
    y_labels, y_codes = _bin_series(df[y_col])
    if not x_labels or not y_labels:
        return {'x_labels': [], 'y_labels': [], 'heatmap_data': [], 'min_value': 0, 'max_value': 0, 'value_label': value_col or 'count'}

    df['_x'] = x_codes
    df['_y'] = y_codes
    df = df[(df['_x'] >= 0) & (df['_y'] >= 0)]

    if value_col in df.columns:
        agg = df.groupby(['_y', '_x'])[value_col].mean().unstack(fill_value=0)
    else:
        agg = df.groupby(['_y', '_x']).size().unstack(fill_value=0)

    agg = agg.reindex(index=range(len(y_labels)), columns=range(len(x_labels)), fill_value=0)
    heatmap_data = [[x, y, float(agg.iat[y, x])] for y in range(len(y_labels)) for x in range(len(x_labels))]
    values = [item[2] for item in heatmap_data] or [0]

    return {
        'x_labels': x_labels,
        'y_labels': y_labels,
        'heatmap_data': heatmap_data,
        'min_value': min(values),
        'max_value': max(values),
        'value_label': value_col or 'count'
    }


@viz_bp.route('/get_data', methods=['POST'])
def get_data():
    current_df = current_app.config.get('CURRENT_DF')
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    params = request.json or {}
    x_col = params.get('x_col')
    y_col = params.get('y_col')
    chart_type = (params.get('chart_type') or 'scatter').lower()
    value_col = params.get('value_col')

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'}), 400

    if chart_type == 'heatmap':
        return jsonify({'code': 200, 'data': _build_heatmap_response(current_df, x_col, y_col, value_col)})

    df = current_df[[x_col, y_col]].dropna().copy()
    return jsonify({'code': 200, 'data': {'x_values': _series_to_list(df[x_col]), 'y_values': _series_to_list(df[y_col])}})


@viz_bp.route('/view.html')
def go_view():
    return render_template('view.html')
