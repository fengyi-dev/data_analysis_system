"""
app.py — 数据分析系统主入口
=============================
注册各功能模块的 Blueprint，提供共用配置和路由。
各模块文件：
  - upload.py  → 任务A：数据上传 + 预览
  - viz.py     → 任务C：可视化
  - data_cleaner.py → 任务B：数据清洗（质量分析 + 自动清洗）
"""

import traceback

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 注册 Blueprint
# ---------------------------------------------------------------------------
from upload import upload_bp
from viz import viz_bp
from data_cleaner import analyze_quality, apply_cleaning

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100MB 上传限制
app.config['CURRENT_DF'] = None                         # 当前数据（各模块共享）
app.config['CURRENT_FILENAME'] = None                   # 当前文件名

app.register_blueprint(upload_bp)
app.register_blueprint(viz_bp)

# ---------------------------------------------------------------------------
# 工具函数：获取当前 DataFrame
# ---------------------------------------------------------------------------

def get_current_df():
    """从 app.config 获取当前 DataFrame（统一入口）"""
    return app.config.get('CURRENT_DF')

def get_current_filename():
    """从 app.config 获取当前文件名"""
    return app.config.get('CURRENT_FILENAME')

# ---------------------------------------------------------------------------
# 路由：分析（线性回归 + 3D 聚类）
# ---------------------------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    df = get_current_df()
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'})

    data = request.json
    x, y, z = data.get('x_col'), data.get('y_col'), data.get('z_col')
    k = data.get('n_clusters', 3)

    if x not in df.columns or y not in df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'})
    if z and z not in df.columns:
        return jsonify({'code': 400, 'msg': f'列 {z} 不存在'})

    try:
        # --- 线性回归 ---
        df_reg = df[[x, y]].dropna()
        X, Y = df_reg[[x]].values, df_reg[y].values
        if len(X) < 2:
            return jsonify({'code': 400, 'msg': '数据点不足'})
        model = LinearRegression().fit(X, Y)

        result = {
            'regression': {
                'r2_score': round(model.score(X, Y), 4),
                'slope': round(model.coef_[0], 4),
                'intercept': round(model.intercept_, 4),
                'predictions': model.predict(X).tolist()
            }
        }

        # --- 3D 聚类（当提供了 z 列时） ---
        if z:
            df_cls = df[[x, y, z]].dropna()
            if len(df_cls) >= k:
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(df_cls.values)
                result['cluster_3d'] = {
                    'labels': labels.tolist(),
                    'features': [x, y, z],
                    'data': df_cls.values.tolist(),
                    'n_clusters': k
                }
            else:
                result['cluster_3d'] = {'error': f'数据点不足，至少需要{k}行'}

        return jsonify({'code': 200, 'data': result})

    except Exception as e:
        return jsonify({'code': 400, 'msg': f'分析失败：{str(e)}'})

# ---------------------------------------------------------------------------
# 路由：清洗（任务B 基础版）
# ---------------------------------------------------------------------------

@app.route('/clean', methods=['POST'])
def clean():
    df = get_current_df()
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

    app.config['CURRENT_DF'] = df_copy

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

# ---------------------------------------------------------------------------
# 路由：数据质量分析（任务B 增强版）
# ---------------------------------------------------------------------------

@app.route('/data_quality', methods=['GET'])
def data_quality():
    df = get_current_df()
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    try:
        report = analyze_quality(df)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': f'质量检测异常: {str(e)}'}), 500

    return jsonify({'code': 200, 'data': report})

# ---------------------------------------------------------------------------
# 路由：自动清洗（任务B 增强版）
# ---------------------------------------------------------------------------

@app.route('/auto_clean', methods=['POST'])
def auto_clean():
    df = get_current_df()
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    config = request.json
    before_shape = df.shape

    try:
        df, result = apply_cleaning(df, config)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': f'清洗出错: {str(e)}'}), 500

    app.config['CURRENT_DF'] = df

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

# ---------------------------------------------------------------------------
# 路由：页面导航
# ---------------------------------------------------------------------------

@app.route('/clean.html')
def go_clean():
    return render_template('clean.html')

@app.route('/analyze.html')
def go_analyze():
    return render_template('analyze.html')

# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return jsonify({'code': 413, 'msg': '文件过大，最大支持 100MB'})

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
