"""
analyze.py — 任务E：分析功能
============================
- 线性回归（R²、斜率、截距、预测值）
- K-Means 3D 聚类分析
- Flask Blueprint，通过 current_app.config 读取共享数据
"""

from flask import Blueprint, current_app, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

analyze_bp = Blueprint('analyze', __name__)


@analyze_bp.route('/analyze', methods=['POST'])
def analyze():
    """线性回归 + 可选 3D K-Means 聚类"""
    df = current_app.config.get('CURRENT_DF')
    if df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json
    x, y, z = data.get('x_col'), data.get('y_col'), data.get('z_col')
    k = data.get('n_clusters', 3)

    if x not in df.columns or y not in df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'}), 400
    if z and z not in df.columns:
        return jsonify({'code': 400, 'msg': f'列 {z} 不存在'}), 400

    try:
        # --- 线性回归 ---
        df_reg = df[[x, y]].dropna()
        X_arr, Y_arr = df_reg[[x]].values, df_reg[y].values
        if len(X_arr) < 2:
            return jsonify({'code': 400, 'msg': '数据点不足'}), 400

        model = LinearRegression().fit(X_arr, Y_arr)
        result = {
            'regression': {
                'r2_score': round(model.score(X_arr, Y_arr), 4),
                'slope': round(model.coef_[0], 4),
                'intercept': round(model.intercept_, 4),
                'predictions': model.predict(X_arr).tolist()
            }
        }

        # --- 3D 聚类 ---
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
        return jsonify({'code': 400, 'msg': f'分析失败：{str(e)}'}), 400


@analyze_bp.route('/analyze.html')
def go_analyze():
    return render_template('analyze.html')
