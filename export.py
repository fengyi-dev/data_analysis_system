"""
export.py — 任务D：数据导出 + 界面整合
======================================
- 导出 CSV 功能
- 整体页面路由（首页、各模块入口）
- 错误处理
"""

import io

from flask import Blueprint, current_app, jsonify, render_template, send_file

export_bp = Blueprint('export', __name__)


# ---------------------------------------------------------------------------
# 首页
# ---------------------------------------------------------------------------

@export_bp.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# 各模块页面入口
# ---------------------------------------------------------------------------

@export_bp.route('/upload.html')
def go_upload():
    return render_template('upload.html')

# clean.html / view.html / analyze.html 页面路由分别由各自蓝图处理


# ---------------------------------------------------------------------------
# 导出 CSV
# ---------------------------------------------------------------------------

@export_bp.route('/export', methods=['GET'])
def export():
    """将当前 DataFrame 导出为 CSV 文件下载"""
    df = current_app.config.get('CURRENT_DF')
    filename = current_app.config.get('CURRENT_FILENAME')

    if df is None:
        return jsonify({'code': 400, 'msg': '没有可导出的数据'}), 400

    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    download_name = filename or 'cleaned_data.csv'
    if download_name.endswith(('.xlsx', '.xls')):
        download_name = download_name.rsplit('.', 1)[0] + '.csv'
    elif not download_name.endswith('.csv'):
        download_name += '.csv'

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=download_name
    )


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

@export_bp.app_errorhandler(413)
def too_large(e):
    return jsonify({'code': 413, 'msg': '文件过大，最大支持 100MB'}), 413
