"""
upload.py — 任务A：数据上传 + 预览
====================================
功能：
  1. 文件上传（CSV / Excel），含编码自动检测与分隔符识别
  2. 读取后返回前10行预览 + 列名 + 数据类型
  3. 后端：Flask Blueprint 接收文件，pandas 读取

架构：Flask Blueprint，通过 current_app.config 与主应用共享 DataFrame
"""

import os
import traceback

import chardet
import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Blueprint 定义
# ---------------------------------------------------------------------------
upload_bp = Blueprint('upload', __name__)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def safe_json_serialize(df, rows=10):
    """
    将 DataFrame 安全地转换为 JSON 可序列化格式。
    处理 NaN、Infinity、日期时间等特殊类型。
    """
    preview_df = df.head(rows).copy()

    # 将 NaN 替换为 None，datetime 转为字符串
    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].where(preview_df[col].notna(), None)

    rows_data = preview_df.values.tolist()
    # 二次处理：确保每行中的 NaN/NaT 转为 None
    rows_data = [
        [None if (isinstance(cell, float) and pd.isna(cell)) or cell is pd.NA else cell
         for cell in row]
        for row in rows_data
    ]

    return {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'rows': rows_data,
        'shape': list(df.shape),
    }


def detect_encoding(file_obj):
    """检测文件的编码格式（用于中文 CSV 自动识别）"""
    pos = file_obj.tell()
    raw_data = file_obj.read(10000)
    file_obj.seek(pos)

    if not raw_data:
        return 'utf-8'

    result = chardet.detect(raw_data)
    encoding = result.get('encoding', 'utf-8') or 'utf-8'
    # 统一常见别名
    encoding = encoding.lower().replace('gb2312', 'gbk').replace('gb18030', 'gbk')
    return encoding


def detect_delimiter(file_obj, encoding='utf-8'):
    """自动检测 CSV 的分隔符（逗号、分号、制表符、竖线）"""
    pos = file_obj.tell()
    try:
        sample = file_obj.read(4096).decode(encoding)
    except (UnicodeDecodeError, UnicodeError):
        file_obj.seek(pos)
        sample = file_obj.read(4096).decode(encoding, errors='replace')
    file_obj.seek(pos)

    lines = sample.split('\n')
    if not lines:
        return ','

    header_line = None
    for line in lines:
        line = line.strip()
        if line:
            header_line = line
            break

    if not header_line:
        return ','

    delimiters = [',', ';', '\t', '|']
    best_delim = ','
    best_count = 0

    for delim in delimiters:
        count = len(header_line.split(delim))
        if count > best_count:
            best_count = count
            best_delim = delim

    return best_delim

# ---------------------------------------------------------------------------
# 路由：上传文件（任务A核心）
# ---------------------------------------------------------------------------

@upload_bp.route('/upload', methods=['POST'])
def upload():
    """
    接收上传的 CSV / Excel 文件：
    - 校验文件类型和大小
    - CSV：自动检测编码 + 分隔符
    - Excel：自动选择引擎（openpyxl / xlrd）
    - 返回前10行预览、列名、数据类型
    """
    # 校验是否有文件
    if 'file' not in request.files:
        return jsonify({'code': 400, 'msg': '请求中没有文件'})

    file = request.files['file']
    if not file.filename:
        return jsonify({'code': 400, 'msg': '未选择文件'})

    # 校验文件类型
    if not allowed_file(file.filename):
        return jsonify({'code': 400, 'msg': '只支持 CSV / Excel 文件（.csv/.xlsx/.xls）'})

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename.lower())[1]

    try:
        if ext == '.csv':
            # 自动检测编码和分隔符
            encoding = detect_encoding(file)
            delimiter = detect_delimiter(file, encoding)
            try:
                df = pd.read_csv(file, encoding=encoding, delimiter=delimiter)
            except UnicodeDecodeError:
                # 编码检测失败时回退
                file.seek(0)
                df = pd.read_csv(file, encoding='utf-8', errors='replace', delimiter=delimiter)
        else:
            # Excel 文件
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except Exception:
                file.seek(0)
                df = pd.read_excel(file, engine='xlrd')

        # 空数据检查
        if df.empty:
            return jsonify({'code': 400, 'msg': '文件为空，请检查数据'})

        # 存储到 app.config 供其他模块（分析、可视化）使用
        current_app.config['CURRENT_DF'] = df
        current_app.config['CURRENT_FILENAME'] = filename

        # 构建预览响应
        preview = safe_json_serialize(df)
        preview['filename'] = filename

        return jsonify({
            'code': 200,
            'data': preview,
            'msg': f'上传成功 — 共 {preview["shape"][0]} 行 × {preview["shape"][1]} 列'
        })

    except pd.errors.EmptyDataError:
        return jsonify({'code': 400, 'msg': '文件为空，无法读取'})
    except pd.errors.ParserError as e:
        return jsonify({'code': 400, 'msg': f'文件解析失败：{str(e)}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': f'读取文件失败：{str(e)}'})
