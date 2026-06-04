import io
import os
import traceback

import chardet
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.linear_model import LinearRegression
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置文件上传限制：最大 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# 全局变量存储当前数据
current_df = None
current_filename = None

ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def safe_json_serialize(df, rows=10):
    """
    将 DataFrame 安全地转换为 JSON 可序列化格式。
    处理 NaN、Infinity、日期时间等特殊类型。
    """
    # 用 head() 避免复制整个 df
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
    """检测文件的编码格式"""
    # 保存文件指针位置
    pos = file_obj.tell()
    raw_data = file_obj.read(10000)
    file_obj.seek(pos)  # 恢复指针

    if not raw_data:
        return 'utf-8'

    result = chardet.detect(raw_data)
    encoding = result.get('encoding', 'utf-8') or 'utf-8'
    # 统一常见别名
    encoding = encoding.lower().replace('gb2312', 'gbk').replace('gb18030', 'gbk')
    return encoding


def detect_delimiter(file_obj, encoding='utf-8'):
    """自动检测 CSV 的分隔符"""
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

    # 取前几行（跳过空行）
    header_line = None
    for line in lines:
        line = line.strip()
        if line:
            header_line = line
            break

    if not header_line:
        return ','

    # 尝试常见分隔符，选解析出最多列的那个
    delimiters = [',', ';', '\t', '|']
    best_delim = ','
    best_count = 0

    for delim in delimiters:
        count = len(header_line.split(delim))
        if count > best_count:
            best_count = count
            best_delim = delim

    return best_delim


@app.route('/')
def index():
    return render_template('index.html')


# 上传接口
@app.route('/upload', methods=['POST'])
def upload():
    global current_df, current_filename

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
                current_df = pd.read_csv(file, encoding=encoding, delimiter=delimiter)
            except UnicodeDecodeError:
                # 编码检测失败时回退
                file.seek(0)
                current_df = pd.read_csv(file, encoding='utf-8', errors='replace', delimiter=delimiter)
        else:
            # Excel 文件
            try:
                current_df = pd.read_excel(file, engine='openpyxl')
            except Exception:
                file.seek(0)
                current_df = pd.read_excel(file, engine='xlrd')

        # 记录文件名
        current_filename = filename

        # 空数据检查
        if current_df.empty:
            return jsonify({'code': 400, 'msg': '文件为空，请检查数据'})

        preview = safe_json_serialize(current_df)
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

# 分析接口（线性回归）
@app.route('/analyze', methods=['POST'])
def analyze():
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'}), 400

    try:
        df = current_df[[x_col, y_col]].dropna()
        X = df[[x_col]].values
        y = df[y_col].values

        if len(X) < 2:
            return jsonify({'code': 400, 'msg': '有效数据点不足'}), 400

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        return jsonify({
            'code': 200,
            'data': {
                'r2_score': round(model.score(X, y), 4),
                'slope': round(model.coef_[0], 4),
                'intercept': round(model.intercept_, 4),
                'predictions': predictions.tolist()
            }
        })
    except Exception as e:
        return jsonify({'code': 400, 'msg': f'分析失败：{str(e)}'}), 400

# 清洗接口
@app.route('/clean', methods=['POST'])
def clean():
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json
    method = data.get('method', 'drop')  # drop / fill_mean / fill_median
    columns = data.get('columns', current_df.columns.tolist())

    df = current_df.copy()

    if method == 'drop':
        df = df[columns].dropna()
    elif method == 'fill_mean':
        for col in columns:
            if col in df.columns and df[col].dtype in ('float64', 'int64'):
                df[col] = df[col].fillna(df[col].mean())
    elif method == 'fill_median':
        for col in columns:
            if col in df.columns and df[col].dtype in ('float64', 'int64'):
                df[col] = df[col].fillna(df[col].median())

    current_df = df

    return jsonify({
        'code': 200,
        'data': {
            'columns': df.columns.tolist(),
            'rows': df.head(10).values.tolist(),
            'shape': list(df.shape),
            'null_count': int(df.isnull().sum().sum())
        },
        'msg': f'cleaning done, {df.shape[0]} rows x {df.shape[1]} cols'
    })

# 获取数据接口（用于图表绑定）
@app.route('/get_data', methods=['POST'])
def get_data():
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'}), 400

    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'}), 400

    df = current_df[[x_col, y_col]].dropna()
    return jsonify({
        'code': 200,
        'data': {
            'x_values': df[x_col].values.tolist(),
            'y_values': df[y_col].values.tolist()
        }
    })


# 导出接口
@app.route('/export', methods=['GET'])
def export():
    global current_df, current_filename
    if current_df is None:
        return jsonify({'code': 400, 'msg': '没有可导出的数据'}), 400

    output = io.BytesIO()
    current_df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)

    # 使用原始文件名或默认名
    download_name = current_filename or 'cleaned_data.csv'
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

# 自定义错误处理
@app.errorhandler(413)
def too_large(e):
    return jsonify({'code': 413, 'msg': '文件过大，最大支持 100MB'})

@app.route('/upload.html')
def go_upload():
    return render_template("upload.html")

@app.route('/clean.html')
def go_clean():
    return render_template("clean.html")

@app.route('/view.html')
def go_view():
    return render_template("view.html")

@app.route('/analyze.html')
def go_analyze():
    return render_template("analyze.html")

if __name__ == '__main__':
    app.run(debug=True)
    