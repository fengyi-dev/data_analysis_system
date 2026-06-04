from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import io
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
current_df = None

@app.route('/')
def index():
    return render_template('index.html')

# 上传接口
@app.route('/upload', methods=['POST'])
def upload():
    global current_df
    file = request.files['file']
    if file.filename.endswith('.csv'):
        current_df = pd.read_csv(file)
    elif file.filename.endswith(('.xlsx', '.xls')):
        current_df = pd.read_excel(file)
    else:
        return jsonify({'code': 400, 'msg': '只支持CSV/Excel文件'})
    
    preview = {
        'columns': current_df.columns.tolist(),
        'rows': current_df.head(10).values.tolist(),
        'shape': list(current_df.shape)
    }
    return jsonify({'code': 200, 'data': preview, 'msg': '上传成功'})

# 分析接口（线性回归）
@app.route('/analyze', methods=['POST'])
def analyze():
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'})

    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'})

    try:
        df = current_df[[x_col, y_col]].dropna()
        X = df[[x_col]].values
        y = df[y_col].values

        if len(X) < 2:
            return jsonify({'code': 400, 'msg': '有效数据点不足'})

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
        return jsonify({'code': 400, 'msg': f'分析失败：{str(e)}'})

# 清洗接口
@app.route('/clean', methods=['POST'])
def clean():
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '请先上传数据'})

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
        return jsonify({'code': 400, 'msg': '请先上传数据'})

    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')

    if x_col not in current_df.columns or y_col not in current_df.columns:
        return jsonify({'code': 400, 'msg': '所选列不存在'})

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
    global current_df
    if current_df is None:
        return jsonify({'code': 400, 'msg': '没有可导出的数据'}), 400
    output = io.BytesIO()
    current_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='cleaned_data.csv')

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
    