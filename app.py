from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import io
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# 全局变量存储当前数据
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
        'shape': current_df.shape
    }
    return jsonify({'code': 200, 'data': preview, 'msg': '上传成功'})

# 分析接口（线性回归）
@app.route('/analyze', methods=['POST'])
def analyze():
    global current_df
    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    
    df = current_df[[x_col, y_col]].dropna()
    X = df[[x_col]].values
    y = df[y_col].values
    
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

# 获取数据接口（用于图表绑定）
@app.route('/get_data', methods=['POST'])
def get_data():
    global current_df
    data = request.json
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    
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
    output = io.BytesIO()
    current_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='cleaned_data.csv')

if __name__ == '__main__':
    app.run(debug=True)