"""
app.py — 数据分析系统主入口
=============================
任务E：Flask 整体框架搭建，注册五个功能模块 Blueprint。

模块文件对应关系：
  upload.py   → 任务A：数据上传 + 预览
  data_cleaner.py → 任务B：数据清洗（缺失值 / 异常值 / 质量分析）
  viz.py      → 任务C：可视化（图表数据接口）
  export.py   → 任务D：数据导出 + 页面路由 + 错误处理
  analyze.py  → 任务E：分析功能（线性回归 + K-Means 聚类）

数据共享：通过 app.config['CURRENT_DF'] 在各模块间传递当前 DataFrame
"""

from flask import Flask

# ---------------------------------------------------------------------------
# 创建应用 & 全局配置
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100MB 上传限制
app.config['CURRENT_DF'] = None                         # 当前数据（各模块共享）
app.config['CURRENT_FILENAME'] = None                   # 当前文件名

# ---------------------------------------------------------------------------
# 注册五模块 Blueprint
# ---------------------------------------------------------------------------
from upload import upload_bp           # 任务A：数据上传 + 预览
from data_cleaner import clean_bp      # 任务B：数据清洗
from viz import viz_bp                 # 任务C：可视化
from export import export_bp           # 任务D：导出 + 界面整合
from analyze import analyze_bp         # 任务E：分析功能

app.register_blueprint(upload_bp)
app.register_blueprint(clean_bp)
app.register_blueprint(viz_bp)
app.register_blueprint(export_bp)
app.register_blueprint(analyze_bp)

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
