import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import hurst


def compute_css(metric_values):
    # 标准差评分
    sds = 1 / np.std(metric_values)

    # 变化率评分
    roc = np.mean(np.abs(np.diff(metric_values)))
    rocs = 1 / roc

    # 自相关评分（滞后1）
    acs = acf(metric_values, nlags=1)[1]

    # 赫斯特指数评分
    h = hurst.compute_Hc(metric_values, kind='price')[0]
    hes = h

    # 综合评分 (权重可以根据需求调整)
    alpha, beta, gamma, delta = 0.25, 0.25, 0.25, 0.25
    css = alpha * sds + beta * rocs + gamma * acs + delta * hes

    return {
        'SDS': sds,
        'ROCS': rocs,
        'ACS': acs,
        'HES': hes,
        'CSS': css
    }


def calculate_css_for_directory(directory_path, column_name='acc'):
    css_scores = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            if column_name in data.columns:
                metric_values = data[column_name].dropna().to_numpy()
                css_score = compute_css(metric_values)
                css_scores.append((filename, css_score))
            else:
                print(f"Column '{column_name}' not found in {filename}. Skipping this file.")
    return css_scores


# 使用示例
directory_path = "/path/to/csv/files"
column_name = 'acc'  # 可以是'acc', 'pur', 'ami'等
css_scores = calculate_css_for_directory(directory_path, column_name=column_name)

for filename, scores in css_scores:
    print(f"File: {filename}")
    for key, value in scores.items():
        print(f"  {key}: {value}")
