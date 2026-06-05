"""词云生成脚本"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
import matplotlib.font_manager as fm

try:
    # 尝试使用开源中文字体
    font_path = fm.findfont(fm.FontProperties(family='sans-serif'))
except:
    font_path = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def clean_text_for_wordcloud(text_series):
    """清洗文本：去除HTML标签、特殊字符、数字，保留中文字符和字母数字空格"""
    all_text = ' '.join(text_series.astype(str))
    # 移除HTML标签
    all_text = re.sub(r'<[^>]+>', '', all_text)
    # 保留中文、字母、数字、空格，其它移除
    all_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', all_text)
    return all_text

def generate_wordcloud(text_series, title, width=800, height=400,
                       background_color='white', colormap='viridis'):
    """生成词云并返回matplotlib figure对象"""
    if text_series.empty or len(text_series) == 0:
        print(f"警告：{title} 文本为空，无法生成词云")
        return None
    text = clean_text_for_wordcloud(text_series)
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(width//100, height//100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)   # 现在标题会正常显示中文
    return fig