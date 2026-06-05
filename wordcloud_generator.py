import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
import matplotlib.font_manager as fm

# 全局查找中文字体（只执行一次）
def get_chinese_font():
    # 常见 Linux 中文字体名称
    font_names = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
    for name in font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=name))
            if font_path and 'ttf' in font_path:
                return font_path
        except:
            continue
    return None

CHINESE_FONT_PATH = get_chinese_font()

def clean_text_for_wordcloud(text_series):
    """清洗文本：去除HTML标签、特殊字符、数字，保留中文字符和字母数字空格"""
    all_text = ' '.join(text_series.astype(str))
    all_text = re.sub(r'<[^>]+>', '', all_text)
    all_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', all_text)
    return all_text

def generate_wordcloud(text_series, title, width=800, height=400,
                       background_color='white', colormap='viridis'):
    """生成词云并返回matplotlib figure对象"""
    if text_series.empty or len(text_series) == 0:
        print(f"警告：{title} 文本为空，无法生成词云")
        return None
    text = clean_text_for_wordcloud(text_series)
    
    # 构建 WordCloud 参数，如果找到中文字体则使用，否则使用默认（可能方块但不报错）
    wc_params = {
        'width': width, 'height': height,
        'background_color': background_color,
        'colormap': colormap,
        'max_words': 100
    }
    if CHINESE_FONT_PATH:
        wc_params['font_path'] = CHINESE_FONT_PATH
    
    wordcloud = WordCloud(**wc_params).generate(text)
    
    fig, ax = plt.subplots(figsize=(width//100, height//100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    return fig