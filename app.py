import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import os
import pickle
import io          
import warnings
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from visualization_templates import (
    create_sentiment_pie,
    create_fan_vs_public_bar,
    create_sentiment_timeline,
    plot_sankey,
    COLORS
)
from wordcloud_generator import generate_wordcloud
from lda_analysis import get_topic_keywords
warnings.filterwarnings('ignore')

# ----------------------------- 页面配置 ---------------------------------
st.set_page_config(
    page_title="白鹿舆情情感分析看板",
    page_icon="📊",
    layout="wide"
)

st.title("🎭 白鹿综艺表态风波舆情分析")
st.markdown("> 基于微博评论的情感分析与公众态度洞察")

# ----------------------------- 数据加载 ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bailu_clean.csv", encoding="utf-8-sig")
    df['created_at_datetime'] = pd.to_datetime(df['created_at'])
    label_map = {'positive': '正面', 'negative': '负面', 'neutral': '中性'}
    if df['sentiment_label'].dtype == 'object':
        df['sentiment_label'] = df['sentiment_label'].map(label_map).fillna(df['sentiment_label'])
    df['is_fan'] = df['is_fan'].astype(bool)
    return df

try:
    df = load_data()
    st.success(f"✅ 数据加载成功！共 {len(df)} 条评论")
    st.dataframe(df.head())
except FileNotFoundError:
    st.stop()

st.write("### 调试信息")
st.write("情感标签分布（前5）：", df['sentiment_label'].value_counts().head())
st.write("正面评论数量：", len(df[df['sentiment_label'] == '正面']))
st.write("中性评论数量：", len(df[df['sentiment_label'] == '中性']))
st.write("text 列是否为空？", df['text'].isna().sum())
st.write("neutral_subtype 非空数量：", df['neutral_subtype'].notna().sum())
# ----------------------------- 关键指标 ---------------------------------
total_comments = len(df)
positive_rate = (df['sentiment_label'] == '正面').mean()
fan_rate = (df['is_fan']).mean()

col1, col2, col3 = st.columns(3)
col1.metric("📝 总评论数", total_comments)
col2.metric("😊 正面评论占比", f"{positive_rate:.1%}")
col3.metric("👥 粉丝评论占比", f"{fan_rate:.1%}")

st.markdown("---")

# ----------------------------- 整体情感分布（环形图） -------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("🍩 整体情感分布") 
    fig_pie = create_sentiment_pie(df)
    st.plotly_chart(fig_pie, use_container_width=True)
    

with col2:
    st.subheader("👥 粉丝 vs 路人态度对比")
    fig_bar = create_fan_vs_public_bar(df)
    st.plotly_chart(fig_bar, use_container_width=True)
   
# ----------------------------- 情感演化折线图 ---------------------------
st.subheader("📈 情感演化趋势")
fig_line = create_sentiment_timeline(df)
st.plotly_chart(fig_line, use_container_width=True)

# ----------------------------- 词云（正/负） ----------------------------
st.subheader("📝 评论词云对比")
col1, col2 = st.columns(2)
# 提取正面和负面评论的文本
positive_text = df[df['sentiment_label'] == '正面']['text']
negative_text = df[df['sentiment_label'] == '负面']['text']

with col1:
    st.markdown("##### 👍 正面评论词云")
    fig_pos = generate_wordcloud(positive_text, "正面评论词云")
    if fig_pos:
        st.pyplot(fig_pos)
        # 下载按钮
        img_bytes = io.BytesIO()
        fig_pos.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        st.download_button(
            label="📥 下载正面词云",
            data=img_bytes,
            file_name="正面词云图.png",
            mime="image/png",
            key='download_pos'
        )
    else:
        st.info("暂无正面评论数据")

with col2:
    st.markdown("##### 👎 负面评论词云")
    fig_neg = generate_wordcloud(negative_text, "负面评论词云")
    if fig_neg:
        st.pyplot(fig_neg)
        img_bytes = io.BytesIO()
        fig_neg.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        st.download_button(
            label="📥 下载负面词云",
            data=img_bytes,
            file_name="负面词云图.png",
            mime="image/png",
            key='download_neg'
        )
    else:
        st.info("暂无负面评论数据")

# ----------------------------- 中立亚型桑基图 ---------------------------
st.subheader("🔀 中立态度的流向与细分")

df_neutral = df[df['sentiment_label'] == '中性'].copy()
if not df_neutral.empty and df_neutral['neutral_subtype'].notna().any():
    from visualization_templates import plot_sankey
    fig_sankey = plot_sankey(df_neutral)
    if fig_sankey:
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("无有效中立亚型数据")
else:
    st.info("暂无中立亚型数据")


# ----------------------------- 中立亚型时间演化（堆叠面积图） -----------
st.subheader("📊 中立亚型随时间演化")
df_neutral_time = df[df['sentiment_label'] == '中性'].copy()
if not df_neutral_time.empty:
    df_neutral_time['date'] = df_neutral_time['created_at_datetime'].dt.date
    subtype_daily = df_neutral_time.groupby(['date', 'neutral_subtype']).size().unstack(fill_value=0)
    subtype_daily_prop = subtype_daily.div(subtype_daily.sum(axis=1), axis=0)
    fig_area = px.area(
        subtype_daily_prop.reset_index(),
        x='date',
        y=subtype_daily_prop.columns,
        title='中立亚型占比演化（堆叠面积图）',
        labels={'value': '占比', 'date': '日期'},
        color_discrete_map={
            '纯中性': '#A0A0A0',
            '吃瓜型': "#39DB77",
            '理性型': "#C275EB",
            '隐蔽正面': "#E0E84C",
            '隐蔽负面': "#D3573E"
        }
    )
    fig_area.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig_area, use_container_width=True)
else:
    st.info("无中性评论数据")

# ----------------------------- LDA 主题建模 -----------------
st.subheader("📚 中立评论主题挖掘（LDA）")
required_files = ['lda_model.pkl', 'dictionary.pkl', 'lda_visualization.html', 'topic_timeline.png']
with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)
with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
topic_keywords = get_topic_keywords(lda_model)
if topic_keywords:
    cols = st.columns(len(topic_keywords))
    for i, (col, kw) in enumerate(zip(cols, topic_keywords)):
        col.info(f"**主题{i+1}** : " + "、".join(kw))
with open('lda_visualization.html', 'r', encoding='utf-8') as f:
    html_str = f.read()
st.components.v1.html(html_str, height=600)
st.image('topic_timeline.png')


# ----------------------------- 页脚 ------------------------------------
st.markdown("---")
st.caption("数据来源：微博评论 | 分析工具：Streamlit + Plotly + WordCloud + SnowNLP")