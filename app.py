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
plt.rcParams['font.sans-serif'] = ['SimHei']  #  Windows
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 导入自定义模块
from visualization_templates import (
    create_sentiment_pie,
    create_fan_vs_public_bar,
    create_sentiment_timeline,
    plot_sankey,
    create_sentiment_timeline_proportion,   
    create_neutral_subtype_timeline,        
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
    # 处理日期
    df['created_at_datetime'] = pd.to_datetime(df['created_at'])
    # 处理情感标签（强制转换）
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.lower().str.strip()
    df.loc[df['sentiment_label'] == 'positive', 'sentiment_label'] = '正面'
    df.loc[df['sentiment_label'] == 'negative', 'sentiment_label'] = '负面'
    df.loc[df['sentiment_label'] == 'neutral', 'sentiment_label'] = '中性'
    # 处理粉丝列
    df['is_fan'] = df['is_fan'].astype(bool)
    return df

try:
    df = load_data()
    st.success(f"✅ 数据加载成功！共 {len(df)} 条评论")
    st.dataframe(df.head())
except FileNotFoundError:
    st.stop()

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

# ----------------------------- 整体情感堆叠面积图（百分比） -----------------
st.subheader("📊 整体情感占比演化")
fig_prop = create_sentiment_timeline_proportion(df)
st.plotly_chart(fig_prop, use_container_width=True)

# ----------------------------- 词云（正/负） ----------------------------
st.subheader("📝 评论词云对比")
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### 👍 正面评论词云")
    if os.path.exists("positive_wordcloud.png"):
        st.image("positive_wordcloud.png", use_container_width=True)
        # 可选：提供下载按钮
        with open("positive_wordcloud.png", "rb") as f:
            st.download_button("📥 下载正面词云", f, file_name="positive_wordcloud.png", mime="image/png")
    else:
        st.info("正面词云图片未找到")
with col2:
    st.markdown("##### 👎 负面评论词云")
    if os.path.exists("negative_wordcloud.png"):
        st.image("negative_wordcloud.png", use_container_width=True)
        with open("negative_wordcloud.png", "rb") as f:
            st.download_button("📥 下载负面词云", f, file_name="negative_wordcloud.png", mime="image/png")
    else:
        st.info("负面词云图片未找到")
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
fig_area = create_neutral_subtype_timeline(df)
if fig_area:
    st.plotly_chart(fig_area, use_container_width=True)
else:
    st.info("无中性评论数据")

# ----------------------------- 页脚 ------------------------------------
st.markdown("---")
st.caption("数据来源：微博评论 | 分析工具：Streamlit + Plotly + WordCloud + SnowNLP")