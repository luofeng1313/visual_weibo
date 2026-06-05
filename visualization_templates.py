"""情感分析可视化模板（使用Plotly）"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 配色方案
COLORS = {
    '正面': '#2E86AB',   # 蓝
    '中性': '#F18F01',   # 橙
    '负面': '#A23B72',   # 紫红
    '粉丝': '#1B998B',   # 绿松
    '路人': '#C9E4C5'    # 浅绿
}

def create_sentiment_pie(df, sentiment_col='sentiment_label'):
    """情感占比环形图"""
    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = [sentiment_col, 'count']
    fig = px.pie(
        counts,
        names=sentiment_col,
        values='count',
        title='情感分布环形图',
        color=sentiment_col,
        color_discrete_map=COLORS,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_fan_vs_public_bar(df, sentiment_col='sentiment_label', fan_col='is_fan'):
    """粉丝 vs 路人情感对比分组柱状图"""
    cross = pd.crosstab(df[fan_col], df[sentiment_col], normalize='index')
    cross = cross.reset_index().melt(id_vars=fan_col, var_name=sentiment_col, value_name='proportion')
    cross[fan_col] = cross[fan_col].map({True: '粉丝', False: '路人'})
    fig = px.bar(
        cross,
        x=sentiment_col,
        y='proportion',
        color=fan_col,
        barmode='group',
        title='粉丝与路人的情感倾向对比',
        labels={'proportion': '占比', sentiment_col: '情感', fan_col: '用户类型'},
        color_discrete_map={'粉丝': COLORS['粉丝'], '路人': COLORS['路人']}
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

def create_sentiment_timeline(df, date_col='created_at_datetime', sentiment_col='sentiment_label'):
    """情感演化折线图（按天聚合计数）"""
    df['date'] = pd.to_datetime(df[date_col]).dt.date
    timeline = df.groupby(['date', sentiment_col]).size().reset_index(name='count')
    fig = px.line(
        timeline,
        x='date',
        y='count',
        color=sentiment_col,
        title='情感演化趋势',
        labels={'count': '评论数', 'date': '日期', sentiment_col: '情感'},
        color_discrete_map=COLORS,
        markers=True
    )
    return fig

def create_sentiment_timeline_proportion(df, date_col='created_at_datetime', sentiment_col='sentiment_label'):
    """情感演化堆叠面积图（百分比）"""
    df['date'] = pd.to_datetime(df[date_col]).dt.date
    daily_counts = df.groupby(['date', sentiment_col]).size().unstack(fill_value=0)
    daily_prop = daily_counts.div(daily_counts.sum(axis=1), axis=0)
    fig = px.area(
        daily_prop.reset_index(),
        x='date',
        y=daily_prop.columns,
        title='情感占比演化（堆叠面积图）',
        labels={'value': '占比', 'date': '日期'},
        color_discrete_map=COLORS
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

def prepare_sankey_data(df, neutral_subtype_col='neutral_subtype'):
    """准备桑基图数据（中立亚型）"""
    sentiment_nodes = ['正面', '中性', '负面']
    # 请根据实际亚型名称调整
    neutral_sub_nodes = ['纯中性', '吃瓜型', '理性型', '隐蔽正面', '隐蔽负面']
    all_nodes = sentiment_nodes + neutral_sub_nodes
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    links = []
    neutral_df = df[df['sentiment_label'] == '中性'].copy()
    subtype_counts = neutral_df[neutral_subtype_col].value_counts()

    source_idx = node_indices['中性']
    for subtype, count in subtype_counts.items():
        if subtype in neutral_sub_nodes:
            target_idx = node_indices[subtype]
            links.append({'source': source_idx, 'target': target_idx, 'value': count})
    return all_nodes, links

def plot_sankey(df, neutral_subtype_col='neutral_subtype'):
    """绘制桑基图"""
    nodes, links = prepare_sankey_data(df, neutral_subtype_col)
    if not links:
        return None
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes, color="blue"
        ),
        link=dict(
            source=[l['source'] for l in links],
            target=[l['target'] for l in links],
            value=[l['value'] for l in links]
        )
    )])
    fig.update_layout(title="中立态度流向细分", font_size=12)
    return fig