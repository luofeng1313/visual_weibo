"""LDA主题建模分析（中立评论）- 优化版"""
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora, models
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 配置 ==================
DATA_FILE = "bailu_clean.csv"         
TEXT_COL = "text"                
DATE_COL = "created_at"       
SENTI_COL = "sentiment_label"   

# ================== 增强的停用词表 ==================
def load_stopwords(path='stopwords.txt'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
    except:
        # 默认停用词（大幅扩充）
        stopwords = set([
            # 通用虚词
            '的', '了', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们', '这', '那', '有', '在', 
            '不', '也', '就', '都', '说', '看', '去', '来', '什么', '不是', '回复', '粉丝', '就是', '这么', 
            '自己', '没有', '不要', '知道', '一句', '怎么', '真的', '一个', '这个', '那个', '可以', '还是',
            '因为', '所以', '但是', '如果', '然后', '而且', '或者', '已经', '成为', '觉得', '认为', '应该',
            '可能', '所有', '一些', '这些', '那些', '不会', '不能', '没有', '还有', '等等', '给', '到', '跟',
            '把', '让', '被', '对', '对于', '关于', '与', '和', '及', '而已', '罢了', '哦', '啊', '嗯', '哈',
            '吧', '吗', '呢', '啦', '哟', '呵', '嘻', '哎', '呀', '哇', '哦哦', '哈哈', '嘿嘿', '呵呵','什么', 
            '不是', '回复', '粉丝', '就是', '这么', '自己', '没有', '不要', '知道', '一句', '怎么', '真的',
            # 表情和HTML残留
            'https', 'http', 'span', 'class', 'url', 'icon', 'img', 'alt', 'src', 'emoji', 'br', 'nbsp',
            # 数字和单字
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'
        ])
    return stopwords

def clean_text(text):
    """清洗：去除HTML标签、表情符号、特殊字符"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text

def segment(text, stopwords):
    words = jieba.lcut(text)
    # 过滤：停用词、长度<=1、纯数字
    words = [w for w in words if w not in stopwords and len(w) > 1 and not w.isdigit()]
    return words

def prepare_corpus(df, text_col=TEXT_COL, max_docs=None):
    if max_docs:
        df = df.head(max_docs)
    stopwords = load_stopwords()
    texts = []
    for t in df[text_col].dropna():
        t_clean = clean_text(t)
        if len(t_clean) < 5:
            continue
        words = segment(t_clean, stopwords)
        if words:
            texts.append(words)
    dictionary = corpora.Dictionary(texts)
    # 过滤：词频低于10次 或 出现在超过30%的文档中
    dictionary.filter_extremes(no_below=15, no_above=0.2)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return texts, dictionary, corpus

def train_lda(corpus, dictionary, num_topics=4, passes=20):
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary,
                                num_topics=num_topics, passes=passes,
                                random_state=42, eta='auto')
    return lda_model

def get_topic_keywords(lda_model, num_words=8):
    topics = lda_model.print_topics(num_words=num_words)
    topic_keywords = []
    for topic_id, topic_str in topics:
        # 提取关键词（更健壮）
        keywords = []
        parts = topic_str.split(' + ')
        for part in parts:
            if '*' in part:
                word = part.split('*')[1].strip('"')
                keywords.append(word)
        topic_keywords.append(keywords)
    return topic_keywords

def plot_topic_bubble(lda_model, dictionary, corpus):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("✅ 气泡图已保存为 lda_visualization.html")

def plot_topic_timeline(df, lda_model, dictionary, save_path='topic_timeline.png'):
    df['date'] = pd.to_datetime(df[DATE_COL]).dt.date
    stopwords = load_stopwords()
    topic_props = []
    for idx, row in df.iterrows():
        t = clean_text(row[TEXT_COL])
        if len(t) < 5:
            continue
        words = segment(t, stopwords)
        if not words:
            continue
        bow = dictionary.doc2bow(words)
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        probs = [0]*lda_model.num_topics
        for topic_id, prob in topic_dist:
            probs[topic_id] = prob
        topic_props.append({'date': row['date'], 'topic_probs': probs})
    if not topic_props:
        print("⚠️ 没有有效的文档，无法生成时序热力图")
        return
    df_topic = pd.DataFrame(topic_props)
    for t in range(lda_model.num_topics):
        df_topic[f'topic_{t}'] = df_topic['topic_probs'].apply(lambda x: x[t])
    daily_mean = df_topic.groupby('date')[[f'topic_{t}' for t in range(lda_model.num_topics)]].mean()
    plt.figure(figsize=(12, 6))
    sns.heatmap(daily_mean.T, cmap='YlOrRd', annot=True, fmt='.2f',
                cbar_kws={'label': '平均主题占比'})
    plt.xlabel('日期')
    plt.ylabel('主题编号')
    plt.title('中立评论主题随时间演化热力图')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 热力图已保存为 {save_path}")

# ================== 主程序 ==================
if __name__ == '__main__':
    print("📂 加载数据...")
    df_clean = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    df_clean[DATE_COL] = pd.to_datetime(df_clean[DATE_COL])
    print(f"总数据量: {len(df_clean)}")

    # 筛选中性评论（注意：你的标签列可能是中文'中性'还是英文'neutral'？）
    # 如果你的数据中 sentiment_label 已经是中文，请改用 '中性'
    # 此处假设为 'neutral'（根据你本地输出）
    df_neutral = df_clean[df_clean[SENTI_COL] == 'neutral'].copy()
    print(f"📌 中立评论数量: {len(df_neutral)}")

    if len(df_neutral) < 100:
        print("⚠️ 中立评论数量过少，无法进行 LDA 建模。")
        exit()

    print("🔄 正在准备语料（分词、去停用词）...")
    texts, dictionary, corpus = prepare_corpus(df_neutral, max_docs=None)  # 不限制数量
    print(f"有效文档数: {len(texts)}")

    print("🎯 训练 LDA 模型（主题数=4，passes=20）...")
    lda_model = train_lda(corpus, dictionary, num_topics=4, passes=20)

    # 保存模型和词典
    with open('lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    print("💾 模型和词典已保存为 lda_model.pkl, dictionary.pkl")

    # 输出主题关键词
    topic_keywords = get_topic_keywords(lda_model)
    print("\n📖 主题关键词：")
    for i, kw in enumerate(topic_keywords):
        print(f"主题{i+1}: {', '.join(kw)}")

    # 生成可视化文件
    plot_topic_bubble(lda_model, dictionary, corpus)
    plot_topic_timeline(df_neutral, lda_model, dictionary, save_path='topic_timeline.png')

    print("\n🎉 LDA 建模完成！")