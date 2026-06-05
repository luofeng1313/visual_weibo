import pandas as pd
df = pd.read_csv('bailu_clean.csv')
print("情感标签分布：", df['sentiment_label'].value_counts())
print("neutral_subtype 非空数量：", df['neutral_subtype'].notna().sum())
print("neutral_subtype 唯一值：", df['neutral_subtype'].dropna().unique())
print("is_fan 分布：", df['is_fan'].value_counts())
print("created_at 示例：", df['created_at'].head())