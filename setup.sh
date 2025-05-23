#!/bin/bash

set -e

# ベースディレクトリ作成
mkdir -p bertopic-time-clustering-llm/data
mkdir -p bertopic-time-clustering-llm/notebooks

# sample_texts.csv
cat <<EOF > bertopic-time-clustering-llm/data/sample_texts.csv
timestamp,text
2023-01-01,今日は良い天気です。
2023-01-02,昨日のサッカーの試合は面白かった。
2023-01-10,新しい映画を見に行きました。
2023-02-01,今日は雪が降っています。
2023-02-05,最近はテクノロジーの話題が多いです。
EOF

# requirements.txt
cat <<EOF > bertopic-time-clustering-llm/requirements.txt
bertopic
pandas
scikit-learn
matplotlib
openai
EOF

# README.md
cat <<EOF > bertopic-time-clustering-llm/README.md
# BERTopic + 時系列クラスタリング + LLM ハンズオンリポジトリ

## 概要
- 日本語テキストに対してBERTopicでトピック抽出
- トピックごとの時系列的な頻度を集計し、簡単にクラスタリング
- OpenAI GPTを使ってトピック名を自動生成

## 使い方
1. 必要ライブラリをインストール  
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. OpenAI APIキーを設定（任意）  
   \`\`\`bash
   export OPENAI_API_KEY="your_api_key"
   \`\`\`

3. ノートブックを実行して結果を確認

## データ
- \`data/sample_texts.csv\` にサンプル時系列テキストを用意
EOF

# notebooks/bertopic_time_clustering_llm.ipynb (Pythonコードのみテキスト保存)
cat <<EOF > bertopic-time-clustering-llm/notebooks/bertopic_time_clustering_llm.ipynb
# 必要ライブラリのインポート
import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import openai
import os

# 1. データ読み込み
df = pd.read_csv("../data/sample_texts.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. BERTopicでトピック抽出
texts = df['text'].tolist()
topic_model = BERTopic(language="japanese")
topics, _ = topic_model.fit_transform(texts)
df['topic'] = topics

# 3. トピック×日付の頻度集計（月単位）
df['date'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
topic_time = df.groupby(['topic', 'date']).size().unstack(fill_value=0)

# 4. 時系列クラスタリング（KMeans）
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(topic_time)
topic_time['cluster'] = clusters

# 5. LLMでトピック名を自動生成
openai.api_key = os.getenv("OPENAI_API_KEY")

def label_topic(words):
    prompt = f"以下の単語群からトピック名を日本語で1つだけ教えてください:\\n{words}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

topic_labels = {}
for topic_num in set(topics):
    if topic_num == -1:
        continue
    words = topic_model.get_topic(topic_num)
    word_list = [w for w, _ in words]
    topic_labels[topic_num] = label_topic(word_list)

print("トピックラベル:", topic_labels)

# 6. 可視化
topic_time.drop(columns=['cluster']).T.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Topic Frequency Over Time")
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.show()
EOF

echo "bertopic-time-clustering-llm ディレクトリとファイル作成完了！"
