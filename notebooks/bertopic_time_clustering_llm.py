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
    prompt = f"以下の単語群からトピック名を日本語で1つだけ教えてください:\n{words}"
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
