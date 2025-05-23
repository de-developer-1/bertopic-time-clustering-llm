# BERTopic + 時系列クラスタリング + LLM ハンズオンリポジトリ

## 概要
このリポジトリは以下を実装した最小構成のハンズオン用サンプルです。

- 日本語テキストのBERTopicによるトピック抽出  
- トピックごとの時系列（月単位）頻度を集計し、KMeansによるクラスタリング  
- OpenAI GPT（gpt-4o-mini）を使って各トピックに自動ラベル付け

## ディレクトリ構成
bertopic-time-clustering-llm/
├── data/ # サンプルテキストデータ（CSV）
│ └── sample_texts.csv
├── notebooks/ # 分析用Jupyter Notebook（Pythonコードテキスト）
│ └── bertopic_time_clustering_llm.ipynb
├── requirements.txt # 必要パッケージリスト
└── README.md # このファイル

## 環境構築と実行手順
1.必要ライブラリをインストール
```bash
pip install -r requirements.txt
OpenAI APIキーを環境変数に設定（任意、トピックラベル付けで使用）
export OPENAI_API_KEY="your_api_key"
```

2.Jupyter Notebookを開いてコードを順に実行してください。

## 注意事項
LLMのAPI呼び出しには料金が発生する場合がありますのでご注意ください。

