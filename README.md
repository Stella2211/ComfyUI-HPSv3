# ComfyUI-HPSv3

HPSv3++のNF4モデルを使い、画像とプロンプトの評価・画像からのプロンプト生成をローカルで行うComfyUI拡張です。

| ノード | 入力 | 動作・出力 |
| --- | --- | --- |
| HPSv3++ Model Loader | モデルフォルダ | 2つの処理ノードで使うモデル設定 |
| HPSv3++ Score | モデル、画像、プロンプト | スコア付きPNGを保存・プレビューし、IMAGEとFLOATを出力 |
| HPSv3++ Caption | モデル、画像 | 画像ごとに生成したプロンプトをSTRINGとして出力 |

## 動作環境

- BF16対応のNVIDIA CUDA GPU。12GB VRAMを目安にしてください。( 8GB でも動作する可能性はありますが未検証です。)
- CPU・AMD・Apple GPUには対応していません。
- CUDA 13.0対応のNVIDIAドライバー。
- ComfyUI、Git。
- モデル重み約6.5GBに加え、専用Python環境とインストール用キャッシュの空き容量。

## インストール

ComfyUI-Managerからこのリポジトリをインストールすると、`install.py`がサブモジュールと専用環境をセットアップします。モデル重みは自動ダウンロードしません。初回はPython・CUDA対応PyTorchなどのダウンロードに時間がかかります。

手動の場合は、`ComfyUI/custom_nodes`で以下を実行してください。

```sh
git clone https://github.com/Stella2211/ComfyUI-HPSv3.git
cd ComfyUI-HPSv3
uv run --no-project python install.py
```

次に、同じフォルダからモデルをダウンロードします。このコマンドを実行したときだけHugging Faceにアクセスします。

```sh
uv run --no-project --python .venv hf download stella221125/HPSv3-PlusPlus-bnb-NF4 --local-dir ../../models/hpsv3pp/HPSv3-PlusPlus-bnb-NF4
```

## 使い方

[サンプルワークフロー](examples/caption_and_score.json)をComfyUIへ読み込み、Load Imageの画像とModel Loaderのモデルを選ぶと、CaptionからScoreへ接続した状態で試せます。

### 画像を評価する

1. Model LoaderをScoreの`model`へ接続します。
2. Load ImageなどのIMAGEをScoreの`images`へ接続し、`prompt`に評価対象のプロンプトを入力します。
3. `score_mode`を選び、実行します。

- `banner`: 画像上部に白い余白を追加し、スコアを表示します。元画像は覆いません。
- `metadata`: 画像に文字を描かず、PNGの`hpsv3pp`テキスト項目へJSON形式でモデル名・スコア・評価プロンプトを保存します。
- `both`: 画像上部にスコアを表示し、同じPNGのメタデータにもモデル名・スコア・評価プロンプトを保存します。

標準のSave Imageと同じ日付・ノード値の置換を利用できます。例えば、`%date:yyyy%/%date:MM%/%date:dd%/HPSv3pp`は`output/2026/09/08/HPSv3pp_00001_.png`のように保存できます。

メタデータはこのノードが保存したPNGに付属します。出力のIMAGEを別のSave Imageへ接続しても、このスコアメタデータは引き継がれません。

複数画像には同じプロンプトを適用できます。プロンプトがリストの場合は画像と同じ件数を渡してください。IMAGE・FLOATは画像ごとのリストとして、入力順に出力します。

### 画像からプロンプトを生成する

Model Loaderと画像をCaptionへ接続します。`max_new_tokens`で生成の長さの上限を指定できます。元の生成プロンプトを復元する機能ではなく、画像をもとに短い説明を生成します。内容を確認してから利用してください。

Captionの出力をScoreの`prompt`へ接続すれば、生成した説明を使って評価できます。複数画像の場合は両ノードへ同じ画像を同じ順序で渡してください。

## ライセンス

拡張自身のコードは[MIT](LICENSE)です。推論には私が作成した[hpsv3-4bit](https://github.com/Stella2211/hpsv3-4bit)を使用します。HPSv3++の実装をコピーしているわけではなく、hpsv3-4bitを経由したGitサブモジュールとして参照します。上流実装とモデル重みは別の条件が適用されるため、利用・再配布前に[第三者ライセンス情報](THIRD_PARTY_NOTICES.md)と各参照先を確認してください。
