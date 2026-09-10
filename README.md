# ComfyUI-HPSv3

HPSv3++のNF4モデルを使い、画像とプロンプトの評価・画像からのプロンプト生成をローカルで行うComfyUI拡張です。

| ノード | 入力 | 動作・出力 |
| --- | --- | --- |
| HPSv3++ Model Loader | モデルフォルダ | 2つの処理ノードで使うモデル設定 |
| HPSv3++ Score | モデル、画像、プロンプト | スコア付きPNGを保存・プレビューし、IMAGEとFLOATを出力 |
| HPSv3++ Caption | モデル、画像 | 画像ごとに生成したプロンプトをSTRINGとして出力 |

## 動作環境

- BF16対応のNVIDIA CUDA GPU。12GB VRAMを目安にしてください。CPU・AMD・Apple GPUには対応していません。
- CUDA 13.0対応のNVIDIAドライバー。
- ComfyUI、Git。手動インストールでは[uv](https://docs.astral.sh/uv/getting-started/installation/)も必要です。
- モデル重み約6.5GBに加え、専用Python環境とインストール用キャッシュの空き容量。

推論用のPython 3.12環境を拡張内の`.venv`に作成します。ComfyUI側のPyTorch・Transformersは変更しません。

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

すでにモデルがある場合は、モデルリポジトリの全ファイルを`ComfyUI/models/hpsv3pp/<モデル名>/`に配置してください。重みだけでなく、`config.json`、`reward_config.json`、トークナイザー、プロセッサーのファイルも必要です。`extra_model_paths.yaml`の`hpsv3pp`で別のモデル置き場を指定できます。

ComfyUIを再起動し、Model Loaderでモデルを選びます。モデルフォルダを追加した場合は、ComfyUIのノード定義を更新してください。

## 使い方

[サンプルワークフロー](examples/caption_and_score.json)をComfyUIへ読み込み、Load Imageの画像とModel Loaderのモデルを選ぶと、CaptionからScoreへ接続した状態で試せます。

### 画像を評価する

1. Model LoaderをScoreの`model`へ接続します。
2. Load ImageなどのIMAGEをScoreの`images`へ接続し、`prompt`に評価対象のプロンプトを入力します。
3. `score_mode`を選び、実行します。

- `banner`: 画像上部に白い余白を追加し、スコアを表示します。元画像は覆いません。
- `metadata`: 画像に文字を描かず、PNGの`hpsv3pp`テキスト項目へJSON形式でモデル名・スコア・評価プロンプトを保存します。
- `both`: 画像上部にスコアを表示し、同じPNGのメタデータにもモデル名・スコア・評価プロンプトを保存します。

PNGはComfyUIのoutputフォルダへ保存されます。`filename_prefix`で保存先のサブフォルダと接頭辞を指定できます。通常のワークフローメタデータも保存します。`--disable-metadata`使用時は`banner`を選んでください。

標準のSave Imageと同じ日付・ノード値の置換を利用できます。例えば、`%date:yyyy%/%date:MM%/%date:dd%/HPSv3pp`は`output/2026/09/08/HPSv3pp_00001_.png`のように保存します。末尾の`HPSv3pp`はファイル名の接頭辞です。日付はブラウザーでキューに送信するときに展開されるため、APIから直接実行する場合は展開済みの文字列を渡してください。

メタデータはこのノードが保存したPNGに付属します。出力のIMAGEを別のSave Imageへ接続しても、このスコアメタデータは引き継がれません。

複数画像には同じプロンプトを適用できます。プロンプトがリストの場合は画像と同じ件数を渡してください。IMAGE・FLOATは画像ごとのリストとして、入力順に出力します。

### 画像からプロンプトを生成する

Model Loaderと画像をCaptionへ接続します。`max_new_tokens`で生成の長さの上限を指定できます。元の生成プロンプトを復元する機能ではなく、画像をもとに短い説明を生成します。内容を確認してから利用してください。

Captionの出力をScoreの`prompt`へ接続すれば、生成した説明を使って評価できます。複数画像の場合は両ノードへ同じ画像を同じ順序で渡してください。

## 評価とメモリ

参照実装の前処理・報酬トークン・NF4設定をそのまま利用し、報酬ヘッドの`mu`をスコアとして返します。スコアは確率や百分率ではありません。`iter_step`は通常の評価用の`0.0`です。

公開NF4モデルでは報酬ヘッドなどの一部がBF16で読み込まれ、上流からFP32との差に関する警告が出ます。本拡張はこの精度設定を変更しません。NF4モデルとフル精度モデルのスコアが一致する保証はありません。

HPSv3++は同時に評価する画像の組み合わせが結果へ影響するため、この拡張では常に1枚ずつ評価します。入力バッチを変えても各画像の評価条件を揃えられます。

処理前にComfyUI管理下のモデルをGPUから退避させ、専用プロセスで推論します。処理が終わるかキャンセルされると専用プロセスが終了し、モデルのVRAMを解放します。処理ノードごとにモデルを読み直すため、CaptionからScoreへ接続する場合は2回の読み込みが発生します。

画像・プロンプトはローカルの一時フォルダを介して推論プロセスへ渡し、完了・失敗・キャンセル時に削除します。推論はオフラインで実行します。

## ライセンス

拡張自身のコードは[MIT](LICENSE)です。推論には私が作成した[hpsv3-4bit](https://github.com/Stella2211/hpsv3-4bit)を使用します。HPSv3++の実装をコピーしているわけではなく、hpsv3-4bitを経由したGitサブモジュールとして参照します。上流実装とモデル重みは別の条件が適用されるため、利用・再配布前に[第三者ライセンス情報](THIRD_PARTY_NOTICES.md)と各参照先を確認してください。

固定コミット時点のHPSv3++上流実装にはLICENSEがなく、利用・改変・再配布の許諾範囲は確認できていません。サブモジュール参照や上流からの取得によって許諾が補われるわけではありません。依存先を含む構成全体がMITであるとは扱わないでください。
