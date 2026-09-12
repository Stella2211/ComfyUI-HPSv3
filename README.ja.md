# ComfyUI-HPSv3

[English](README.md) | **日本語**

HPSv3・HPSv3++のNF4モデルを使い、画像とプロンプトの評価・画像からのプロンプト生成をローカルで行うComfyUI拡張です。

| ノード | 入力 | 動作・出力 |
| --- | --- | --- |
| HPSv3++ Model Loader | モデルフォルダ | 標準モデルが未配置なら自動取得し、2つの処理ノードで使うモデル設定を出力 |
| HPSv3++ Score | モデル、画像、プロンプト | スコア付きPNGを保存・プレビューし、IMAGEとFLOATを出力 |
| HPSv3++ Caption | モデル、画像 | 画像ごとに生成したプロンプトをSTRINGとして出力 |
| HPSv3 Model Loader | モデルフォルダ | HPSv3の標準NF4モデルが未配置なら自動取得し、モデル設定を出力 |
| HPSv3 Score | HPSv3モデル、画像、プロンプト | スコア付きPNGを保存・プレビューし、IMAGEとFLOATを出力 |
| HPSv3 Caption | HPSv3モデル、画像 | 画像ごとに生成したプロンプトをSTRINGとして出力 |

HPSv3とHPSv3++のモデルには互換性がありません。同じ系列のModel LoaderをScore・Captionへ接続してください。既存のHPSv3++ワークフローはそのまま使えます。

## 動作環境

- BF16対応のNVIDIA CUDA GPU。12GB VRAMを目安にしてください。( 8GB でも動作する可能性はありますが未検証です。)
- CPU・AMD・Apple GPUには対応していません。
- Python 3.12以降、Transformers 5.17.x。必要な依存パッケージはManagerが自動で導入します。
- ComfyUIのPyTorch・CUDAに対応するNVIDIAドライバー。実機検証環境はCUDA 13.0です。
- モデル重みはHPSv3が約5.9GB、HPSv3++が約6.5GB。ホスト環境の依存関係とモデル用の空き容量が必要です。PyTorch・torchvisionはComfyUIのものを使用し、置き換えません。

## インストール

拡張をインストールすると、標準モデルはワークフローの初回実行時にModel Loaderが自動でダウンロードします。モデル取得のために手動でコマンドを実行する必要はありません。

### ComfyUI-Managerからインストールする

ComfyUI-Managerが使える状態で始めます。Managerが表示されない場合は、利用しているComfyUIの配布形態に合わせて[公式のManager導入手順](https://docs.comfy.org/manager/install)を確認してください。

1. ComfyUIで**Manager**を開きます。新UIでは検索種別を**Node Pack**にします。旧UIでは**Install Nodes**からカスタムノードの一覧を開きます。表示名はManagerのバージョンや表示言語によって異なります。
2. `ComfyUI-HPSv3`を検索します。見つからない場合は`hpsv3`でも検索し、「インストール済み」などの絞り込みを外してください。
3. 対象の詳細を開き、リポジトリが[`Stella2211/ComfyUI-HPSv3`](https://github.com/Stella2211/ComfyUI-HPSv3)であることを確認します。
4. **Install**を押します。バージョンを選択する場合は、Registryで公開されている番号付きのバージョンを選んでください。新UIでは詳細の**Version**から選べます。
5. インストール完了まで待ちます。初回はホスト環境へ依存パッケージを追加するため、時間が必要です。エラーが出た場合はComfyUIのターミナル／ログを確認してください。
6. Managerの案内に従ってComfyUIを再起動します。画面が更新されない場合はブラウザーを再読み込みします。その後、下の「インストールを確認する」へ進みます。

必要なファイルはManagerが自動で取得します。インストール時にはGitHubへ、初回のモデル取得時にはHugging Faceへ接続します。追加のコマンド実行は不要です。インストールに失敗した場合は、接続を確認してManagerから再インストールしてください。

比較用途では同じモデルと環境を使ってください。モデルや依存パッケージの更新でスコアが変わる場合があります。

既存インストールへ追加する場合は、Managerで依存関係を修復してからComfyUIを再起動してください。

UIの詳細は[新UIの操作ガイド](https://docs.comfy.org/manager/pack-management)または[旧UIの操作ガイド](https://docs.comfy.org/manager/legacy-ui)を参照してください。新UIの検索・インストールはRegistry経由です。[Registryページ](https://registry.comfy.org/nodes/comfyui-hpsv3)で利用可能なバージョンがあるか確認し、検索できない場合はComfyUI・Managerを更新して再起動してください。Registryで公開処理中のバージョンは、インストール候補に現れるまで待つ必要があります。

### Gitで手動インストールする

Managerを使わない場合は、`ComfyUI/custom_nodes`で以下を実行して拡張を配置し、ComfyUIのPython環境へ`requirements.txt`の依存パッケージを導入してください。Managerですでにインストール済みなら、同じ拡張を別フォルダへ重複してcloneする必要はありません。

```sh
git clone https://github.com/Stella2211/ComfyUI-HPSv3.git
cd ComfyUI-HPSv3
uv pip install --python <ComfyUIのPython> -r requirements.txt
```

依存関係の導入後、次のコマンドでセットアップを完了します。Managerから導入した場合、この操作は不要です。

```sh
uv run --no-project --python <ComfyUIのPython> python install.py
```

### HPSv3のモデル取得と実行

**HPSv3 Model Loader**で`HPSv3-bnb-NF4`を選び、**HPSv3 Score**または**HPSv3 Caption**へ接続します。[HPSv3の評価サンプル](examples/hpsv3_score.json)も利用できます。Load Imageで画像を選び、Scoreの`prompt`を入力して実行してください。

未配置なら[`stella221125/HPSv3-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-bnb-NF4)を`ComfyUI/models/hpsv3/HPSv3-bnb-NF4/`へ自動取得します。統合済みNF4モデルを使うため、ベースモデルや元の報酬チェックポイントを別途配置・変換する必要はありません。

手動配置する場合は、Hugging Faceのモデルページからモデル一式をComfyUI/models/hpsv3/HPSv3-bnb-NF4/へ配置してください。

重みだけでなく、モデルのconfig・reward_config・トークナイザー・プロセッサーを含む一式が必要です。推論はローカルファイルだけを使用します。

画像から生成した説明で評価する場合は、[HPSv3のCaption→Scoreサンプル](examples/hpsv3_caption_and_score.json)を読み込んでください。同じ画像をCaption・Scoreへ渡し、Captionの出力をScoreの`prompt`へ接続しています。Load Imageで画像を選ぶだけで実行できます。

### HPSv3++のモデルの自動ダウンロード

Model Loaderで`HPSv3-PlusPlus-bnb-NF4`を選び、ScoreまたはCaptionにつないだワークフローを実行してください。モデルが未配置でも選択肢に表示されます。初回実行時にHugging Faceの[`stella221125/HPSv3-PlusPlus-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4)からモデル一式を取得します。

標準の保存先は`ComfyUI/models/hpsv3pp/HPSv3-PlusPlus-bnb-NF4/`です。初回はインターネット接続と約6.5GB以上の空き容量が必要で、進捗はComfyUIのターミナル／ログに表示されます。取得済みのモデルは再利用し、Score・Captionの推論は引き続きオフラインで行います。

新規ダウンロード時は公開されている最新のモデルを取得します。取得済みモデルの自動更新は行わないため、更新版を取得したい場合は既存モデルフォルダを別の場所へ移してから再実行してください。

取得中にキャンセルした場合や通信に失敗した場合は、接続と空き容量を確認してワークフローを再実行してください。途中のデータは再試行用に保持し、ダウンロードと検証が完了するまではモデルとして使用しません。自動取得の対象は標準モデルのみです。

### 手動でモデルを配置する（任意）

オフライン環境では、Hugging Faceからモデル一式を取得して対応する`ComfyUI/models/hpsv3/`または`ComfyUI/models/hpsv3pp/`へ配置してください。`config.json`、`reward_config.json`、トークナイザー、プロセッサー、全safetensors shardが必要です。

### インストールを確認する

1. 拡張のインストール後にComfyUIを再起動します。
2. ノード検索で`HPSv3++`を検索し、**Model Loader**・**Score**・**Caption**の3つのノードが表示されることを確認します。
3. [サンプルワークフロー](examples/caption_and_score.json)を読み込みます。**Load Image**で手元の画像を選び、**Model Loader**で`HPSv3-PlusPlus-bnb-NF4`を選択します。サンプル用画像は同梱していません。
4. ワークフローを実行し、Scoreのスコア付き画像がプレビュー・保存されることを確認します。Captionが生成した説明はScoreの`prompt`へ渡されます。未配置の場合は自動ダウンロードが先に行われるため、初回の実行には時間がかかります。

| 症状 | 確認すること |
| --- | --- |
| 再起動後もノードがない／読み込みエラーが出る | Managerで拡張が無効化されていないか確認し、ComfyUIの起動ログでこの拡張のエラーを確認します。 |
| `transformers`や`bitsandbytes`が見つからない | Managerでこの拡張の依存関係を修復または再インストールし、ComfyUIを再起動します。PyTorch・torchvisionは置き換えないでください。 |
| モデルの自動ダウンロードが失敗する | インターネット接続、Hugging Faceへのアクセス、保存先の空き容量と書き込み権限を確認し、ワークフローを再実行します。詳細はComfyUIのターミナル／ログで確認してください。 |
| 手動配置したモデルが不完全と表示される | 上記の配置先とモデル一式を確認します。既存の不完全なモデルは自動で上書きしません。必要なファイルを補うか、既存フォルダを別の場所へ移してから標準モデルの自動取得を試してください。 |
| CUDA関連エラー／GPUメモリ不足が出る | 「動作環境」のGPU・ドライバー条件を確認し、他のGPU使用アプリを終了して再試行します。 |

## 使い方

[サンプルワークフロー](examples/caption_and_score.json)をComfyUIへ読み込み、Load Imageの画像とModel Loaderのモデルを選ぶと、CaptionからScoreへ接続した状態で試せます。

### 画像を評価する

1. Model LoaderをScoreの`model`へ接続します。
2. Load ImageなどのIMAGEをScoreの`images`へ接続し、`prompt`に評価対象のプロンプトを入力します。
3. `score_mode`を選び、実行します。

- `banner`: 画像上部に白い余白を追加し、スコアを表示します。元画像は覆いません。
- `metadata`: 画像に文字を描かず、PNGの`hpsv3pp`（HPSv3の場合は`hpsv3`）テキスト項目へJSON形式でモデル名・スコア・評価プロンプトを保存します。
- `both`: 画像上部にスコアを表示し、同じPNGのメタデータにもモデル名・スコア・評価プロンプトを保存します。

標準のSave Imageと同じ日付・ノード値の置換を利用できます。例えば、`%date:yyyy%/%date:MM%/%date:dd%/HPSv3pp`は`output/2026/09/08/HPSv3pp_00001_.png`のように保存できます。

メタデータはこのノードが保存したPNGに付属します。出力のIMAGEを別のSave Imageへ接続しても、このスコアメタデータは引き継がれません。

複数画像には同じプロンプトを適用できます。プロンプトがリストの場合は画像と同じ件数を渡してください。IMAGE・FLOATは画像ごとのリストとして、入力順に出力します。

### 画像からプロンプトを生成する

Model Loaderと画像をCaptionへ接続します。`max_new_tokens`で生成の長さの上限を指定できます。元の生成プロンプトを復元する機能ではなく、画像をもとに短い説明を生成します。内容を確認してから利用してください。

Captionの出力をScoreの`prompt`へ接続すれば、生成した説明を使って評価できます。複数画像の場合は両ノードへ同じ画像を同じ順序で渡してください。

HPSv3・HPSv3++のCaption→Scoreサンプルには、生成した説明を表示する**Generated Caption**ノード（ComfyUI標準のPreview as Text）も接続しています。実行後、このノードでキャプションを確認できます。
