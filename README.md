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

拡張のインストールとモデルの配置が必要です。ComfyUI-Managerを利用する場合も、モデル重みは別途ダウンロードしてください。

### ComfyUI-Managerからインストールする

ComfyUI-Managerが使える状態で始めます。Managerが表示されない場合は、利用しているComfyUIの配布形態に合わせて[公式のManager導入手順](https://docs.comfy.org/manager/install)を確認してください。

1. ComfyUIで**Manager**を開きます。新UIでは検索種別を**Node Pack**にします。旧UIでは**Install Nodes**からカスタムノードの一覧を開きます。表示名はManagerのバージョンや表示言語によって異なります。
2. `ComfyUI-HPSv3`を検索します。見つからない場合は`hpsv3`でも検索し、「インストール済み」などの絞り込みを外してください。
3. 対象の詳細を開き、リポジトリが[`Stella2211/ComfyUI-HPSv3`](https://github.com/Stella2211/ComfyUI-HPSv3)であることを確認します。Registry上のパッケージIDは`comfyui-hpsv3`、Publisher IDは`stella`です。
4. **Install**を押します。バージョンを選択する場合は、Registryで公開されている番号付きのバージョンを選んでください。新UIでは詳細の**Version**から選べます。
5. インストール完了まで待ちます。初回は専用Python環境とCUDA対応PyTorchなどを取得するため、時間とディスク容量が必要です。エラーが出た場合はComfyUIのターミナル／ログを確認してください。
6. Managerの案内に従ってComfyUIを再起動し、必要に応じてブラウザーを再読み込みします。その後、下の「モデルをダウンロードする」へ進みます。

セットアップでは拡張内の`.venv`に推論用環境を作成します。ComfyUI本体のPyTorch・Transformersは変更しません。モデル重みはこの段階では取得しません。

UIの詳細は[新UIの操作ガイド](https://docs.comfy.org/manager/pack-management)または[旧UIの操作ガイド](https://docs.comfy.org/manager/legacy-ui)を参照してください。新UIの検索・インストールはRegistry経由です。[Registryページ](https://registry.comfy.org/nodes/comfyui-hpsv3)で利用可能なバージョンがあるか確認し、検索できない場合はComfyUI・Managerを更新して再起動してください。Registryで公開処理中のバージョンは、インストール候補に現れるまで待つ必要があります。

### Gitで手動インストールする

Managerを使わない場合は、[uv](https://docs.astral.sh/uv/getting-started/installation/)をインストールし、`ComfyUI/custom_nodes`で以下を実行してください。Managerですでにインストール済みなら、同じ拡張を別フォルダへ重複してcloneする必要はありません。

```sh
git clone https://github.com/Stella2211/ComfyUI-HPSv3.git
cd ComfyUI-HPSv3
uv run --no-project python install.py
```

### モデルをダウンロードする（共通）

1. 以下のコマンドを使う場合は、ターミナルで`uv --version`を実行できることを確認します。見つからない場合は[uvの導入手順](https://docs.astral.sh/uv/getting-started/installation/)に従い、ターミナルを開き直してください。
2. インストールされた拡張フォルダへ移動します。`ComfyUI/custom_nodes`内にある、`install.py`と`.venv`を含むフォルダです。Manager経由では`comfyui-hpsv3`、上のGit手順では`ComfyUI-HPSv3`など、導入方法によって名前が異なる場合があります。
3. そのフォルダで以下を実行し、モデルリポジトリの全ファイルをダウンロードします。このコマンドはHugging Faceへアクセスします。

```sh
uv run --no-project --python .venv hf download stella221125/HPSv3-PlusPlus-bnb-NF4 --local-dir ../../models/hpsv3pp/HPSv3-PlusPlus-bnb-NF4
```

配置先は`ComfyUI/models/hpsv3pp/HPSv3-PlusPlus-bnb-NF4/`です。すでに[モデル一式](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4/tree/main)が手元にあれば、このフォルダへ配置しても構いません。重みだけでなく、`config.json`、`reward_config.json`、トークナイザー、プロセッサーなども必要です。

### インストールを確認する

1. モデルの配置後にComfyUIを再起動します。
2. ノード検索で`HPSv3++`を検索し、**Model Loader**・**Score**・**Caption**の3つのノードが表示されることを確認します。
3. [サンプルワークフロー](examples/caption_and_score.json)を読み込みます。**Load Image**で手元の画像を選び、**Model Loader**で`HPSv3-PlusPlus-bnb-NF4`を選択します。サンプル用画像は同梱していません。
4. ワークフローを実行し、Scoreのスコア付き画像がプレビュー・保存されることを確認します。Captionが生成した説明はScoreの`prompt`へ渡されます。初回のモデル読み込みには時間がかかります。

| 症状 | 確認すること |
| --- | --- |
| 再起動後もノードがない／読み込みエラーが出る | Managerで拡張が有効か確認し、ComfyUIの起動ログでこの拡張のエラーを確認します。 |
| `HPSv3++ runtime is missing`と表示される | 専用環境のセットアップが完了していません。インストールログを確認し、Git・uvが使えるターミナルで拡張フォルダから`uv run --no-project python install.py`を実行した後、ComfyUIを再起動します。 |
| Model Loaderの選択肢が空／モデルが不完全と表示される | 上記の配置先とモデル一式を確認します。フォルダが二重になっていないか、`config.json`と`reward_config.json`が重みと同じモデルフォルダにあるか確認し、ComfyUIを再起動します。 |
| CUDA関連エラー／GPUメモリ不足が出る | 「動作環境」のGPU・ドライバー条件を確認し、他のGPU使用アプリを終了して再試行します。 |

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
