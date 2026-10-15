# ComfyUI-HPSv3

[English](README.md) | **日本語**

HPSv3・HPSv3++のNF4モデルを使い、画像とプロンプトの評価・画像からのプロンプト生成をローカルで行うComfyUI拡張です。

| ノード | 入力 | 出力 |
| --- | --- | --- |
| HPSv3++ Model Loader / HPSv3 Model Loader | モデルフォルダ | ScoreとCaptionで使うモデル設定。標準モデルが未配置なら取得します。 |
| HPSv3++ Score / HPSv3 Score | モデル、画像、プロンプト | スコア付きPNGを保存し、IMAGEとFLOATのリストを出力 |
| HPSv3++ Caption / HPSv3 Caption | モデル、画像 | 画像ごとに生成したプロンプトのSTRINGリスト |

HPSv3とHPSv3++のモデルに互換性はありません。Model Loaderは同じ系列のScore・Captionへ接続してください。

## 動作環境

- BF16対応のNVIDIA CUDA GPUと、ComfyUIのPyTorchに対応するドライバー。VRAMは12GBが目安です（8GBでも動く可能性はありますが未検証）。CPU・AMD・Apple GPUには対応していません。
- Python 3.12以降、Transformers 5.17.x。依存パッケージはComfyUIの環境へ追加され、PyTorch・torchvisionは置き換えません。
- モデル用の空き容量：HPSv3は約5.9GB、HPSv3++は約6.5GB。

## インストール

### ComfyUI-Manager

1. **Manager**で`ComfyUI-HPSv3`を検索し、リポジトリが[`Stella2211/ComfyUI-HPSv3`](https://github.com/Stella2211/ComfyUI-HPSv3)であることを確認します。
2. **Install**を押します（バージョンを聞かれたら番号付きのものを選択）。初回は依存パッケージの追加に時間がかかります。エラーが出た場合はComfyUIのターミナルを確認してください。
3. ComfyUIを再起動し、ブラウザーを再読み込みします。

見つからない場合はComfyUI・Managerを更新するか、[Registryページ](https://registry.comfy.org/nodes/comfyui-hpsv3)を確認してください。既存のインストールで依存パッケージが不足している場合は、Managerで修復して再起動します。

### Git

`ComfyUI/custom_nodes`で次を実行します。最後のコマンドはHPSv3++の互換ソースを取得するもので、Managerで導入した場合は不要です。

```sh
git clone https://github.com/Stella2211/ComfyUI-HPSv3.git
cd ComfyUI-HPSv3
uv pip install --python <ComfyUIのPython> -r requirements.txt
uv run --no-project --python <ComfyUIのPython> python install.py
```

## モデル

Model Loaderで標準モデルを選んで実行します。未配置の場合は初回実行時にHugging Faceから取得し、進捗はComfyUIのターミナルに表示されます。以降のScore・Captionはローカルファイルのみを使い、オフラインで動作します。

| 系列 | モデル名 | 取得元 | 配置先 |
| --- | --- | --- | --- |
| HPSv3 | `HPSv3-bnb-NF4` | [`stella221125/HPSv3-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-bnb-NF4) | `ComfyUI/models/hpsv3/` |
| HPSv3++ | `HPSv3-PlusPlus-bnb-NF4` | [`stella221125/HPSv3-PlusPlus-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4) | `ComfyUI/models/hpsv3pp/` |

- 統合済みのNF4モデルです。ベースモデルや元のチェックポイントは不要です。
- 自動取得の対象は標準モデルのみです。取得をキャンセルした場合や失敗した場合は、ワークフローを再実行すると再開できます。取得と検証が終わるまでモデルは使用されません。
- 取得済みのモデルは自動更新されません。新しい版を取得するには、既存のモデルフォルダを別の場所へ移して再実行してください。
- オフライン環境では、Hugging Faceからモデルフォルダ一式を取得して上記の場所へ配置してください。重みだけでは動作せず、`config.json`、`reward_config.json`、トークナイザーとプロセッサーのファイル、すべてのsafetensorsが必要です。
- モデルや依存パッケージの更新でスコアが変わる場合があるため、比較する際は同じ環境を使ってください。

## 使い方

サンプルワークフローを読み込み、Load Imageで画像を選んで実行します。サンプル用の画像は同梱していません。

| サンプル | 内容 |
| --- | --- |
| [caption_and_score.json](examples/caption_and_score.json) | HPSv3++：Captionが生成したプロンプトをScoreへ渡す |
| [hpsv3_caption_and_score.json](examples/hpsv3_caption_and_score.json) | HPSv3：同じワークフロー |
| [hpsv3_score.json](examples/hpsv3_score.json) | HPSv3：入力したプロンプトでScoreを実行 |

![HPSv3++ Caption → Score workflow](images/hpsv3pp_caption_and_score.png)

![HPSv3 Caption → Score workflow](images/hpsv3_caption_and_score.png)

生成されたキャプションは**Generated Caption**ノード（Preview as Text）に表示され、スコア付き画像はScoreがプレビュー・保存します。

### Score

Model Loaderを`model`、IMAGEを`images`へ接続し、`prompt`に評価対象のプロンプトを入力します。`score_mode`は次から選びます。

- `banner`: 画像の上に白い余白を追加し、スコアを表示します。元画像は覆いません。
- `metadata`: 画像には何も描かず、PNGのテキスト項目`hpsv3pp`（HPSv3は`hpsv3`）にモデル名・スコア・プロンプトをJSONで保存します。IMAGE出力は入力画像そのままです。
- `both`: 両方を行います。

保存先はComfyUIのoutputフォルダ内のPNGです。`filename_prefix`では、Save Imageと同じ日付・ノード値の置換が使えます（例：`%date:yyyy%/%date:MM%/%date:dd%/HPSv3pp`）。置換はブラウザー側で行うため、API経由の実行では動作しません。スコアのメタデータはこのノードが保存したPNGに付くもので、IMAGE出力を別のSave Imageへ接続しても引き継がれません。

プロンプトが1つなら全画像に適用され、複数の場合は画像と同じ件数が必要です。画像は1枚ずつ個別に評価され、IMAGEとFLOATは入力順の画像ごとのリストとして出力されます。

### Caption

Model Loaderと画像を接続します。`max_new_tokens`（16〜512、既定96）で生成の長さの上限を指定します。画像をもとに短い説明を生成するもので、元の生成プロンプトを復元するものではないため、内容を確認してから利用してください。画像自身の説明で評価するには、Captionの出力をScoreの`prompt`へ接続し、両ノードへ同じ画像を渡します。

## トラブルシューティング

| 症状 | 確認すること |
| --- | --- |
| 再起動後もノードがない／読み込みエラーが出る | Managerで拡張が有効か確認し、ComfyUIの起動ログを確認します。 |
| `transformers`や`bitsandbytes`が見つからない | Managerで依存関係を修復または再インストールして再起動します。PyTorch・torchvisionは置き換えないでください。 |
| モデルの自動取得が失敗する | インターネット接続、Hugging Faceへのアクセス、空き容量、書き込み権限を確認して再実行します。 |
| モデルが不完全または不正と表示される | [モデル](#モデル)に挙げたファイルを確認します。既存フォルダは上書きされないため、不足分を補うか、フォルダを別の場所へ移して再取得してください。 |
| "requires an NVIDIA CUDA GPU"、またはCUDA・メモリ不足のエラー | 動作環境を確認し、GPUを使う他のアプリを終了して再試行します。 |
| "Metadata is disabled" | ComfyUIが`--disable-metadata`で起動しています。`banner`を使うか、オプションを外してください。 |
| "got N images but M prompts" | プロンプトを1つにするか、画像と同じ件数を渡してください。 |

## ライセンス

拡張本体はMITライセンスです。モデル重みには別の条件があるため、各モデルリポジトリのライセンスに従ってください。第三者のコードとモデルに関する表記は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
