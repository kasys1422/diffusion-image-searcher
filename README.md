[日本語/[English](https://github.com/kasys1422/diffusion-image-searcher/blob/main/README_EN.md)]
# diffusion-image-searcher

diffusion-image-searcher は、入力したテキストに関連した画像をローカルコンピュータ内から検索するソフトウェアです。また、画像を元に類似画像を検索することも可能です。

## インストール方法

### windows exe version (推奨)

1. [Release](https://github.com/kasys1422/diffusion-image-searcher/releases)内から最新版のファイルをダウンロードして任意の場所に解凍してください。
2. (オプション) 追加のモデルをRelease内からダウンロードし、解凍したファイルを /res/model の中に移動してください。
3. 解答したフォルダ内にある実行ファイルをダブルクリックして実行してください。

※アンチウイルスソフトウェアが誤検知する場合があるので、例外指定等の対応を適宜行ってください。

### python version

（注）Windowsで動作確認を行っているため、それ以外のOSでの実行する場合エラーが発生する可能性があります。

1. python3.9をインストールします。
2. 任意の場所にフォルダ作成し、そこにリポジトリをクローンします。
3. [Release](https://github.com/kasys1422/diffusion-image-searcher/releases)内から最新版のファイルをダウンロードして /res/model から学習済みモデルを適切な階層にコピーします。
4. pythonの仮想環境を作成し、有効化します。
5. requirements.txtを元にモジュールをインストールします。
6. diffusion_image_searcher.pyを実行します。

## 使用方法

### ・テキストから画像を検索

1. ソフトウェアを起動します。
2. (オプション、追加のモデルをダウンロード済みかつ12GB以上の空きメモリがある場合) 設定の「画像生成モデル」に「stable-diffusion-v1-4-openvino-fp16-CPU」を指定します。
3. 「テキストから検索」タブを選択します。
4. 「検索を行うフォルダ」を指定します。
5. 検索したい画像の特徴を英語で「検索するテキスト(英語)」内に英語で記述します。
6. 「検索」ボタンをクリックします。
7. 検索が開始されるのでしばらく待ちます。(検索枚数やディスクのアクセス速度によっては10分以上かかる場合があります)
8. 結果が一覧表示されます。
9. しきい値を適切な値に指定して再検索するとより適切な結果を得られます。

### ・画像から類似画像を検索

1. ソフトウェアを起動します。
2. 「テキストから検索」タブを選択します。
3. 「検索を行うフォルダ」を指定します
4. 検索したい画像のに近い画像を「検索するファイル」から指定します。
5. 「検索」ボタンをクリックします。
6. 検索が開始されるのでしばらく待ちます。(検索枚数やディスクのアクセス速度によっては10分以上かかる場合があります)
7. 結果が一覧表示されます。
8. しきい値を適切な値に指定して再検索するとより適切な結果を得られます。

## 仕組み

入力されたテキストを元に、拡散モデルを用いて画像を生成して、その画像とローカルコンピュータ内の画像ファイルの類似度を算出することで検索を実現しています。

## 動作要件

<details>
  <summary>
    windows exe version
  </summary>
  <dl>
    <dt>OS</dt>
    <dd>Windows10 もしくは Windows11</dd>
    <dt>CPU</dt>
    <dd>AVX2命令もしくはSSE2命令に対応した4コア以上のx64 CPU（Intel製、2019年以降の製品を推奨） <br>※AVX命令もしくはSSE2命令に対応したCPU</dd>
    <dt>RAM</dt>
    <dd>16GB以上 ※12GB以上</dd>
    <dt>ROM</dt>
    <dd>10GB以上の空き容量</dd>
    <dt>ディスプレイ</dt>
    <dd>拡大率100％で解像度1280x720より広い表示領域</dd>
    ※最低動作要件
  </dl>
</details>

<details>
  <summary>
    python version
  </summary>
  <dl>
    <dt>Python Version</dt>
    <dd>python 3.9</dd>
    <dt>CPU</dt>
    <dd>AVX2命令もしくはSSE2命令に対応した4コア以上のx64 CPU（Intel製、2019年以降の製品を推奨） <br>※AVX命令もしくはSSE2命令に対応したCPU</dd>
    <dt>RAM</dt>
    <dd>16GB以上 ※12GB以上</dd>
    <dt>ROM</dt>
    <dd>10GB以上の空き容量</dd>
    <dt>ディスプレイ</dt>
    <dd>拡大率100％で解像度1280x720より広い表示領域</dd>
    ※最低動作要件
  </dl>
</details>
