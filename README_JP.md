# diffusion-image-searcher

diffusion-image-searcher は、入力したテキストに関連した画像をローカルコンピュータ内から検索するソフトウェアです。また、画像を元に類似画像を検索することも可能です。

## 使い方

### windows exe version

1. Release内から最新版のファイルをダウンロードして任意の場所に解凍してください。
2. 解答したフォルダ内にある実行ファイルをダブルクリックで実行してください。

※アンチウイルスソフトウェアが誤検知する場合があるので、例外指定等の対応を適宜行ってください。

### python version

（注）Windowsで動作確認を行っているため、それ以外のOSでの実行の場合エラーが発生する可能性があります。

1. python3.9をインストールします。
2. 任意の場所にフォルダ作成し、そこにリポジトリをクローンします。
3. Release内から最新版のファイルをダウンロードして /res/model から学習済みモデルを適切な階層にコピーします。
4. pythonの仮想環境を作成し、有効化します。
5. requirements.txtを元にモジュールをインストールします。
6. diffusion_image_searcher.pyを実行します。

## 仕組み

入力されたテキストを元に、拡散モデルを用いて画像を生成して、その画像とローカルコンピュータ内のファイルの類似度を算出することで検索を実現しています。

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
    ※最低動作要件
  </dl>
</details>
