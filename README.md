# パーティクル検知クエスト 2023 

このコンテストでは、半導体製造装置で使用する「異物」を画像認識で判定することを課題とし、「画像認識の技術」と「異物混入をどのようなアルゴリズムで検出するか？」を競います。

## Setup

本コンテストでは、Dockerを用いてローカル環境Jupyter Notebookを構築し、データの分析、可視化、精度確認をおこないます。

[Jupyter Notebook](https://jupyter.org/) は、ブラウザ上で利用可能なデータ分析のためのプログラミング実行環境です。

### Requirements for Cluster iMac

- [Docker Desktop for Mac](https://meta.intra.42.fr/articles/imac-docker)

### Requirements for PC

事前にDockerおよびDocker Composeをインストールしてください。Docker Desktopをインストールすると、Docker Composeも同時にインストールされます。

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

#### Windows

- [Docker Desktop for Windows](https://docs.docker.com/docker-for-windows/install/)

#### Mac

- [Docker Desktop for Mac](https://docs.docker.com/docker-for-mac/install/)

#### Linux

Docker Desktopを用いない場合、Docker EngineおよびDocker Composeをそれぞれインストールしてください。

- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

Linuxの場合、Docker公式のConvenience Scriptを用いてインストールすることもできます。

```bash
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sh get-docker.sh
```

### Prepare

こちらのリポジトリーを適切なディレクトリにコピーしてください。
※クラスターのiMacを活用している場合、個人の`/goinfre`ディレクトリにこのリポジトリーをコピーしてください。

Jupyter Notebookにアクセスするためのパスワードを設定します。
以下のコマンドで、`.env.sample`ファイルをコピーして`.env`ファイルを作成します。

```bash
$ cp work/.env.sample work/.env
```

`.env`ファイルをエディタで開き、`NOTEBOOK_PASSWORD`に任意のパスワードを記入します。

```bash
$ vi work/.env
```

work/.env
```
NOTEBOOK_PASSWORD=<Your Password for Jupyter Notebook>
```

### Unzip Data

下記コマンドで、`LSWMD_25519.pkl.zip` ファイルを解凍します。

```bash
$ unzip work/input/LSWMD_25519.pkl.zip -d work/input/

$ ls work/input/
 LSWMD_25519.pkl  LSWMD_25519.pkl.zip
```


## Start

docker-compose.ymlファイルのあるルートディレクトリで、下記コマンドを実行しDockerコンテナを起動します。

```bash
$ docker-compose up -d
```

### Jupyter Notebook

起動が完了したら、ブラウザで [http://localhost:8888/](http://localhost:8888/) にアクセスし、Jupyter Notebookを開きます。初回アクセス時は、上記手順で設定した`NOTEBOOK_PASSWORD`の入力を求められます。

[01_Tutorial](/work/01_Tutorial)フォルダには、セットアップしたデータに対して基本的な操作をおこなうためのチュートリアルが含まれています。

[02_Submission](/work/02_Submission)フォルダには、評価方法のサンプルを用意しています。

## Stop

作業を終了するときは、下記コマンドでDockerコンテナを停止してください。

```bash
$ docker-compose down
```
### Remove

コンテナの削除：

```bash
$ docker-compose rm
```

イメージの削除：

```bash
$ docker image rm particle-detection-quest-2023_jupyter
```
