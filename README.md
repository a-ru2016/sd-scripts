* 使った論文/内の学習手法を実装
# 実装済み
- universal lora
- universal lalora
- universal null lalora
- universal null lalora plus
- universal null lalora plus effi abm

This repository contains training, generation and utility scripts for Stable Diffusion.
# Null-Space EffiLoRA with ABM & LaLoRA Training Scripts

このリポジトリは、Stable Diffusion (SD1.5 / SDXL) のための高度な実験的学習スクリプトセットです。
[Kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) をベースに、以下の最新・実験的な手法を組み合わせています。

## 主な機能 (Features)

1.  **Null-Space Decomposition (Null-LoRA)**
    * 事前学習済みモデルの重みをSVD（特異値分解）し、"Null Space"（モデルが学習していない、あるいは重要度が低い次元）を利用して学習を行います。
    * 既存の知識を破壊せずに新しい概念を注入することを目的としています。

2.  **EffiLoRA (MoE: Mixture of Experts)**
    * LoRAのUp層（B行列）を複数のExpertに分割し、Router (Gate) によって動的に選択・合成します。
    * Down層（A行列）は共有することでパラメータ効率を維持しつつ、表現力を向上させます。

3.  **ABM (Adaptive Boundary Margin) Initialization**
    * 学習の初期段階で、モデルの出力決定境界にマージン（Hinge Loss）を設ける初期化ステージを実行します。
    * これにより、概念の分離性を高めることを狙っています。

4.  **LaLoRA (Latent LoRA) Regularization**
    * データセットの曲率（Curvature）を事前に計算し、正則化項として利用することで過学習を防ぎます。

5.  **DoRA / LoRA+ Support**
    * **DoRA:** 重みの大きさ(Magnitude)と方向(Direction)を分離して学習。
    * **LoRA+:** U層とD層で異なる学習率を適用し、学習効率を向上。

## ファイル構成

* `sdxl_train_network_ABM.py`: SDXL用 ABM対応学習スクリプト
* `train_network_ABM.py`: SD1.5用 ABM対応学習スクリプト
* `network_null_lalora_plus_Effi_ABM.py`: 上記すべての機能を搭載したネットワークモジュール（推奨）
* `calc_lalora_context.py`: LaLoRA用のコンテキスト（正則化パラメータ）を計算するツール
* その他 `network_*.py`: 各機能のサブセット版モジュール

## 必要要件

* Python 3.10+
* PyTorch 2.0+
* `sd-scripts` の依存関係 (diffusers, accelerate, transformers, etc.)
* `bitsandbytes` (推奨)

## 使用方法 (Usage)

### 1. 準備

本スクリプトは `sd-scripts` の環境内で動作することを想定しています。
また、**Universal Basis (基底ファイル)** が別途必要になります（事前にSVD等で抽出された `.safetensors` ファイル）。

### 2. LaLoRAコンテキストの計算 (オプション)

LaLoRA正則化を使用する場合、事前にデータセットからコンテキストを計算します。

```bash
python calc_lalora_context.py \
    --pretrained_model_name_or_path "model_path.safetensors" \
    --basis_path "basis_file.safetensors" \
    --source_data_dir "/path/to/training_images" \
    --output_path "lalora_context.safetensors" \
    --resolution 1024 \
    --batch_size 4 \
    --mixed_precision "fp16"

提示されたコード群は、Kohya-ss (sd-scripts) をベースに、非常に高度で実験的な学習手法（Null-Space, EffiLoRA/MoE, ABM, LaLoRA, DoRA）を統合した拡張スクリプトセットのようです。

これらの機能を包括的に説明し、使用方法を記述した README.md を作成しました。

Markdown

# Null-Space EffiLoRA with ABM & LaLoRA Training Scripts

このリポジトリは、Stable Diffusion (SD1.5 / SDXL) のための高度な実験的学習スクリプトセットです。
[Kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) をベースに、以下の最新・実験的な手法を組み合わせています。

## 主な機能 (Features)

1.  **Null-Space Decomposition (Null-LoRA)**
    * 事前学習済みモデルの重みをSVD（特異値分解）し、"Null Space"（モデルが学習していない、あるいは重要度が低い次元）を利用して学習を行います。
    * 既存の知識を破壊せずに新しい概念を注入することを目的としています。

2.  **EffiLoRA (MoE: Mixture of Experts)**
    * LoRAのUp層（B行列）を複数のExpertに分割し、Router (Gate) によって動的に選択・合成します。
    * Down層（A行列）は共有することでパラメータ効率を維持しつつ、表現力を向上させます。

3.  **ABM (Adaptive Boundary Margin) Initialization**
    * 学習の初期段階で、モデルの出力決定境界にマージン（Hinge Loss）を設ける初期化ステージを実行します。
    * これにより、概念の分離性を高めることを狙っています。

4.  **LaLoRA (Latent LoRA) Regularization**
    * データセットの曲率（Curvature）を事前に計算し、正則化項として利用することで過学習を防ぎます。

5.  **DoRA / LoRA+ Support**
    * **DoRA:** 重みの大きさ(Magnitude)と方向(Direction)を分離して学習。
    * **LoRA+:** U層とD層で異なる学習率を適用し、学習効率を向上。

## ファイル構成

* `sdxl_train_network_ABM.py`: SDXL用 ABM対応学習スクリプト
* `train_network_ABM.py`: SD1.5用 ABM対応学習スクリプト
* `network_null_lalora_plus_Effi_ABM.py`: 上記すべての機能を搭載したネットワークモジュール（推奨）
* `calc_lalora_context.py`: LaLoRA用のコンテキスト（正則化パラメータ）を計算するツール
* その他 `network_*.py`: 各機能のサブセット版モジュール

## 必要要件

* Python 3.10+
* PyTorch 2.0+
* `sd-scripts` の依存関係 (diffusers, accelerate, transformers, etc.)
* `bitsandbytes` (推奨)

## 使用方法 (Usage)

### 1. 準備

本スクリプトは `sd-scripts` の環境内で動作することを想定しています。
また、**Universal Basis (基底ファイル)** が別途必要になります（事前にSVD等で抽出された `.safetensors` ファイル）。

### 2. LaLoRAコンテキストの計算 (オプション)

LaLoRA正則化を使用する場合、事前にデータセットからコンテキストを計算します。

```bash
python calc_lalora_context.py \
    --pretrained_model_name_or_path "model_path.safetensors" \
    --basis_path "basis_file.safetensors" \
    --source_data_dir "/path/to/training_images" \
    --output_path "lalora_context.safetensors" \
    --resolution 1024 \
    --batch_size 4 \
    --mixed_precision "fp16"
```

3. 学習の実行 (Training)
train_network_ABM.py (または sdxl_train_network_ABM.py) を使用し、ネットワークモジュールとして network_null_lalora_plus_Effi_ABM.py を指定します。

accelerate launch --num_cpu_threads_per_process 2 sdxl_train_network_ABM.py \
    --pretrained_model_name_or_path "/path/to/sdxl_model.safetensors" \
    --train_data_dir "/path/to/train_data" \
    --output_dir "./outputs" \
    --output_name "my_effilora_model" \
    --network_module "network_null_lalora_plus_Effi_ABM" \
    --network_dim 32 \
    --network_alpha 16 \
    --network_args \
        "basis_path=/path/to/basis.safetensors" \
        "lalora_context_path=/path/to/lalora_context.safetensors" \
        "lalora_lambda=0.001" \
        "enable_effilora=True" \
        "num_experts=4" \
        "use_dora=True" \
        "loraplus_lr_ratio=16.0" \
    --abm_enabled \
    --abm_steps 500 \
    --abm_lr 1e-4 \
    --abm_margin 0.5 \
    --learning_rate 1e-4 \
    --text_encoder_lr 5e-5 \
    --unet_lr 1e-4 \
    --mixed_precision "fp16" \
    --save_model_as "safetensors" \
    --save_every_n_epochs 1 \
    --max_train_epochs 10

# 引数説明 (Arguments)
* ネットワーク引数 (--network_args)
引数名,デフォルト,説明
basis_path,必須,事前計算された基底ファイルのパス。
lalora_context_path,None,calc_lalora_context.py で生成した正則化用ファイル。
lalora_lambda,0.0,LaLoRA正則化の強さ。0より大きい場合有効化されます。
enable_effilora,False,EffiLoRA (MoE) を有効にするか。
num_experts,4,EffiLoRA有効時のExpert数。
use_dora,False,DoRA (Weight-Decomposed LoRA) を有効にするか。
loraplus_lr_ratio,16.0,LoRA+ のレート（Up層の学習率倍率）。
svd_cache_path,None,Null Space計算結果のキャッシュパス（次回以降の起動高速化）。
abm_batch_size,None,ABMステージ専用のバッチサイズ（VRAM節約用）。

* ABM (初期化ステージ) 引数
引数名,デフォルト,説明
--abm_enabled,False,ABM初期化ステージを有効にするフラグ。
--abm_steps,500,ABMを実行するステップ数。
--abm_lr,1e-4,ABMステージの学習率。
--abm_margin,0.5,Hinge Lossのマージン値。
--abm_weight_strategy,uniform,"重み付け戦略 (uniform, linear, quadratic)。"

注意事項
Basis Path: network_null_*.py 系は basis_path が必須です。これはモデルの重み空間の基底を定義する重要なファイルです。

VRAM使用量: EffiLoRAやABMは通常のLoRAよりもVRAMを消費する可能性があります。--gradient_checkpointing や --mixed_precision を適切に設定してください。

Experimental: これらの機能は非常に実験的であり、パラメータ設定によっては学習が発散する可能性があります。

License
Apache License 2.0 (Based on sd-scripts)
