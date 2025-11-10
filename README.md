# AlexNet-vs-VisionTransformer
#1. Model Descriptions
1(a) Architecture Summary
Introduction

In this project I evaluated transfer learning performance using two different pretrained backbone architectures: a convolutional neural network (AlexNet) and a transformer-based vision model (DeiT-Base). These models were selected to represent the two major families of visual feature extractors—CNNs and Vision Transformers—and to reproduce the transfer-learning behaviors described in What Makes Transfer Learning Work for Medical Images while applying them to my two target datasets: FERPlus (emotion recognition) and CIFAR-10 (natural images). Below, I summarize the architectural properties of each backbone as implemented in my experiments, including input resolution, depth, patch/convolution structure, and total parameter counts after adapting their classification heads to the number of classes in FERPlus and CIFAR-10.

AlexNet (CNN Backbone)

AlexNet served as the CNN backbone in this study. Although originally introduced in 2012, it remains widely used as a baseline architecture in transfer-learning analyses because of its relatively simple structure and its clear division between convolutional feature extractors and fully connected classification layers.

Architecture Summary

Input Resolution Used in This Project:
All experiments used 224×224 RGB images, normalized with ImageNet mean and standard deviation.

Convolutional Feature Extractor (5 convolution layers):

Conv1: 64 filters, 11×11 kernel, stride 4, padding 2, ReLU → MaxPool(3×3, stride 2)

Conv2: 192 filters, 5×5 kernel, padding 2, ReLU → MaxPool(3×3, stride 2)

Conv3: 384 filters, 3×3 kernel, padding 1, ReLU

Conv4: 256 filters, 3×3 kernel, padding 1, ReLU

Conv5: 256 filters, 3×3 kernel, padding 1, ReLU → MaxPool(3×3, stride 2)

The convolutional layers produce a 256 × 6 × 6 feature map before flattening.

Classifier (fully connected block):

FC1: 9216 → 4096 (ReLU, Dropout)

FC2: 4096 → 4096 (ReLU, Dropout)

FC3: 4096 → num_classes (classification head)

In my experiments, this final layer was replaced depending on the dataset:

FERPlus: 4096 → 8

CIFAR-10: 4096 → 10

Model Size

The original ImageNet AlexNet has ~61.1M parameters.
Replacing the final 4096→1000 layer with a smaller output layer reduces the total to approximately:

FERPlus: ~57M parameters

CIFAR-10: ~57M parameters

Relevance for Transfer Learning

AlexNet provides a classical CNN hierarchy with spatial locality, shared convolutional filters, and deeper fully connected layers that can be selectively frozen or unfrozen.
This separation makes it ideal for implementing:

Linear probe: freeze all convolutional and FC layers except the final classifier.

Partial fine-tuning: unfreeze Conv5 (or last few convolutional layers).

Full fine-tuning: unfreeze all layers.

DeiT-Base (Vision Transformer Backbone)

DeiT-Base (from the Data-efficient Image Transformers family) served as the Vision Transformer (ViT) model in this project. It uses patch embeddings and multi-head self-attention instead of convolutional filters, making it fundamentally different from AlexNet in terms of inductive biases and feature reuse.

Architecture Summary

Input Resolution Used in This Project:
All experiments used 224×224 RGB images, with ImageNet normalization.

Patch Embedding

Patch size: 16×16

A 224×224 image becomes 14×14 = 196 patches

Each patch is linearly projected into a 768-dimensional embedding

A learned class token is prepended, giving a total sequence length of 197 tokens.

Positional Embeddings

A learned positional embedding of size 197 × 768 is added to the token sequence.

Transformer Encoder

Number of blocks: 12

Embedding dimension: 768

Attention heads: 12 per block

FFN dimension: 3072 (MLP ratio = 4)

Each block includes:

Multi-Head Self-Attention

Residual connections

LayerNorm

MLP feedforward network

Classification Head

A linear projection:

768 → num_classes

Customized in my runs:

FERPlus: 768 → 8

CIFAR-10: 768 → 10

Model Size

DeiT-Base has approximately 86M parameters.
Replacing the final head reduces the parameter count slightly (~0.7M fewer), resulting in:

FERPlus: ~85.9M parameters

CIFAR-10: ~85.9M parameters

Relevance for Transfer Learning

Unlike AlexNet, DeiT does not hard-code locality or translational equivariance. Its attention-based representation makes it highly sensitive to data scale and transfer regimes. It supports:

Linear probe: freeze the entire transformer; train only the 768→C head.

Partial fine-tuning: unfreeze only the final transformer block(s).

Full fine-tuning: unfreeze all 12 blocks and the head.

This allows a direct investigation of how ViTs reuse high-level attention features compared to CNN residual/convolutional features.









#2) Target Datasets (A & B)
Dataset A: CIFAR-10 (Closer to ImageNet Domain)

In this project, Dataset A is CIFAR-10, a widely used natural-image classification benchmark that is considered close to ImageNet in both content and visual statistics. CIFAR-10 consists of small, colored, object-centered natural images; this makes it an ideal choice for studying transfer learning effectiveness in settings where the source and target domains are similar.

Domain

CIFAR-10 contains natural RGB images of everyday objects such as animals and vehicles. Because ImageNet contains similar categories and photographic properties, CIFAR-10 represents a high-similarity domain relative to the pretrained weights of both AlexNet and DeiT-Base.

Image Properties

Resolution: 32×32 pixels (upsampled to 224×224 for pretrained ImageNet models).

Color: 3-channel RGB.

Format: Low-resolution, centered objects.

Dataset Size

Total samples: 60,000 images

Training set: 50,000 images

Test set: 10,000 images

Class Labels (10 classes)

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

Train/Validation/Test Split (In My Project)

I follow a stratified split from the original training set:

40,000 images → Training

10,000 images → Validation (20% of CIFAR-10 train)

10,000 images → Test (original CIFAR-10 test set)

This ensures identical splits across all transfer regimes (linear probe, partial fine-tune, full fine-tune) to maintain fairness.

Dataset B: FERPlus (Farther From ImageNet Domain)

For Dataset B, I use FERPlus, a facial expression recognition dataset that is substantially more distant from ImageNet in both domain and visual properties. FERPlus features grayscale facial images without the natural-scene context seen in ImageNet. This domain shift allows me to examine the transferability of CNNs versus ViTs under conditions that mirror the “far-from-ImageNet” scenarios discussed in Matsoukas et al. (2022).

Domain

FERPlus contains human facial expression images, collected from the FER2013 dataset and re-annotated through multiple crowd-sourced votes. Unlike ImageNet, the dataset contains:

facial crops instead of natural scenes

monochrome or low-color images

emotion-based class distinctions rather than object categories

This makes FERPlus an excellent “domain-shifted” dataset to test transfer robustness.

Image Properties

Resolution: 48×48 pixels (resized to 224×224 for ImageNet-pretrained models)

Channels: Primarily grayscale (converted to 3-channel RGB)

Content: Aligned faces with varying emotions

Variation: Expression, lighting, partial occlusion, pose differences

Dataset Size

FERPlus has:

28,709 training images

3,589 validation images

3,589 test images

Class Labels (8 emotion categories)

anger

disgust

fear

happiness

sadness

surprise

neutral

contempt

(Depending on the annotation scheme, some implementations collapse or expand classes, but I use the standard 8-label format.)

Train/Validation/Test Split (In My Project)

FERPlus already contains train/val/test folders.
To maintain consistency with the transfer learning protocol:

I use the provided test set untouched.

For training, I apply a 20% stratified validation split inside the provided training folder:

~22,967 images → Training

~5,742 images → Validation

~3,589 images → Test

This gives comparable train/val sizes to CIFAR-10 and ensures fair cross-model comparisons.




#Project: CIFAR-10 Transfer Learning with AlexNet — My Designs, Settings, and Results
Dataset & Preprocessing

I used CIFAR-10 (50k train / 10k test; 10 classes). Because AlexNet is pretrained on ImageNet at 224×224, I resized each image to 224×224 and normalized inputs with the standard ImageNet mean/std. For evaluation I used Resize(256) → CenterCrop(224) → ToTensor → Normalize. For training in the fine-tuning settings I added light augmentation (RandomResizedCrop, RandomHorizontalFlip) and, in the full fine-tune, also ColorJitter.

My Model Families (what I built and why)
1) Frozen Feature Extractor → Linear SVM (no fine-tuning)

What I did. I loaded AlexNet with ImageNet pretrained weights and set the model to eval() so no weights update. I forwarded images through features → avgpool → flatten to get a 9216-dim feature per image, saved those features for train/test, and trained a LinearSVC (with standardization in a pipeline) on top.

Why I did it. This isolates representation quality from deep optimization—fast to run, and a clean baseline.

Key settings.

Batch for feature pass: 128

Feature dim: 9216 (AlexNet 224×224)

Classifier: StandardScaler → LinearSVC(C=1.0, max_iter=10k)

Result. 71.95% test accuracy with a pure frozen-feature + Linear SVM pipeline (quick baseline).

2) Linear Probe (freeze backbone; train only the last FC layer)

What I did. I replaced the last FC layer (classifier[6]) with a 10-way linear layer and froze all other AlexNet parameters. I trained only this new head with cross-entropy.

Why I did it. It’s the standard “linear-probe” diagnostic: very fast, and usually stronger than classical classifiers on frozen features.

Key settings.

Optimizer: AdamW (head only), LR = 1e-3, cosine schedule (10 epochs)

Train transforms: (i) minimal eval-style for a clean probe (first run), (ii) a light augmented setup (second run)

Results.

Linear probe (minimal aug): 83.61% test accuracy

Linear probe (light aug, separate run): 82.76% test accuracy
So my linear-probe results sit around 82–84% on CIFAR-10.

3) Partial Fine-Tune (unfreeze AlexNet last conv block + full classifier)

What I did. I froze the early convolutional layers and unfroze the tail of the conv stack (starting at features[10]) and the entire classifier. I used discriminative LRs (smaller for the unfrozen conv tail, larger for the classifier).

Why I did it. This lets me adapt high-level features to CIFAR-10 semantics without touching the whole backbone (a good accuracy/compute trade-off).

Key settings.

Train aug: RandomResizedCrop(224, scale=(0.8,1.0)), RandomHorizontalFlip

Optimizer: AdamW with two param groups
– Conv tail: LR = 3e-5, WD = 1e-4
– Classifier: LR = 1e-3, WD = 1e-4

Scheduler: CosineAnnealingLR, 10 epochs

Result. Final test accuracy 88.23% with strong, steady gains over epochs.

4) Full Fine-Tune (train all layers end-to-end)

What I did. I replaced the last layer with a 10-class head and trained the entire network with SGD + momentum, mixed precision, cosine LR, a held-out validation split (5k from train), and best-checkpoint saving.

Why I did it. This is the highest-capacity setting and, on CIFAR-10, should beat partial FT when regularized decently.

Key settings.

Split: 45k train / 5k val / 10k test

Train aug: RandomResizedCrop(224, scale=(0.6,1.0)), HorizontalFlip, ColorJitter

Optimizer: SGD (momentum 0.9), LR = 0.01, WD = 5e-4, cosine schedule (40 epochs)

Mixed precision (AMP), best-val checkpointing

Result. Best checkpoint at epoch 35; Test = 91.78%.

5) SVM (and other classical classifiers) on the penultimate features

What I did. After full fine-tuning, I extracted 4096-dim penultimate features (i.e., after classifier[0..5], before the final 10-way layer) for train/val/test. I then trained a suite of classical classifiers.

Why I did it. To check how linearly separable the learned representation became and to compare classical classifiers on top of my best learned features.

Models I tried & results (Val/Test):

Linear SVM: 91.88% / 91.42%

Logistic Regression (multinomial): 91.88% / 91.54%

SVM (RBF) with PCA→256: 92.50% / 91.85%, and after refitting on train+val it reached 92.01% on test

SGD (hinge): 91.40% / 91.15%

kNN (PCA→128, k=7): 91.04% / 90.66%

Random Forest (400 trees): 91.58% / 90.79%

MLP (512 hidden): 91.12% / 90.51%

Takeaway. On my fine-tuned representation the best classical model was SVM with RBF kernel + PCA(256), peaking at ~92% test accuracy. Linear SVM and multinomial logistic were right behind (~91.4–91.5%). This confirms my full-FT features are highly separable.

Pretraining details (what weights I used and how they were trained)

Backbone & weights. I initialized AlexNet with supervised ImageNet-1k (AlexNet_Weights.IMAGENET1K_V1 from torchvision). The model card confirms these weights reproduce paper-level results and exposes the IMAGENET1K_V1/DEFAULT aliases. 
PyTorch Documentation
+1

Dataset used for pretraining. ImageNet-1k / ILSVRC (≈1.2M labeled images, 1000 classes). The challenge and dataset are described in Russakovsky et al. (2015). 
SpringerLink
+1

Pretraining strategy. Supervised classification using the AlexNet architecture introduced by Krizhevsky, Sutskever & Hinton (2012). The original recipe used random crops, horizontal flips, and PCA color (Fancy PCA) augmentation, plus ReLU and dropout in the FC layers. Mixup, CutMix, and RandAugment did not exist at that time and were not used in AlexNet pretraining. 
NeurIPS Proceedings

Modern fine-tuning augments (optional, downstream). When I fine-tuned, I stuck to classical flips/crops and mild ColorJitter. If I expand the search, I will try RandAugment (Cubuk et al., 2019/2020), Mixup (Zhang et al., 2017/2018), and CutMix (Yun et al., 2019) as downstream regularizers (not part of the 2012 pretraining). 
CVF Open Access
+4
CVF Open Access
+4
arXiv
+4

Short Auto-ML search space I actually target (per model)

I kept the search small and Colab-friendly. For a quick model/augment selection, I suggest these grids:

Shared knobs

Epochs (short runs): {6, 10, 15}

Batch size (VRAM-gated): {16, 32, 64} (I often used 128 when available)

Optimizer: AdamW or SGD+momentum

Scheduler: Cosine w/ warmup (3–10% iters) or Step (70/90%)

Weight decay: {1e-4, 5e-4}

Augment (downstream only): baseline (Crop+Flip) vs +ColorJitter vs RandAugment; optionally Mixup α∈{0.1, 0.2} and CutMix β∈{0.5, 1.0} for the fine-tuning settings. 
CVF Open Access
+2
arXiv
+2

Frozen features → Linear SVM

No epochs (train on features).

SVM C: {0.1, 1, 10}; class_weight ∈ {None, "balanced"}

Optional: PCA {256, 512} before RBF SVM.

Linear probe (head-only)

LR (head): {1e-3, 3e-4, 1e-4}

Epochs: {6, 10, 15}

Batch size: {32, 64}

Partial fine-tune (last conv block + head)

LRs (head / conv-tail): {(1e-3 / 1e-4), (3e-4 / 3e-5)}

Epochs: {10, 15}

Batch size: {32, 64}

Light ColorJitter helpful.

Full fine-tune

LR (SGD): {0.01, 0.02} with momentum 0.9, WD {1e-4, 5e-4}

Epochs: {15, 30, 40} for short-to-medium sweeps

Keep a val split (e.g., 5k) and save best-val checkpoints.

What I learned across my models

Frozen features + Linear SVM gave me a quick 72% baseline.

Linear probe jumped to ~82–84%, confirming the pretrained representation transfers strongly to CIFAR-10.

Partial fine-tune (last block + head) pushed to ~88%, a great speed/accuracy compromise.

Full fine-tune reached ~91.8% test accuracy with proper regularization and checkpointing.

On my fine-tuned representation, classical classifiers (especially SVM-RBF + PCA) hit ~92%, showing that the learned penultimate features are highly separable.

Reproducibility notes

I set seed=42 for Python/NumPy/Torch and used pinned memory and num_workers=2 for the loaders.

I logged training/validation curves, stored the best checkpoint by validation accuracy in full FT, and reported test metrics at the best checkpoint.



Data & Setup

Dataset. We used FER2013Plus (8 classes: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise). The dataset was fetched in Google Colab via the Kaggle CLI after uploading kaggle.json (Account → Create New API Token). Files were downloaded and unzipped to /content/fer2013plus.

Directory layout. We pointed Keras to the dataset under one of:

/content/fer2013plus/fer2013plus/fer2013/{train,test} (primary), or

/content/fer2013plus/{train,test} (fallback).

Preprocessing. Images were resized to 224×224 and rescaled to 
[
0
,
1
]
[0,1]. During PyTorch forward passes, we additionally applied ImageNet normalization with mean 
[
0.485
,
0.456
,
0.406
]
[0.485,0.456,0.406] and std 
[
0.229
,
0.224
,
0.225
]
[0.229,0.224,0.225].

Augmentation. On the training split, we used:

rotation_range 
15
∘
15
∘
,

width/height shift 
0.1
0.1,

zoom 
0.1
0.1,

horizontal flip.

Splits. We created a virtual validation split by reserving 20% of the training directory via validation_split=0.2. Final counts detected by Keras:

Train: 22,712 images (8 classes)

Validation: 5,674 images (8 classes)

Test: 7,099 images (8 classes)

Batch size. 64 across all Keras generators.

Models and Training Regimes

We evaluated three regimes with AlexNet (ImageNet pretraining):

Type 1 — Linear Probe (Freeze Backbone; Train Final Layer Only)

Backbone: torchvision.models.alexnet(weights=IMAGENET1K_V1).

Frozen: All layers except the final linear (
4096
→
8
4096→8).

Optimizer: Adam (
lr
=
10
−
3
lr=10
−3
), no weight decay.

Loss: Cross-entropy.

LR scheduling: ReduceLROnPlateau on 
1
−
val_acc
1−val_acc.

Epochs: 10.

Input pipeline: Batches streamed from Keras ImageDataGenerator, then normalized for ImageNet in PyTorch.

Results.

Validation Acc: 0.5516

Test Acc: 0.5411

Per-class reports showed strong performance on majority classes (neutral, happiness) and weak precision/recall for minority classes (e.g., contempt, disgust), consistent with class imbalance and limited capacity in a linear probe setting.

Type 2 — Feature Extraction + Linear SVM

Feature extractor: AlexNet features + avgpool + flatten → 9216-D descriptors (frozen).

Classifier: scikit-learn SVM (linear kernel) inside a pipeline with StandardScaler; class_weight='balanced'.

Training: Fit the SVM on all training descriptors; evaluate on validation/test features.

Results.

Validation Acc: 0.5649

Test Acc: 0.6050

Compared to Type 1, decoupling representation learning (fixed) and a stronger linear margin classifier improved test accuracy by ~6.4 pp. Confusion matrices still reflected minority-class underperformance.

Type 3 — End-to-End Fine-Tuning (Keras Generators → PyTorch)

Backbone: AlexNet (ImageNet). Final head replaced: 
4096
→
8
4096→8.

Trainable: Full network (you left option to freeze early convs commented out).

Loss: Cross-entropy with label smoothing = 0.05.


Early stopping: on validation loss with patience = 3; best weights saved to alexnet_finetuned_best.pth.

Results.

Validation Acc: 0.7966

Test Acc: 0.7986

Large gains over Types 1–2, indicating that updating the convolutional representation on FERPlus is crucial. Performance remained strongest on happiness and neutral; contempt underperformed due to extremely low support (common on FERPlus). Label smoothing likely contributed to stability.

Post-hoc Classical Classifiers on Fine-Tuned Embeddings

After fine-tuning, we extracted 4096-D embeddings by cutting AlexNet right before the final linear layer (classifier up to ReLU
5
5
	​

). We then trained classical classifiers on these embeddings:

Linear SVM: Val 0.7406, Test 0.7487

Logistic Regression: Val 0.7328, Test 0.7373

k-NN (k=15, distance weights): Val 0.7804, Test 0.7836

Random Forest (300 trees, balanced_subsample): Val 0.7943, Test 0.7957 (best classical)

These closely matched the end-to-end model, showing that fine-tuned representations are linearly/semi-linearly separable and highly reusable across classifiers. We also produced PCA and t-SNE 2-D plots for qualitative cluster structure.

Artifacts & Reproducibility

We exported self-contained bundles for local inference:

AlexNet+SVM (Type 2)

svm_alexnet_pipeline.joblib (StandardScaler + linear SVC),

alexnet_features_scripted.pt (TorchScript feature extractor),

classes.json, meta.json, zipped to alexnet_svm_artifacts.zip.

Fine-tuned AlexNet (Type 3)

alexnet_finetuned_best.pth (best checkpoint dict),

alexnet_finetuned_scripted.pt (TorchScript inference graph),

classes.json, meta.json, zipped to alexnet_finetuned_artifacts.zip.

This lets you do CPU-only local inference with consistent preprocessing (RGB → 
[
0
,
1
]
[0,1] → ImageNet normalize → 224×224).

Key Numbers (single-run)
Regime	Validation Acc	Test Acc
Type 1 — Linear probe (frozen AlexNet, train last layer)	0.5516	0.5411
Type 2 — AlexNet features + Linear SVM	0.5649	0.6050
Type 3 — End-to-end fine-tuning	0.7966	0.7986
Type 3 embeddings → Random Forest	0.7943	0.7957
Type 3 embeddings → k-NN (k=15)	0.7804	0.7836
Type 3 embeddings → Linear SVM	0.7406	0.7487
Type 3 embeddings → Logistic Regression	0.7328	0.7373

(All on 8-class FER2013Plus; batch size 64; 20% validation split from train directory.)

Interpretation & Notes

Why Type 3 wins. The 18–25 pp jump over Types 1–2 shows that task-specific adaptation of early/mid CNN layers matters on FERPlus (domain shift from ImageNet; facial expressions are subtle and benefit from tuning low-level filters and higher-level features).

Minority classes. Contempt and disgust have very low support; most runs show unstable precision/recall there. Your training used label smoothing and balanced class weights in the SVM/LogReg baselines, but additional tactics (oversampling, class-balanced loss, focal loss, mixup/cutmix) could target those classes specifically.

Validation vs Test. Type 2 slightly over-performs on test vs val (0.605 vs 0.565), likely benign variance from the directory structure. Type 3 generalizes consistently (≈0.80 on both).

Augmentations. The moderate augmentation you used likely helped robustness without heavy distortions. If you need more gains, consider stronger policies (RandAugment), light color jitter, or Cutout; keep faces natural to avoid breaking expression cues.

Hyperparameters. Your choices were sensible:

Linear probe: Adam 
10
−
3
10
−3
 on the last layer only.

Fine-tuning: AdamW 
10
−
4
10
−4
, Cosine schedule, early stopping on val loss, label smoothing 0.05—each stabilizes training and reduces overfit.

Pipelines as deliverables. The TorchScript exports and joblib SVM bundle make it easy to:

run webcam/demo with the fine-tuned model, or

do fast CPU inference via feature extraction + SVM.


Transfer-Learning Setups with DeiT-Base on CIFAR-10

We evaluated three standard transfer-learning regimes using a DeiT-Base (patch-16, 224×224) backbone from timm on CIFAR-10. For all models, we (a) resized images to 224×224 and normalized with ImageNet statistics, (b) used AutoAugment (ImageNet policy), RandomHorizontalFlip, and RandomErasing (p=0.2) for training, (c) trained with mixed precision (autocast + GradScaler), AdamW, cosine learning-rate schedule with warmup, and early stopping on validation loss. The CIFAR-10 training set was split stratified 80/20 into train/validation (40,000/10,000); the official 10,000-image test set was held out for final evaluation. Unless noted, we used label smoothing as specified below.

Type 1 — Linear Probe (Head-Only)

Goal. Assess the discriminative power of frozen DeiT features via a lightweight supervised adapter.

Setup.

Initialize ImageNet-pretrained DeiT-Base; freeze the entire backbone.

Replace the classification head with a fresh linear layer for 10 classes; train only the head.

Batch size = 128, epochs = 30, early stopping patience = 5.

Optimizer/Loss/Schedule. AdamW on head params only (lr = 3e-3, weight decay = 0.0); no label smoothing (0.0). Cosine LR with warmup.

Capacity. Total params: 85,806,346; trainable: 7,690 (head only).

Result (Test). 95.01% accuracy (macro F1 ≈ 0.95). This confirms that frozen DeiT representations transfer strongly to CIFAR-10 with minimal task-specific parameters.

Type 2 — Partial Fine-Tune (Last Block + Norm + Head)

Goal. Trade a modest increase in trainable capacity for better alignment of high-level features to CIFAR-10.

Setup.

Start from the same ImageNet-pretrained DeiT-Base; freeze all layers.

Unfreeze only: the last transformer block (blocks[-1]), the final LayerNorm (model.norm), and the classification head.

Batch size = 64, epochs = 40, early stopping patience = 5.

Discriminative LRs. AdamW with parameter groups: last block and norm 5e-5, head 1e-4; weight decay = 0.01; label smoothing = 0.1. Cosine LR with warmup.

Capacity. Total params: 85,806,346; trainable: 7,097,098.

Result (Test). 97.15% accuracy (macro F1 ≈ 0.97). Compared with Type 1, carefully unfreezing the final stage substantially improves performance while keeping compute and overfitting risk controlled.

Type 3 — End-to-End Fine-Tuning

Goal. Fully adapt the representation to the target dataset.

Setup.

Initialize ImageNet-pretrained DeiT-Base; unfreeze all layers and fine-tune end-to-end.

Batch size = 64, epochs = 40, early stopping patience = 5.

Optimizer/Loss/Schedule. AdamW on all params (lr = 1e-4, weight decay = 0.01); label smoothing = 0.1. Cosine LR with warmup.

Capacity. Total/Trainable params: 85,806,346.

Result (Test). 97.36% accuracy (macro F1 ≈ 0.97), the best among the three regimes, with small but consistent gains over partial fine-tuning.

Comparison & Takeaways

Data split/validation usage. All models used the same stratified 80/20 train/validation split for model selection and early stopping; the test set remained untouched until final evaluation.

Performance vs. trainable parameters.

Linear probe achieves strong results with ~7.7K trainable params (95.01%).

Partial fine-tuning adds ~7.1M trainable params, yielding a large jump to 97.15%.

Full fine-tuning (all 85.8M params trainable) provides a smaller additional gain to 97.36%.

Regularization/optimization details. Label smoothing (0.1) and weight decay (0.01) were beneficial when backbone layers were trainable (Types 2–3), while the linear probe favored no label smoothing and a higher head LR.

Practical guidance. If compute or overfitting is a concern, Type 2 offers an excellent trade-off. When maximum accuracy is the priority and resources allow, Type 3 is preferred. Type 1 is ideal for rapid baselines and low-resource scenarios.

Image Preprocessing and Augmentation

I used standard ImageNet normalization because DeiT was pretrained on ImageNet-1k:

Mean = 
[
0.485
,
0.456
,
0.406
]
[0.485,0.456,0.406]

Std = 
[
0.229
,
0.224
,
0.225
]
[0.229,0.224,0.225]

All images were resized to 224×224.

Augmentation (Train Only)

I used DeiT-appropriate strong augmentation:

AutoAugment(IMAGENET)

RandomHorizontalFlip

RandomErasing (p=0.2)

Resize → ToTensor → Normalize

These heavy augmentations are consistent with DeiT’s official training recipe.

✅ Pretraining Details for DeiT

I used DeiT-Base (patch16/224) from timm, with pretrained ImageNet weights.

Pretraining Dataset

ImageNet-1k (1.28M images, 1000 classes)

Training Strategy

Supervised pretraining, not self-supervised

Based on the original DeiT paper (Touvron et al., 2021)

Original DeiT augmentations

I relied on augmentations used during DeiT’s official training:

RandAugment / AutoAugment

Mixup

CutMix

Random Erasing

Stochastic Droplayer

Label Smoothing

I replicated only the augmentations compatible with FERPlus and kept the remaining components (stochastic depth, positional embeddings, DeiT training hyperparameters) via the pretrained model.

Citations

Touvron et al. (2021). Training data-efficient image transformers & distillation through attention.

Timm model card: https://github.com/rwightman/pytorch-image-models

✅ MY Three DeiT Models

Below is the full write-up of all three DeiT pipelines I trained.

⭐ TYPE 1 — Linear Probe (I Freeze the Entire DeiT Backbone)
What I Did

I created a typical linear probe setup. I:

Loaded DeiT-Base with ImageNet weights.

Froze 100% of the model parameters.

Only trained the final classification head (4096→8).

Used:

AdamW (lr = 5e-3)

Label smoothing = 0.1

Cosine LR with warmup

20 epochs + early stopping

This allowed me to measure the quality of DeiT’s pretrained representation without modifying its features.

Result

Validation Accuracy: ~70%

Test Accuracy: 68.31%

This linear probe significantly outperformed AlexNet’s linear probe, showing DeiT’s stronger representation.

⭐ TYPE 2 — Frozen DeiT Features + Linear SVM
What I Did

I evaluated DeiT as a pure feature extractor by setting:

num_classes=0


This returned the 768-dim CLS token embedding for each image.

I extracted embeddings for:

Training (22,708 images)

Validation (5,678 images)

Test (7,099 images)

Then I trained a classical Linear SVM:

StandardScaler → LinearSVC(C=1)

Maximum iterations = 20,000

Results

Validation: 69.00%

Test: 67.47%

Performance was similar to the linear probe, with slight differences depending on class imbalance.

This showed that the DeiT CLS embeddings are highly separable and usable for classical ML methods.

⭐ TYPE 3 — Full Fine-Tuning (Most Important and Best Performing Model)
What I Did

This was my strongest model. I:

Loaded DeiT-Base pretrained on ImageNet.

Replaced the classifier with an 8-class head.

Kept the entire backbone trainable.

Used a strong augmentation pipeline:

AutoAugment (IMAGENET)

Random Erasing

Horizontal flips

Used a modern training recipe:

AdamW (lr = 1e-4, wd = 0.01)

Label smoothing = 0.1

Mixed precision training

Cosine LR schedule with warmup

Early stopping on validation loss

Results

This model achieved the best performance across all experiments:

Validation Accuracy: 85.28%

Test Accuracy: 83.98%

This is a very strong result for FERPlus (8-class).

Class-wise performance

Happiness, Neutral, Surprise: excellent

Sadness and Anger: moderate

Contempt & Disgust: poor recall (dataset imbalance → only 50–60 images)

✅ AutoML Search Space I Used

To tune my DeiT models efficiently on Colab, I used short runs and reasonable hyperparameter grids:

Learning Rate

{5e-3, 1e-3, 1e-4}

Epochs

{10, 20, 40}

Batch Size

{32, 64}

Augmentation strength

AutoAugment ON/OFF

Random Erasing p = {0.1, 0.2, 0.3}

Optimizers

AdamW

SGD+Momentum (optional)

This grid allowed small but meaningful adjustments without overloading Colab resources.

✅ Classifier Integration (What I Did After Fine-Tuning)

After training DeiT, I extracted embeddings from the final pre-logits layer and fed them into several classical classifiers:

Linear SVM

Logistic Regression

k-Nearest Neighbor

Random Forest

Random Forest and k-NN performed almost as well as DeiT itself, showing that the fine-tuned representation became linearly separable.

✅ Final Interpretation

DeiT fine-tuning (Type 3) was by far the strongest model (~84% test accuracy).

DeiT linear probe and SVM models performed similarly (~68%).

DeiT models dramatically outperformed all AlexNet variants.

The fine-tuned DeiT model learned robust expression features even with class imbalance.




