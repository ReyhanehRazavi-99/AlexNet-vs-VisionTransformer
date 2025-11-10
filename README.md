# AlexNet-vs-VisionTransformer
# 1. Model Descriptions
## 1(a) Architecture Summary
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









# 2) Target Datasets (A & B)
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



## Project: CIFAR-10 Transfer Learning with AlexNet — My Designs, Settings, and Results
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


## pipeline 2: alexnet on ferplus

Models and Training Regimes

We evaluated three training regimes using AlexNet pretrained on ImageNet.

Type 1 — Linear Probe (Backbone Frozen, Train Final Layer Only)

Backbone: torchvision.models.alexnet with ImageNet weights.
Frozen: All layers except the final linear layer (4096 to 8).
Optimizer: Adam with learning rate 1e-3, no weight decay.
Loss: Cross-entropy.
Learning-rate scheduling: ReduceLROnPlateau on 1 - validation accuracy.
Epochs: 10.
Input pipeline: Images streamed from Keras ImageDataGenerator, then converted and normalized for ImageNet preprocessing in PyTorch.

Results for Type 1:
Validation Accuracy: 0.5516
Test Accuracy: 0.5411
Per-class performance was strongest for high-support classes such as "neutral" and "happiness." Precision and recall were very low for minority classes (e.g., "contempt," "disgust"), which is typical for a linear probe with class imbalance and limited representational adaptation.

Type 2 — Feature Extraction + Linear SVM

Feature extractor: AlexNet with weights frozen. Output features were avgpool + flatten, producing 9216-dimensional descriptors.
Classifier: Linear SVM (scikit-learn) wrapped in a pipeline with StandardScaler and class_weight="balanced".
Training: The SVM was trained on all 9216-D training descriptors and evaluated on the validation and test descriptors.

Results for Type 2:
Validation Accuracy: 0.5649
Test Accuracy: 0.6050
Compared to Type 1, separating representation extraction and classification produced a noticeable improvement, increasing test accuracy by about 6.4 percentage points. Confusion matrices still showed weak performance on minority classes.

Type 3 — End-to-End Fine-Tuning

Backbone: AlexNet (ImageNet). The final fully connected layer was replaced with a new 4096 to 8 classifier.
Trainable parameters: Entire network was trainable (the option to freeze early convolutional layers was left commented out).
Loss: Cross-entropy with label smoothing = 0.05.
Optimizer: AdamW with learning rate 1e-4 and weight decay = 0.01.
Scheduler: CosineAnnealingLR with Tmax = 10 and eta_min = 1e-6.
Early stopping: Based on validation loss with patience of 3. Best model stored in alexnet_finetuned_best.pth.

Results for Type 3:
Validation Accuracy: 0.7966
Test Accuracy: 0.7986
This was a major improvement over Types 1 and 2, showing that adapting the entire convolutional representation to FERPlus is essential. The model performed best on "happiness" and "neutral." The "contempt" class still remained weak due to extremely low sample count. Label smoothing likely helped stabilize training.

Post-hoc Classical Classifiers on Embeddings from the Fine-Tuned Model

After fine-tuning, we extracted 4096-D embeddings by cutting AlexNet before the final linear layer. Several classical classifiers were trained on these fixed embeddings.

Linear SVM: Validation 0.7406, Test 0.7487
Logistic Regression: Validation 0.7328, Test 0.7373
k-Nearest Neighbors (k = 15, distance weighting): Validation 0.7804, Test 0.7836
Random Forest (300 trees, balanced_subsample): Validation 0.7943, Test 0.7957 (best classical classifier)

These performance levels were close to the end-to-end model, showing that the fine-tuned embeddings are linearly or semi-linearly separable and highly reusable across different classifiers. PCA and t-SNE plots indicated clearly clustered emotional categories.

Artifacts and Reproducibility

Two fully exportable pipelines were created:

AlexNet + SVM (Type 2):
svm_alexnet_pipeline.joblib (StandardScaler + linear SVM)
alexnet_features_scripted.pt (TorchScript feature extractor)
classes.json and meta.json
Packaged as alexnet_svm_artifacts.zip

Fine-tuned AlexNet (Type 3):
alexnet_finetuned_best.pth (best model state_dict)
alexnet_finetuned_scripted.pt (TorchScript inference graph)
classes.json and meta.json
Packaged as alexnet_finetuned_artifacts.zip

Both pipelines support CPU-only inference with consistent preprocessing (RGB to [0,1], ImageNet normalization, resize to 224×224).

Key Results (Single Run)

Type 1 — Linear probe with frozen AlexNet: Validation 0.5516, Test 0.5411
Type 2 — Frozen AlexNet features + Linear SVM: Validation 0.5649, Test 0.6050
Type 3 — End-to-end fine-tuning: Validation 0.7966, Test 0.7986
Type 3 embeddings + Random Forest: Validation 0.7943, Test 0.7957
Type 3 embeddings + k-NN (k=15): Validation 0.7804, Test 0.7836
Type 3 embeddings + Linear SVM: Validation 0.7406, Test 0.7487
Type 3 embeddings + Logistic Regression: Validation 0.7328, Test 0.7373

Dataset: FER2013Plus (8 classes).
Batch size: 64.
Validation split: 20 percent from training directory.

Interpretation and Notes

Type 3 substantially outperformed the other methods. The 18–25 percentage point improvement shows that FERPlus requires task-specific adaptation of the convolutional filters because facial expressions contain subtle cues not present in ImageNet.

Minority classes such as "contempt" and "disgust" have very low support in FERPlus, causing unstable precision and recall. Even with label smoothing and class-balanced weighting in classical models, these remain difficult. Techniques such as oversampling, class-balanced loss, focal loss, mixup, or cutmix could further improve those classes.

Type 2 performed better on the test set than on the validation set, likely due to benign directory-level variance rather than overfitting. Type 3 generalized consistently across validation and test.

Data augmentation was moderate and likely contributed to robustness. Stronger augmentation (RandAugment, light color jitter, cutout) may provide additional benefits without distorting facial structure.

Hyperparameters for all three regimes were well chosen for stability and fairness across comparisons.

## Transfer-Learning Setups with DeiT-Base on CIFAR-10

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


## Deit On Ferplus


My Three DeiT Models

Type 1 — Linear Probe (Backbone Frozen)

I loaded DeiT-Base with ImageNet weights and froze the entire backbone. Only the final classification head (768 to 8) was trained. I used AdamW with learning rate 5e-3, label smoothing 0.1, cosine learning rate schedule with warmup, 20 epochs, and early stopping. This evaluated the pretrained DeiT representation without modifying its features.

Results for Type 1
Validation Accuracy: about 70 percent
Test Accuracy: 68.31 percent
This was far stronger than AlexNet’s linear probe, showing the power of transformer-based representations.

Type 2 — Frozen DeiT Features With Linear SVM

I evaluated DeiT as a pure feature extractor by setting num_classes=0. This outputs the 768-dimensional CLS token embedding for each image.

I extracted embeddings for:
Train: 22,708 images
Validation: 5,678 images
Test: 7,099 images

Then I trained a Linear SVM with StandardScaler and LinearSVC(C=1), max iterations = 20000.

Results for Type 2
Validation Accuracy: 69.00 percent
Test Accuracy: 67.47 percent
Performance was similar to the linear probe, showing that DeiT CLS embeddings are highly separable.

Type 3 — Full Fine-Tuning (Best Model)

I loaded DeiT-Base with ImageNet weights, replaced the classifier with an 8-class head, and fine-tuned the entire backbone. I used AutoAugment, Random Erasing, horizontal flips, and a modern training recipe: AdamW (lr = 1e-4, wd = 0.01), label smoothing 0.1, mixed precision, cosine schedule with warmup, and early stopping.

Results for Type 3
Validation Accuracy: 85.28 percent
Test Accuracy: 83.98 percent
This was the best performance across all experiments.

Class-wise performance:
Happiness, Neutral, Surprise: strong
Sadness, Anger: moderate
Contempt, Disgust: weak due to very small sample sizes

AutoML Search Space Used

To tune efficiently on Colab, I used small, practical hyperparameter grids:

Learning Rates: 5e-3, 1e-3, 1e-4
Epochs: 10, 20, 40
Batch Size: 32, 64
Augmentation strength: AutoAugment ON/OFF
Random Erasing: p = 0.1, 0.2, 0.3
Optimizers: AdamW, optional SGD with momentum

This allowed quick but meaningful exploration without running out of Colab RAM or compute time.

Classifier Integration After Fine-Tuning

After fine-tuning, I extracted pre-logits embeddings and trained classical classifiers:

Linear SVM
Logistic Regression
k-Nearest Neighbors
Random Forest

Random Forest and k-NN performed almost as well as the fine-tuned DeiT model, showing that the learned representations were linearly separable and high quality.

Final Interpretation

Full fine-tuning (Type 3) was by far the best model, reaching about 84 percent test accuracy.
The linear probe and SVM models both performed around 68 percent.
All DeiT models greatly outperformed AlexNet-based models.
The fine-tuned DeiT model learned robust and expressive features even in the presence of severe class imbalance.


