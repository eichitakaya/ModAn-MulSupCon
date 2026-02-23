# Response Letter

We thank the Editor and the Reviewers for the careful reading and constructive feedback.
We have revised the manuscript accordingly and clarified the scope of our claims.
Below we provide a point-by-point response.

## Reviewer 1

### Comment 1
> The method largely reuses existing multi-label supervised contrastive learning with metadata labels. The current framing sometimes reads as if the loss is novel rather than the application choice and evaluation.

**Response**  
Thank you for this important point. We agree and have revised the manuscript to avoid implying that the loss itself is novel. Our contribution is now consistently framed as:
- a metadata-supervised pretraining application using established multi-label supervised contrastive learning, and
- a systematic transfer evaluation on three downstream tasks.

We updated wording in the Abstract, Introduction, Experimental Setting, and table captions to make this distinction explicit.

### Comment 2
> The study evaluates ModAn-MulSupCon only with a ResNet-18 and a 2D setting. The authors also acknowledge this limitation in the Discussion, but no evidence is provided on whether the reported under stronger architectures.

**Response**  
We agree. In this revision, we did not add new large-scale experiments with stronger backbones because of resource and scope constraints for this submission cycle. To address over-interpretation, we revised the manuscript to:
- explicitly limit claims to the tested setting (ResNet-18, 2D, miniRIN pretraining), and
- clearly state this as a limitation in Discussion/Conclusion.

We also added concrete future-work items: evaluation with ViT-style backbones and 3D encoders, and broader multi-institutional settings.

各手法における事前学習には1epochあたり約10分かかること(本文にも記載あり)と、手元のgpuが一枚しかない旨も伝えたい。冒頭をwe agree.で始めるのではなく、もう少し丁寧さが欲しい。

### Comment 3
> Some recent medical image work should be discussed in the Related work, such as [1][2] doing pre-training on graph models or ConvNet, [2] doing pre-training on multimodal MRI data. - [1] Multi-modal hypergraph contrastive learning for medical image segmentation, PR - [2] Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities, MICCAI - [3] Cancer survival prediction from whole slide images with self-supervised learning and slide consistency, TMI

**Response**  
Thank you for the helpful suggestions. We expanded the Related Works section to discuss these recent directions and positioned our method relative to them:
- hypergraph/multimodal representation learning for segmentation and missing-modality robustness,
- and slide-level consistency-based self-supervised learning for pathology WSIs.

We also clarified the difference in problem setting: those studies focus mainly on segmentation/incremental or pathology-specific setups, while our work targets metadata-supervised contrastive pretraining for transferable initialization across modality-anatomy combinations in classification transfer.

### Comment 4
> Each sample appears to have exactly two active labels (one modality, one anatomy). This should be clearly explained.

**Response**  
Thank you. We now state this explicitly in the Method section.

For each image, we encode:
- one modality label (3-way one-hot), and
- one anatomy label (9-way one-hot),

then concatenate them into a 12-dimensional multi-hot target with exactly two active entries.

We additionally clarify the implication for pairwise similarity: because each sample has one modality and one anatomy label, pairwise Jaccard values are discrete (`0`, `1/3`, `1`).

### Comment 5
> The manuscript does not quantify how many positives per anchor arise, the mechanism of contrastive learning needs further clarification.

**Response**  
Thank you for this key request. We quantified positives per anchor using an analysis script that mirrors the training implementation (`scripts/losses.py`, `scripts/datasets.py`), under the default setting (`tau=0.3`, batch size `256`, two-view setup).

Main statistics (973,260 anchors):
- positives per anchor: mean `329.814`, median `353`, p5 `127`, p95 `489`, min `7`, max `507` (candidate pool `2N-1=511`),
- positive ratio `|P_tau(a)|/(2N-1)`: mean `0.6478`, median `0.6908`,
- zero-positive anchor rate: `0.000000`,
- exactly-one-positive anchor rate: `0.000000`.

Mechanism breakdown (mean extra positives per anchor):
- modality-only matches: `216.512`,
- anatomy-only matches: `53.179`,
- both modality+anatomy matches: `59.122`,
- plus one counterpart-view positive per anchor.

Threshold sensitivity:
- `tau=0.3`: mean positives `329.814`,
- `tau=0.34` and `tau=0.5`: mean positives `60.122` (exactly-one-positive rate `0.000540`).

We added this quantification and mechanism clarification to improve transparency and reproducibility.

### Comment 6
> In Eq. (1)-(3), the contrastive losses use the negatives-only denominator. The manuscript does not explicitly state whether positive samples are included in the denominator.

**Response**  
Thank you for identifying the ambiguity. In our implementation, the denominator includes all non-self samples (positives and negatives), consistent with standard InfoNCE/SupCon.

Implementation evidence (`scripts/losses.py`):
- self pairs are removed via `logits_mask` (lines `106-111`),
- denominator is computed from `exp_logits.sum(1, keepdim=True)` (lines `116-117`), where `exp_logits = exp(logits) * logits_mask`,
- positive mask is used only for averaging positive log-probabilities (line `123`).

We therefore clarified the equations and text so that:
- `A(a)` denotes all non-self in-batch views,
- `P_tau(a) \subset A(a)`,
- denominator sums over both positives and negatives; only self is excluded.

## Reviewer 2

### Comment 1
> Please describe more detail of your method in section 3.

**Response**  
Thank you. We expanded Section 3 (Method) with additional detail:
- explicit set definitions (`A(a)`, `P(a)`, `P_tau(a)`),
- explicit statement that denominator includes all non-self samples,
- clearer metadata encoding description (modality + anatomy multi-hot target),
- and additional explanation of how `tau` controls which metadata-overlap pairs become positives.

We also added quantitative positive-per-anchor statistics to better explain how the loss behaves in practice.

### Comment 2
> The size of the images of each is no more less than 1000, if possible, please add more dataset or increase more images to each dataset.

**Response**  
Thank you for this suggestion. We agree that larger downstream datasets would further strengthen the conclusions.

In this revision, we kept the three established public downstream tasks to preserve protocol comparability, and we already used repeated `10 x 5-fold` CV to reduce variance. We now state this sample-size constraint more explicitly as a limitation.

We also strengthened the future-work plan to include:
- adding larger and more diverse downstream datasets,
- scaling pretraining data beyond miniRIN,
- and broader validation across institutions and modalities.

## Summary of Revisions

- Reframed novelty claims to avoid suggesting a new loss function.
- Clarified label structure (exactly one modality + one anatomy per sample).
- Added mechanism-level quantification of positives per anchor and threshold sensitivity.
- Clarified denominator definition in Eq. (1)-(3) as all non-self samples.
- Expanded related work with the suggested recent studies and clearer positioning.
- Strengthened limitation statements and narrowed claims to tested settings.
