---
title: Assignment applications
description: Historical and modern applications of the linear assignment problem — from 1950s operations research to DETR, multi-object tracking, and cluster evaluation in contemporary deep learning.
date: 2026-07-07
tags: [assignment, history]
---

The linear assignment problem is one of the oldest and most broadly applied combinatorial
optimisation problems. This page traces where it shows up, from its origins in workforce
scheduling through to the set-prediction losses that train modern vision transformers.

## Operations research origins

The problem was formalised in the context of **personnel assignment**: given a set of
workers and a set of jobs, and a productivity rating for each worker-job pair, find the
assignment that maximises total productivity. Kuhn's 1955 paper \[Kuhn1955\] coined
the "Hungarian method" motivated precisely by this framing. The same structure appeared
under the name **transportation problem** in the linear programming literature of the same
decade (Hitchcock 1941, Koopmans 1947) — matching supply nodes to demand nodes at minimum
shipping cost — and as a **bipartite matching** in graph theory. All three are the same
problem with different vocabulary.

The assignment problem remains a daily tool in operations research: assigning gates to
arriving aircraft, assigning drivers to delivery routes, matching kidney donors to
recipients, scheduling nurses across shifts. The combinatorial structure is always the same:
two disjoint sets, a cost or benefit for every pair, and a requirement for a one-to-one
correspondence.

## Multi-object tracking

The dominant use case in modern computer vision. A tracker maintains a set of **tracks**
(object hypotheses across time) and receives a set of **detections** (bounding-box outputs
from a detector) each frame. The task is to assign each detection to the track it most
likely continues, while flagging new tracks and terminating lost ones.

The cost matrix is typically `1 − IoU` between all track-detection box pairs, optionally
supplemented by centroid distance, appearance similarity (re-id embedding distance), or
learned affinity scores. Gate entries whose pairs are geometrically infeasible are set to
`+inf`. The assignment is solved per-frame; with a batch dimension over frames it maps
directly onto `jonker_dense_batch`.

SORT \[Bewley2016\] established this template in 2016: linear motion prediction via a Kalman
filter, IoU cost, Hungarian assignment. DeepSORT \[Wojke2017\] added appearance features to
the cost. ByteTrack \[Zhang2022\] extended the idea to also process low-confidence
detections through a second assignment pass. BoT-SORT \[Aharon2022\] and OC-SORT
\[Cao2023\] refined the motion model and re-identification respectively, but the assignment
step remains structurally unchanged across the entire family.

The `_unpacked` op variants — `jonker_dense_batch_unpacked`,
`jonker_compact_batch_unpacked` — return matched pairs, unmatched track indices, and
unmatched detection indices in one pass, replacing the per-batch-element Python loop that
most tracker implementations use to recover these three sets from a raw assignment.

## Set-prediction losses (DETR family)

Transformer-based object detectors output a **fixed-size set** of predictions (bounding
boxes and class logits), padded with no-object slots to a common count. Computing the
training loss requires matching each ground-truth object to exactly one prediction before
evaluating the per-pair L1, GIoU, and classification terms. The matching must be globally
optimal — greedy or random assignment produces a worse training signal.

DETR \[Carion2020\] introduced this formulation and placed the Hungarian matcher explicitly
on the critical path. Subsequent work in the DETR lineage — Conditional DETR, DN-DETR,
DINO-DETR, MaskFormer \[Cheng2021\], Mask2Former, RT-DETR — all preserve the Hungarian
matching step, with variations in how the cost components are weighted or whether matching
is applied across multiple decoder layers. The matcher runs at training time, once per image
per gradient step; its latency directly affects training throughput at large batch sizes.

Batching the per-image cost matrices into a single 3D tensor and solving with
`jonker_dense_batch` eliminates the per-image Python loop and exposes the problem to
`at::parallel_for` on CPU or the tiled CUDA kernel for small square problems.

## Cluster evaluation

Metrics for unsupervised learning and semi-supervised segmentation require an **optimal
relabelling** between predicted cluster IDs and ground-truth class IDs before counting
agreements. Because cluster labels are arbitrary integers — the model could call its cat
cluster `3` and the ground truth calls it `7` — a naive comparison gives a meaningless
score. The correct score uses the permutation that maximises agreements.

The confusion matrix yields an assignment problem: build a `K × K` cost matrix
as the negative confusion-matrix entry `−C[i,j]` (number of samples with predicted label
`i` and true label `j`), then find the permutation of predicted labels that maximises
total agreement. The resulting **assignment accuracy (ACC)** and the **adjusted Rand index
(ARI)** (a metric that measures agreement between two clusterings, corrected for chance)
both depend on this optimal permutation.

Domains where this matters include unsupervised semantic segmentation, clustering
benchmarks (ImageNet-1K linear evaluation, STL-10, CIFAR-100 supercategory), online
quantization (codebook alignment in VQ-VAE variants), and re-identification evaluation.

## Pose estimation and part matching

Multi-person pose estimators that use bottom-up detection (methods that detect individual
body parts across the image and group them into person instances) must group detected
keypoints into person instances. Each detected keypoint is a candidate node; a bipartite
assignment on a joint confidence matrix groups keypoints across body-part types into
coherent skeletons. Top-down methods instead assign detected bounding boxes to tracked
persons across frames — again LAP.

In 6-DoF pose estimation, the assignment problem arises when matching predicted object
instances to ground-truth annotations where multiple overlapping instances are present.

## Graph matching and molecular structure

Computing the similarity between two graphs requires a **bijection between nodes** that
maximises the number of matched edges — the graph isomorphism problem in its weighted
variant. For small graphs or graphs with good node features, a LAP over a node-similarity
cost matrix gives a tractable approximation. Applications include molecular graph
matching (drug-candidate similarity), knowledge-graph entity alignment, and circuit
netlist comparison.

## Sequence alignment (discrete)

Discrete sequence alignment with known length — matching tokens in a predicted sequence to
tokens in a reference — appears in evaluation metrics for structured prediction tasks:
matching predicted named entities to gold entities, matching predicted spans to gold spans
in reading comprehension, or aligning predicted parses to gold parses. When the two
sequences are already segmented into labelled spans, a LAP on a span-overlap cost matrix
gives the optimal alignment.

## References

- \[Kuhn1955\] Kuhn, H. W. (1955). *The Hungarian method for the assignment problem*.
  **Naval Research Logistics Quarterly**, 2(1–2): 83–97.
- \[Bewley2016\] Bewley, A.; Ge, Z.; Ott, L.; Ramos, F.; Upcroft, B. (2016). *Simple online
  and realtime tracking*. **ICIP**, pp. 3464–3468.
- \[Wojke2017\] Wojke, N.; Bewley, A.; Paulus, D. (2017). *Simple online and realtime
  tracking with a deep association metric*. **ICIP**, pp. 3645–3649.
- \[Zhang2022\] Zhang, Y.; Sun, P.; Jiang, Y.; Yu, D.; Weng, F.; Yuan, Z.; Luo, P.; Liu, W.;
  Wang, X. (2022). *ByteTrack: Multi-object tracking by associating every detection box*.
  **ECCV**, LNCS 13682: 1–21.
- \[Aharon2022\] Aharon, N.; Orfaig, R.; Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust
  associations multi-pedestrian tracking*. arXiv:2206.14651.
- \[Cao2023\] Cao, J.; Pang, J.; Weng, X.; Khirodkar, R.; Kitani, K. (2023). *Observation-
  centric SORT: Rethinking SORT for robust multi-object tracking*. **CVPR**, pp. 9686–9696.
- \[Carion2020\] Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.;
  Zagoruyko, S. (2020). *End-to-end object detection with transformers*. **ECCV**, LNCS
  12346: 213–229.
- \[Cheng2021\] Cheng, B.; Schwing, A.; Kirillov, A. (2021). *Per-pixel classification is
  not all you need for semantic segmentation*. **NeurIPS 34**: 17864–17875.

BibTeX entries are in [`references.bib`](/references.bib).
