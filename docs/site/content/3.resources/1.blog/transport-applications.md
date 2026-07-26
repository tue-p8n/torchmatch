---
title: Transport applications
description: Historical and modern applications of optimal transport — from Monge's earth-moving problem through Wasserstein GANs, single-cell genomics, domain adaptation, and geometric deep learning.
date: 2026-07-07
tags: [transport, history]
---

Optimal transport has one of the longest application histories in applied mathematics — 240
years from Monge's engineering problem to today's generative model training loops. This page
traces where the problem arises, why OT is the right formulation, and how torchmatch's
backends map onto each use case.

## Supply chain and resource allocation

The first computational use of the transportation problem (the discrete precursor to general
OT) was in **centrally-planned resource allocation**: given factories producing a good and
warehouses needing it, at what shipping cost should supply be routed to minimise total
freight? Kantorovich developed the LP formulation for exactly this purpose in wartime Soviet
planning, and the 1975 Nobel Prize citation mentions "optimal allocation of resources" as
the core contribution. Koopmans, sharing the prize, applied the same structure to shipping
route optimisation.

The transportation LP — assign continuous mass from supply locations to demand locations at
minimum unit transport cost — is the special case where `a` and `b` are given histograms
and `C` encodes geographic distance. It remains a daily tool in logistics optimisation and
supply chain planning.

## Image retrieval and perceptual similarity

The Earth Mover's Distance entered computer vision through image retrieval
\[Rubner2000\]. When comparing two images by their colour histograms, Euclidean distance
between bin counts is insensitive to perceptually small shifts: moving probability mass from
the "red" bin to the "orange" bin registers as a large L2 distance but a small perceptual
distance. OT charges the actual cost of moving mass between bins according to the
colour-space ground metric (a distance function defined over the colour space, e.g. Euclidean distance in RGB or Lab), so nearby colours are treated as similar.

The same idea extends to texture descriptors, shape histograms, and any feature whose
natural distance is not Euclidean on the raw representation. The `EXACT_EMD` backend in
torchmatch computes the exact earth mover's distance for small histograms where no
regularisation is desired.

## Colour transfer and style matching

Histogram matching via OT underlies a family of image stylisation techniques. Given a
source image and a target whose colour statistics you want to adopt, computing the 1D or 3D
OT map between their colour histograms and applying it as a pixel-wise transformation
transfers the palette of the target to the source with minimal perceptual distortion.
Pitié, Kokaram, and Dahyot \[Pitie2007\] formalised this as iterated 1D projections
(sliced Wasserstein transport — an approximation that computes 1D OT on random projections of the distribution, avoiding the full N-dimensional cost), an approximation that scales to full 3D colour histograms
without materialising an `n³` cost.

## Generative modelling

**Wasserstein GAN** \[Arjovsky2017\] reframed generative adversarial training in terms of
the Wasserstein-1 distance between the real and generated distributions. The Wasserstein
distance is meaningful even when the two distributions have disjoint support (which
commonly occurs during early GAN training), whereas Jensen-Shannon divergence and KL
divergence both saturate to a constant in this regime, providing no gradient signal. The
practical implementation uses the Kantorovich-Rubinstein dual (the equivalent formulation of W1 as a supremum over 1-Lipschitz functions, used to train the discriminator) with gradient-penalised
discriminators rather than directly computing the OT plan, but the connection motivates the
loss design.

**Wasserstein autoencoders** (WAE) \[Tolstikhin2018\] replaced the evidence lower bound
of the VAE with a Wasserstein distance between the aggregate posterior and the prior,
producing sharper reconstructions on image benchmarks.

**Flow matching** \[Lipman2022; Liu2022\] frames diffusion-model training as learning a
vector field that transports a source distribution (Gaussian noise) to a target
distribution (data). The "OT-conditioned flow matching" variant conditions the flow on the
OT displacement plan between individual noise samples and data samples, producing straighter
trajectories and faster inference.

For all three model classes, `transport.matrix.solve` with `SINKHORN_DIVERGENCE` or
`transport.samples.loss` with `debias=True` provides a differentiable Wasserstein-like
training loss that can be substituted for or combined with the standard objectives.

## Domain adaptation

A model trained on a labelled source domain often fails on an unlabelled target domain
because the marginal feature distributions differ. OT provides a principled way to
**measure and correct this discrepancy**: it computes a transport plan between source and target feature distributions, then uses that plan to move source features toward the target domain before training a classifier.

- **OTDA** \[Courty2017\] computes the regularised OT plan between labelled source samples
  and unlabelled target samples in the feature space of a pretrained network. The plan then
  transports source features toward the target, producing pseudo-labelled target
  samples that train a target-domain classifier.
- **DeepJDOT** \[Damodaran2018\] integrates OT alignment into an end-to-end deep network:
  the OT plan between source and target minibatches is recomputed each iteration and used
  as a re-weighting of the classification loss.
- **Distribution matching for dataset distillation**: the Sinkhorn divergence between
  feature distributions of real and distilled datasets serves as the distillation
  objective, requiring a geometry-aware distance that respects the ground metric of feature
  space.

The `UNBALANCED_SINKHORN` backend is particularly useful when source and target have
genuinely different class distributions: relaxing the marginal constraints prevents the
coupling from being dominated by classes that are abundant in one domain but rare in the
other.

## Geometric deep learning and 3D vision

**Point cloud registration and shape matching** require finding correspondences between two
unordered point sets. OT provides the soft coupling that minimises transport cost, giving
a probabilistic correspondence matrix as the plan. This soft matching initialises or
replaces ICP (iterative closest point) in registration pipelines, is differentiable with
respect to the point positions, and scales to large point clouds via the `samples.loss`
streaming kernel.

**Wasserstein barycenters** \[Agueh2011\] — weighted averages of distributions in Wasserstein
space — produce interpolations between shapes that respect the geometry of the ground
metric. Interpolating between two 3D shapes in Wasserstein space moves each point smoothly
toward its corresponding point in the target, unlike Euclidean averaging which collapses
the shape when distributions are disjoint.

**Shape completion and generation**: point-cloud generative models (PointFlow,
ShapeGF, DPM-based point clouds) use Wasserstein or Sinkhorn losses to supervise the
generated shape. `transport.samples.loss` is directly applicable; the streaming Triton
kernel avoids the `N × M` allocation that would dominate memory at generation-scale cloud
sizes.

## Natural language processing

**Word Mover's Distance** \[Kusner2015\] embeds documents as distributions over word
embedding vectors (weighted by TF-IDF or uniform) and computes the OT cost between them
using pretrained embeddings as the ground metric. The resulting distance is insensitive to
synonyms and paraphrases — moving mass from "automobile" to "car" is cheap because the
embeddings are nearby — and outperforms bag-of-words similarity on several retrieval
benchmarks.

The same idea applies to **sentence-level alignment** for mining parallel corpora, where
OT between sentence embedding distributions identifies likely translations without requiring
exact string matches.

## Computational biology

**Waddington-OT** \[Schiebinger2019\] modelled cellular differentiation as an OT problem:
single-cell RNA-seq profiles from adjacent time points define the source and target
distributions over gene-expression space, and the OT plan between them gives the most
parsimonious account of which early cells give rise to which later cells. This transport
interpretation respects the Waddington epigenetic landscape metaphor and produces
biologically interpretable developmental trajectories without requiring paired data.

**Moscot** \[Klein2023\] scaled this framework to million-cell datasets and extended it to
spatial transcriptomics alignment, single-cell multi-omics integration (matching cells
across RNA and protein measurements), and lineage tracing. The computational backbone is
log-domain Sinkhorn with online batching — structurally identical to `LOG_SINKHORN` —
applied at dataset scales that require careful memory management.

**Protein structure comparison**: OT between residue-coordinate distributions provides a
rotation-invariant distance between two protein chains that accounts for insertions,
deletions, and loop flexibility without requiring global structural alignment.

## References

- \[Rubner2000\] Rubner, Y.; Tomasi, C.; Guibas, L. J. (2000). *The Earth Mover's Distance
  as a metric for image retrieval*. **International Journal of Computer Vision**, 40(2):
  99–121.
- \[Pitie2007\] Pitié, F.; Kokaram, A. C.; Dahyot, R. (2007). *Automated colour grading
  using colour distribution transfer*. **Computer Vision and Image Understanding**,
  107(1–2): 123–137.
- \[Arjovsky2017\] Arjovsky, M.; Chintala, S.; Bottou, L. (2017). *Wasserstein generative
  adversarial networks*. **ICML**, PMLR 70: 214–223.
- \[Tolstikhin2018\] Tolstikhin, I.; Bousquet, O.; Gelly, S.; Schölkopf, B. (2018).
  *Wasserstein auto-encoders*. **ICLR**.
- \[Lipman2022\] Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; Le, M. (2022).
  *Flow matching for generative modelling*. **ICLR 2023**.
- \[Liu2022\] Liu, X.; Gong, C.; Liu, Q. (2022). *Flow straight and fast: Learning to
  generate and transfer data with rectified flow*. **ICLR 2023**.
- \[Courty2017\] Courty, N.; Flamary, R.; Tuia, D.; Rakotomamonjy, A. (2017). *Optimal
  transport for domain adaptation*. **IEEE TPAMI**, 39(9): 1853–1865.
- \[Damodaran2018\] Damodaran, B. B.; Kellenberger, B.; Flamary, R.; Tuia, D.; Courty, N.
  (2018). *DeepJDOT: Deep joint distribution optimal transport for unsupervised domain
  adaptation*. **ECCV**, LNCS 11208: 467–483.
- \[Agueh2011\] Agueh, M.; Carlier, G. (2011). *Barycenters in the Wasserstein space*.
  **SIAM Journal on Mathematical Analysis**, 43(2): 904–924.
- \[Kusner2015\] Kusner, M. J.; Sun, Y.; Kolkin, N. I.; Weinberger, K. Q. (2015). *From
  word embeddings to document distances*. **ICML**, PMLR 37: 957–966.
- \[Schiebinger2019\] Schiebinger, G.; Shu, J.; Tabaka, M.; Cleary, B.; Subramanian, V.;
  Solomon, A.; Gould, J.; Liu, S.; Lin, S.; Berube, P.; Lee, L.; Chen, J.; Brumbaugh, J.;
  Rigollet, P.; Hochedlinger, K.; Jaenisch, R.; Regev, A.; Lander, E. S. (2019). *Optimal-
  transport analysis of single-cell gene expression identifies developmental trajectories in
  reprogramming*. **Cell**, 176(4): 928–943.
- \[Klein2023\] Klein, D.; Palla, G.; Lange, M.; Klein, M.; Piran, Z.; Gander, M.;
  Meng-Papaxanthos, L.; Sterr, M.; Treutlein, B.; Lickert, H.; Theis, F. J. (2023).
  *Mapping cells through time and space with moscot*. **bioRxiv**.

BibTeX entries are in [`references.bib`](/references.bib).
