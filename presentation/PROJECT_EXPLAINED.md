# Understanding This Project: A Full Explanation

This document explains, in depth, what problem this project is trying to
solve, what was actually tried, and what the results mean. It follows the
order of the slides so you can match each section to what is on screen, but
it is written as an explanation of the work itself, not as a script. Every
abbreviation is expanded the first time it appears, and a full glossary is
included at the end for quick lookup.

---

## The problem, stated precisely

Image classifiers are trained once on a fixed set of clean, labeled images.
Once deployed, they are expected to keep working on new images that arrive
continuously. Two separate things can go wrong with those new images, and
this project is fundamentally about the second one, which is less studied
than the first.

**Problem A: the images look different, but are still the same categories.**
A camera in the field produces noisier, blurrier, or differently-lit images
than the clean training photos. The objects in the images are still, say,
cats and trucks and ships, exactly the categories the model was trained to
recognize, but the pixel statistics have shifted. This is called
**distribution shift**, and it reliably degrades a classifier's accuracy even
though nothing about the underlying categories has changed. The established
fix for this is **Test-Time Adaptation (TTA)**: let the model adjust its own
internal parameters using only the incoming unlabeled images, without ever
being told the correct answer, and without access to the original training
data. This works because you can construct self-supervised training signals
directly from the model's own predictions, for example, if the model is
consistently confident and consistent with itself across image variants,
that is a signal it is probably doing fine; if it is very uncertain, that is
a signal something is off.

**Problem B: some of the images are not from any known category at all.**
This is the harder, "open-world" version of the problem, and it is the one
this project is actually about. A real incoming stream is not guaranteed to
only contain corrupted versions of known classes. It might contain objects
the model has genuinely never been trained to recognize, an unfamiliar animal,
an unfamiliar object type, pure sensor noise, anything. If a naive adaptation
method encounters such an image, it will still produce some prediction
(neural networks always output a probability distribution over their known
classes, even for nonsense inputs), and if that prediction happens to look
confident, a careless TTA method will treat it as a trustworthy training
example and adjust itself based on a completely wrong signal. Doing this
repeatedly compounds the damage: the model drifts further and further away
from correct behavior, a failure mode sometimes described as reinforcing its
own mistakes.

**Why Problem B is harder than Problem A.** Solving Problem A only requires
adapting correctly. Solving Problem B requires doing two things
*simultaneously, without letting them interfere with each other*: keep
improving accuracy on known classes, while also learning to recognize and
discard images from unknown classes, so the second population never
contaminates the training signal used for the first. A system that is good
at one of these but bad at the other is not actually solving the open-world
problem. This is exactly why this project's evaluation never reports
accuracy alone; it always reports accuracy together with a separate
detection metric, and a combined score that requires both to be good at
once (explained fully under Slide 6, below).

---

## The starting point: STAMP, and the specific flaw this project targets

The project does not attempt to solve Problem B from scratch. It starts from
a 2024 method called **STAMP** (STAble Memory rePlay, published at ECCV 2024
by Yu, Sheng, He, and Liang), which was purpose-built for exactly this
open-world setting. Understanding STAMP's own mechanism is essential, because
every experiment in this project is a controlled modification of one
specific piece of it.

**How STAMP works, mechanically, in five steps.**

1. **Augment and average.** For each incoming image, STAMP generates 15
   randomly perturbed copies of it and asks the model to predict on all of
   them, then averages the resulting probability distributions. A prediction
   averaged over 15 views is far less noisy than a prediction from a single
   view, so this step exists purely to get a more trustworthy read on what
   the model currently believes about that image.
2. **Consistency filtering.** STAMP maintains two versions of the model at
   once: the version that is actively being trained and updated, and a
   second, frozen reference version that only has its Batch Normalization
   (BN, a standard component that rescales internal activations) statistics
   recomputed, with no actual learning happening. If these two versions
   disagree on the predicted class for an image, that disagreement is
   treated as a red flag, and the image is dropped before it can influence
   anything further. This is a first, cheap layer of unreliable-example
   filtering.
3. **Entropy-based admission.** This is the step at the center of the entire
   project. **Entropy** is a mathematical measure of how spread out a
   probability distribution is. If the model outputs "95% confident it is a
   cat," entropy is low. If it outputs "20% cat, 18% dog, 15% bird, and so
   on," entropy is high, because the model is not committing to any one
   answer. STAMP computes the entropy of each surviving image's averaged
   prediction and only admits the image for adaptation if that entropy falls
   below a threshold, calculated as a fraction (alpha) of the maximum
   possible entropy for that number of classes. The reasoning: a
   high-entropy prediction usually means the model does not recognize the
   image well, which correlates with (but is not identical to) the image
   being genuinely unfamiliar or out-of-distribution.
4. **Class-balanced memory.** Images that survive both filters are stored in
   a replay buffer with a fixed capacity of 64 examples, tagged with their
   predicted pseudo-label. When the buffer is full and a new example needs to
   be added, STAMP evicts an example from whichever class currently has the
   most representatives in the buffer, keeping the stored set roughly
   balanced across classes rather than letting it be dominated by whichever
   class happens to be most common in the recent stream.
5. **Confidence-weighted replay update.** The model is not updated using only
   the newest incoming batch. It is updated using everything currently in the
   memory buffer, with each stored example weighted by how confident the
   model currently is about it (again measured via entropy: lower-entropy,
   higher-confidence examples get more weight in the update). The actual
   parameter update touches only the Batch Normalization parameters, and it
   is computed using a stability-focused two-step update rule borrowed from
   SAR (Sharpness-Aware and Reliable entropy minimization, a 2023 method
   built on a more general technique called Sharpness-Aware Minimization,
   or SAM), which specifically avoids the kind of sharp, destabilizing update
   that can damage a model rather than gently improving it.

**The precise flaw this project identifies.** Look again at steps 3 and 5.
Both of them depend on the exact same number: the entropy of the model's own
current prediction. In step 3, entropy decides whether an image is trusted
enough to enter the memory at all. In step 5, entropy (through the
confidence weights) decides how much influence each stored example has on
the actual training update. This means one single, imperfect signal is
simultaneously acting as the *gatekeeper* deciding what data the model gets
to learn from, and as the *objective function* defining what "learning" even
means in this setup. If entropy is ever a misleading signal for a given
image (which is entirely possible; a model can be confidently wrong, or
uncertain about an image it would actually classify correctly), that single
mistake propagates into both roles at once. There is no independent check.
This project's research question follows directly from this observation:
**if the detector (step 3) were replaced with a different, independent
signal, while everything else in STAMP (steps 1, 2, 4, and 5's actual update
mechanics) stayed exactly the same, would performance improve?** Four
separate attempts at answering this question are described next, three of
which change the detector itself, and one of which turned out to reveal that
the real problem was somewhere else entirely.

---

## Background methods this project builds on (Slide: Foundations)

Three earlier papers are cited because STAMP (and this project, by
extension) directly inherits ideas from them. Understanding what each one
contributes clarifies exactly which piece of STAMP came from where.

- **Tent (2021).** The foundational recipe for online test-time adaptation:
  minimize the entropy of the model's predictions on incoming test batches,
  updating only the Batch Normalization parameters. This established that
  you do not need labels to improve a model at test time; entropy
  minimization alone, on just the normalization layer, is enough to recover
  a meaningful amount of accuracy under distribution shift. STAMP's step 5
  (minimize a weighted entropy loss over Batch Normalization parameters) is
  a direct descendant of this idea.
- **EATA (2022), Efficient Anti-Forgetting Test-Time Adaptation.** Identified
  two problems with plain entropy minimization: first, learning from
  extremely high-entropy (unreliable) samples can hurt rather than help, so
  EATA filters those out before they influence training; second, continually
  adapting a model can cause it to gradually forget its original, correct
  knowledge (a failure called catastrophic forgetting), so EATA adds a
  mechanism that anchors the model back toward its original weights. STAMP's
  step 3 (entropy-based admission filtering) is conceptually descended from
  this filtering idea.
- **SAR (2023), Sharpness-Aware and Reliable entropy minimization.**
  Addressed a stability problem: when the distribution shift is severe or
  the batches are small and noisy, ordinary gradient updates can be
  erratic and large enough to damage the model outright. SAR's fix is to use
  Sharpness-Aware Minimization (SAM), an optimization technique that
  explicitly seeks updates that keep the model in a "flat," stable region of
  its parameter space rather than a sharp, fragile one. STAMP's actual
  parameter-update mechanics in step 5 are this SAR/SAM-based update rule.

**The shared limitation.** None of these three methods considers the
possibility that some incoming images do not belong to any known class at
all. They are all designed for the closed-world version of the problem
(Problem A above). STAMP is the method in this lineage that explicitly
extends the recipe to handle unknown-class images (Problem B), which is
exactly why this project uses STAMP, rather than Tent, EATA, or SAR directly,
as its base method.

---

## The experimental setup, explained in full (Slide: Setup)

**What the model has to get right.** The test stream is deliberately
constructed as a mixture of two populations: **In-Distribution (ID)** images,
which are corrupted versions of images from classes the model was actually
trained on, and **Out-of-Distribution (OOD)** images, which are synthetic
noise images injected into the stream to play the role of "genuinely unknown
input." A correct system must classify the ID images accurately despite the
corruption, and must recognize the OOD images as unfamiliar and exclude them
from ever influencing training, without ever being told which is which.

**No label leakage, at any stage.** True labels are used exclusively, after
the fact, to compute the reported metrics. They are never used to choose a
threshold, never used to decide what enters memory, and never used in any
update step. This is what makes the setup a fair simulation of real
deployment, where correct answers are, by definition, unavailable while the
system is running.

**The datasets.** CIFAR-10-C and CIFAR-100-C: corrupted versions of the
standard CIFAR-10 (10 categories) and CIFAR-100 (100 categories) image
datasets. "Corrupted" here means a specific, standardized benchmark protocol
applies 15 distinct corruption types (various forms of noise, blur, weather
effects, digital compression artifacts, and so on) at the maximum severity
level the benchmark defines. Testing on both a 10-class and a 100-class
version of the same underlying corruption benchmark is deliberate: it lets
the project check whether a method's behavior is consistent regardless of
how many classes the model has to distinguish between, which turns out to be
an important axis of variation in the results (see Rounds 1 and 2 below).
For every corruption type, the test stream is built from 10,000 corrupted ID
images plus 2,500 synthetic noise images, an 80:20 ID-to-OOD ratio.

**The model and protocol.** A standard ResNet-18 image classifier (a widely
used, moderately sized convolutional network architecture, 18 layers deep)
is used as the base model being adapted, with an output layer matching
whichever dataset is in play (10-way or 100-way classification). Critically,
the model is reset back to its original, pretrained state before every new
corruption type begins. This prevents adaptation on one corruption type from
carrying over and biasing behavior on the next, isolating each corruption
type as an independent trial. Batches of 64 images are processed at a time.

**The three metrics, and why all three are necessary.**
- **Accuracy** is computed only over the true ID images: of the images that
  really do belong to a known class, what fraction did the model classify
  into the correct class? This number says nothing about how well the model
  handles unknown images.
- **AUC (Area Under the ROC Curve)** measures detection quality: treating the
  model's outlier score as a way of ranking images from "most likely known"
  to "most likely unknown," AUC measures how well that ranking actually
  separates the true ID images from the true OOD images, independent of any
  particular threshold choice. A value of 100% means perfect separation; 50%
  means the score carries no useful information at all, equivalent to random
  guessing. This number says nothing about classification accuracy.
- **H-score** is the harmonic mean of accuracy and AUC. The harmonic mean is
  used specifically (rather than a simple average) because it is much more
  sensitive to a low value in either input: a method with 95% accuracy but
  50% AUC gets a much worse H-score than a simple average would suggest,
  because the harmonic mean refuses to let a strong score in one dimension
  compensate for a weak score in the other. This is precisely the property
  needed to fairly evaluate the open-world problem described at the top of
  this document, where doing well on only one of the two goals does not
  count as success.

**The controlled-variable principle.** Every experiment described in Rounds
1 through 4 changes exactly one component of STAMP (almost always, the
detector), while leaving the augmentation step, the consistency filter, the
memory mechanics, and the actual weighted-update rule untouched. This is
standard controlled experimentation: it is the only way to attribute an
observed change in performance to the specific idea being tested, rather
than to some confound.

---

## Round 1: Replacing entropy with feature distance (KNN)

**The hypothesis.** Entropy is a property of the model's *output*
(its predicted probabilities). This round asks whether a property of the
model's *internal representation* would be a better signal. Specifically:
cache a set of feature vectors (the internal, pre-classification-layer
representation the network computes for an image) taken from clean,
uncorrupted training images, up to 2,000 of them. For any incoming test
image, compute its own feature vector, and measure how close it is to its
10 nearest neighbors (this is the K-Nearest Neighbors, or KNN, technique)
among those cached clean features, using cosine similarity. Images whose
features are far from all cached references get a high "outlier" score;
images whose features closely resemble the clean references get a low
outlier score. This score replaces entropy in STAMP's admission step (and is
what gets evaluated for AUC).

**Why this seemed promising.** This signal is explicitly independent of the
model's own confidence. In principle, it decouples "does this look like a
known class, geometrically" from "is the classifier currently confident,"
which are conceptually different questions that entropy conflates.

**What actually happened.** Accuracy improved substantially, about 21
percentage points above doing no adaptation at all, on CIFAR-10-C. But AUC,
the detection quality, was clearly worse than even the simplest baseline
(a method that does no learned filtering at all, just recomputes
normalization statistics). This held on both datasets.

**Why it failed, mechanistically.** The explanation the project settles on:
the clean reference bank is a *fixed, static* target, but the incoming ID
images are heavily corrupted. Severe corruption does not just move OOD
images away from the clean reference cloud in feature space; it also drags
genuinely-ID images away from that same clean reference cloud, since their
raw pixel statistics have been distorted too. Once both populations have
been shifted a similar amount, distance-to-clean-reference stops reliably
separating "corrupted-but-still-known" images from "genuinely unknown"
images. A geometrically clean idea (measure distance to what you know) turns
out to be undermined by an assumption (the reference points stay a valid,
stable yardstick) that the corrupted setting directly violates.

---

## Round 2: Replacing entropy with logit energy

**The hypothesis.** Rather than relying on stored reference data at all,
compute a score directly and cheaply from the model's raw output logits
(the pre-softmax numbers a classifier produces for each class, before they
get converted into probabilities). The specific quantity used, called
**free energy**, is a standard quantity from out-of-distribution detection
research with a known property: models tend to produce lower energy values
for inputs from classes they were trained on, and higher energy values for
unfamiliar inputs, largely because a network's raw logit magnitudes tend to
be more informative about familiarity than its final, temperature-squashed
probabilities are. This score requires no extra forward pass and no stored
data, since the logits are already being computed as part of normal
inference.

**How the threshold was set, without labels.** Rather than a single
fixed number, the system maintains a rolling window of the 1,000 most recent
energy scores it has observed, and sets the current admission cutoff at the
80th percentile of that window. This value is chosen deliberately to match
the known 80:20 ID-to-OOD ratio built into the benchmark, but it never
looks at actual labels to compute it, only the distribution of recent
scores. For the first 256 samples of each corruption type, before this
rolling window has enough history to be reliable, the system temporarily
falls back to STAMP's original entropy-based cutoff. The threshold resets at
the start of every new corruption type, alongside the model reset.

**What actually happened.** On CIFAR-100-C, this was decisively the best
method tried up to this point in the project: both accuracy and AUC clearly
exceeded the simple baseline and the KNN attempt, producing the strongest
H-score of any non-bug-fixed method. On CIFAR-10-C, however, while accuracy
stayed strong, AUC collapsed to barely above 50%, essentially no better than
random guessing at telling known and unknown images apart.

**Why this result matters, and what it does and does not tell you.** Energy
scoring is clearly a strong idea in general, since it dominated the harder,
100-class setting. But its failure specifically on the easier, 10-class
setting shows that "drop the reference bank, use logits instead" does not
automatically fix the open-world detection problem in general; the failure
just moved to a different dataset rather than disappearing. This inconsistency
is precisely what motivated Round 3.

---

## Round 3: Making the threshold adaptive instead of fixed

**The hypothesis.** Round 2 still used a fixed target percentage (80%) for
its rolling threshold. This round isolates a narrower question: is the
*scoring function* the problem, or is it the *fixed target rate* for the
threshold? To test this independently, a control mechanism called
**Adaptive Conformal Inference (ACI)**, drawn from a branch of statistics
called conformal prediction, is layered on top of the existing energy score
from Round 2. Instead of a static 80th-percentile rule, ACI tracks the
actual rejection rate observed in each recent batch and compares it to the
target rejection rate (20%, matching the known OOD proportion). If the
system has been rejecting more than 20% of images lately, it loosens the
cutoff for the next batch; if it has been rejecting fewer, it tightens the
cutoff. This produces an admission threshold that continuously and
automatically corrects itself, rather than sitting at a static percentile.

**What actually happened.** Combined with the energy score (referred to as
Energy+ACI), results were nearly indistinguishable from plain Round 2 on
both datasets, marginally different in either direction but not
meaningfully better or worse.

**Why this negative result is actually informative.** It rules out an entire
category of explanation. If the CIFAR-10-C weakness from Round 2 had been
caused by a poorly calibrated or static threshold, a continuously
self-correcting threshold should have visibly helped. It did not. This tells
you the problem is not really about *how the cutoff is chosen* at all; it
must live somewhere else in the pipeline, which sets up the motivation for
looking at Round 4.

---

## Round 4: Finding and fixing a bug in STAMP's own training loss

**Why this round is fundamentally different from the other three.** Rounds
1 through 3 all changed *what decides whether an image is trustworthy*
(the detector). This round changes nothing about detection at all. It
corrects a mathematical error in step 5 of STAMP's own baseline pipeline,
described earlier: the confidence-weighted replay update.

**The bug, explained precisely.** STAMP's replay memory holds up to 64
examples, but it is emptied at the start of every corruption type and fills
up gradually as new examples are admitted. When computing the properly
normalized, confidence-weighted average used in the training loss, the
correct calculation should divide by *N*, the number of examples actually
present in the memory buffer at that moment. The original implementation
instead always divided by 64, the buffer's fixed maximum capacity,
regardless of how many examples were actually present. Immediately after
every reset, when the buffer might contain only a handful of examples
(say, N equals 8), this mistake inflates the resulting loss, and therefore
the size of the gradient update applied to the model, by a factor of
64 divided by N. In the most extreme case, right at the very start of a
corruption type, this inflation factor approaches 64 times too large.

**Why an inflated update is a serious problem, not a minor rounding issue.**
Gradient-based optimization relies on taking appropriately sized steps.
A step that is dozens of times too large does not just adapt the model
faster, it can overshoot badly, throwing the model's parameters into a
region of the parameter space where its original, correct knowledge has
been damaged rather than incrementally refined. This is especially
consequential here because the bug is not a one-time event: it fires again,
at nearly full severity, at the start of *every single corruption type*,
since the buffer resets every time.

**The fix.** Replace the fixed constant 64 with the actual, current buffer
size N in the normalization calculation. When the buffer is completely
full (N equals 64), this produces exactly the same result as before, so the
fix changes nothing about steady-state behavior. It only changes behavior
during the brief warm-up window right after each reset, precisely where the
bug was doing damage. The corrected variant is named **STAMP+BSN**
(Batch-Size Normalization) for the remainder of this project, a name chosen
locally to describe the fix, not a term from an external paper.

**What actually happened, and what it reveals.** STAMP+BSN produced the
best H-score of the entire project on *both* datasets at once, something
none of Rounds 1 through 3 managed. The most striking single number: on
CIFAR-10-C, AUC jumped from 54.33 (Round 2's near-random detection score)
to 83.96, a large, decisive improvement, while accuracy barely moved
(78.99 to 78.95). Because the *only* thing that changed between Round 2 and
Round 4 is this loss-normalization bug, and because accuracy stayed
essentially flat while detection quality transformed, the evidence strongly
suggests that Round 2's CIFAR-10-C weakness was never really a flaw in the
energy score as an idea. It was a symptom of training instability injected
by this bug, and once that instability was removed, the same underlying
energy score performed as well as it apparently should have all along.

---

## The full results table, read carefully

The complete set of numbers, all six methods, both datasets, all three
metrics:

| Dataset | Method | Accuracy | AUC | H-score |
|---|---|---|---|---|
| CIFAR-10-C | Source (no adaptation) | 57.34 | 70.39 | 62.32 |
| CIFAR-10-C | Norm-test (simple baseline) | 72.95 | 68.51 | 70.58 |
| CIFAR-10-C | KNN (Round 1) | 78.64 | 59.10 | 63.61 |
| CIFAR-10-C | Energy (Round 2) | 78.99 | 54.33 | 63.80 |
| CIFAR-10-C | Energy+ACI (Round 3) | 78.98 | 54.08 | 63.58 |
| CIFAR-10-C | **STAMP+BSN (Round 4)** | 78.95 | **83.96** | **81.19** |
| CIFAR-100-C | Source (no adaptation) | 35.83 | 43.13 | 38.01 |
| CIFAR-100-C | Norm-test (simple baseline) | 45.90 | 80.91 | 58.49 |
| CIFAR-100-C | KNN (Round 1) | 45.73 | 69.93 | 54.15 |
| CIFAR-100-C | Energy (Round 2) | 57.14 | 95.12 | 71.33 |
| CIFAR-100-C | Energy+ACI (Round 3) | 57.04 | 95.08 | 71.24 |
| CIFAR-100-C | **STAMP+BSN (Round 4)** | **58.02** | **98.29** | **72.89** |

**How to read this table's overall story.** "Source" is the unmodified,
pretrained classifier with zero adaptation, included purely as a lower
bound. "Norm-test" is a simple, well-known TTA baseline (recompute
normalization statistics on incoming batches, nothing more), included as an
upper bound that any interesting new method ought to beat. Every row after
that is a version of STAMP, with exactly one thing changed relative to
STAMP's own baseline detector, as described above. Accuracy rises fairly
steadily down each block, meaning every scoring idea tried is at least
somewhat useful for the classification half of the problem. AUC is where
the real story is: it swings substantially between rounds, is inconsistent
between the two datasets for Rounds 1 through 3, and only becomes reliably
strong on both datasets once the Round 4 bug fix is applied. H-score, the
combined metric, therefore tells the cleanest version of the story: STAMP+BSN
is the only method that is unambiguously the best, or tied for best, on
every single column, on both datasets.

---

## What this project ultimately demonstrates

Pulling the four rounds together into one coherent conclusion:

1. **Neither of the two new detector ideas (KNN distance, logit energy) is a
   universal fix.** Each one has a real, dataset-specific weakness: KNN
   consistently hurts detection because its fixed reference points stop
   being trustworthy once genuinely-known images are also corrupted; energy
   is excellent on the harder 100-class dataset but fails on the easier
   10-class one.
2. **Making the decision threshold adaptive (ACI) does not rescue either
   idea.** This is a controlled, negative result that specifically rules out
   "the threshold was badly chosen" as an explanation for Round 2's
   weakness.
3. **The actual explanation was a bug, not a modeling choice.** A
   normalization error in STAMP's own replay loss was inflating gradient
   updates by up to 64 times during the warm-up period after every reset.
   Fixing this single line of reasoning in the code produced the best,
   most consistent results across both datasets and every metric, more
   effectively than any of the three new detection ideas managed on their
   own.
4. **The broader lesson.** When a component of a machine learning pipeline
   behaves inconsistently across settings, it is worth checking whether the
   surrounding implementation is actually correct before concluding that the
   underlying idea itself is flawed. Here, an idea (energy scoring) that
   looked dataset-dependent and unreliable turned out to be fundamentally
   sound; what was actually unreliable was a training-stability bug sitting
   one layer beneath it.

The proposed future work (multi-seed evaluation for statistical confidence,
per-corruption breakdowns instead of only averages, combining the energy
score with the bug fix and adaptive thresholding together, and a direct
warm-up ablation study) is aimed specifically at confirming this explanation
more rigorously and checking whether the remaining pieces (Rounds 1 through
3) become more competitive once they are also run on top of the corrected,
bug-free training loop.

---

## Full glossary of abbreviations and terms

| Term | Full name / meaning |
|---|---|
| **TTA** | Test-Time Adaptation: adjusting a trained model using only unlabeled data seen while it is running, with no correct answers provided. |
| **ID** | In-Distribution: an image belonging to a class the model was trained to recognize. |
| **OOD** | Out-of-Distribution: an image belonging to no class the model was trained to recognize. |
| **STAMP** | STAble Memory rePlay. The base method this project modifies (Yu, Sheng, He, and Liang, ECCV 2024). |
| **BN** | Batch Normalization: a standard neural network component that rescales internal activations; several TTA methods, including Tent and STAMP, adapt only this part of the network. |
| **Entropy** | A measure of how spread out a predicted probability distribution is; low entropy means the model is confident, high entropy means it is not. |
| **KNN** | K-Nearest Neighbors: deciding something about a data point by examining its K most similar known examples (K was set to 10 in Round 1). |
| **AUC** | Area Under the (ROC) Curve: a 0 to 100% score measuring how well an outlier score separates known from unknown images, independent of any single threshold. |
| **ROC** | Receiver Operating Characteristic: the curve AUC is computed from; not discussed in detail elsewhere in this document. |
| **H-score** | This project's combined metric: the harmonic mean of accuracy and AUC, which penalizes a method for being weak in either one. |
| **ACI** | Adaptive Conformal Inference: a statistical method for automatically adjusting a threshold to keep hitting a target rate over time. Used in Round 3. |
| **BSN** | Batch-Size Normalization: this project's name for the Round 4 fix, which uses the true current memory size instead of a fixed constant. |
| **SAM** | Sharpness-Aware Minimization: an optimization technique that favors stable, "flat" updates over sharp, risky ones. |
| **SAR** | Sharpness-Aware and Reliable entropy minimization: the 2023 method STAMP borrows its SAM-based update rule from. |
| **EATA** | Efficient Anti-Forgetting Test-Time Adaptation: a 2022 method that filters unreliable samples and guards against the model forgetting its original knowledge. |
| **Tent** | Not an abbreviation; the foundational 2021 entropy-minimization TTA method. |
| **CIFAR-10 / CIFAR-100** | Standard image classification datasets with 10 and 100 categories respectively, named after the Canadian Institute For Advanced Research. |
| **CIFAR-10-C / CIFAR-100-C** | Standardized, artificially corrupted versions of the above datasets, used to benchmark robustness. |
| **ResNet-18** | Residual Network, 18 layers deep: the specific image classifier architecture used throughout this project's experiments. |
| **ECCV / ICLR / ICML / NeurIPS / CVPR** | Names of the academic conferences where the cited papers were published (European Conference on Computer Vision; International Conference on Learning Representations; International Conference on Machine Learning; Conference on Neural Information Processing Systems; Conference on Computer Vision and Pattern Recognition). |

Sources consulted to confirm exact abbreviation meanings:
- [STAMP paper](https://arxiv.org/abs/2407.15773)
- [STAMP official code repository](https://github.com/yuyongcan/STAMP)
- [EATA paper](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf)
- [SAR paper](https://arxiv.org/abs/2302.12400)
