# Final Plan: One CNN Each, Run in Parallel Over Break

Written Thu Nov 19, 2026. Final report due **Wed Dec 2**, presentation **Dec 7/9**.
Everything here is sized for a **free Colab T4** and **one week including Thanksgiving**.

---

## UPDATE Nov 26: reduced scope (this section overrides the S-task owners in Section 3 and the calendar in Section 7)

**Roles:** TT = **EfficientNet-B0**. EH = **ResNet-18** (pretrained + the from-scratch GWTNet run). Ethan = **MobileNetV3-Large**.

**The shared code is written** (branch `tt/shared-transfer-pipeline`): split-then-cap and nested class subsets, the `transfer` pipeline for all three backbones, resume after a disconnect, per-run experiment log, top-3/top-10 metrics, optional W&B. You do not need to write any of it. You only run it.

**What a "B run" is.** B1–B6 are our labels for the *baseline sweep*: the same model trained several times where **only one setting changes** from the default (B1), so any difference in validation accuracy can be blamed on that one setting. That is what "tuning hyperparameters" means in the report.

| Run | Change from B1 | What it teaches |
|---|---|---|
| **B1** | none: lr 3e-4, weight decay 0.05, full fine-tune, basic aug, no flip, dropout 0.3, label smoothing 0.1 | The reference everything else is compared to |
| **B2** | `--lr 1e-4` | Is the learning rate too high? |
| **B3** | `--lr 1e-3` | Is it too low? |
| **B4** | `--freeze all --lr 1e-3` | Pretrained features untouched, only the new head learns ("linear probe") |
| **B5** | `--freeze partial` | Early layers (generic features) frozen, later layers learn |
| **B6** | `--aug strong` | Does heavier augmentation reduce overfitting? |
| *B7 (if time)* | `--use-hflip` | Mirroring flips driving side and text; does it hurt? |
| *B8 (if time)* | `--weight-decay 0.5` | Stronger regularization |

**Then one final run** per model: the best of B1–B6 *by validation top-5*, trained on the full training data at 224px for about 10 epochs, with `--eval-test`. That is the only run whose test numbers go in the report. EH also runs one from-scratch GWTNet at the same settings for the "scratch vs. pretrained" row.

**Sweep settings** (all B runs): 111 countries, 300 training images per country, 160px, 8 epochs, batch size 128, `--patience 99` (no early stopping, because the LR schedule needs to finish), no `--eval-test`.

**Run command** (`--backbone` is `efficientnet_b0`, `resnet18` or `mobilenet_v3_large`; run names look like `effb0_B1_default`, `rn18_B3_lr1e-3`, `mnv3_B5_partial`):

```
python scripts/run_pipeline.py --model transfer --backbone efficientnet_b0 \
  --data-dir data/processed/streetview-256 --num-classes 111 --max-train-per-class 300 \
  --crop-size 160 --resize-to 183 --epochs 8 --batch-size 128 --patience 99 --workers 2 \
  --lr 3e-4 --weight-decay 0.05 --dropout 0.3 --label-smoothing 0.1 --no-use-hflip --aug basic --freeze none \
  --weights-dir /content/drive/MyDrive/tcu-cosc-50523-final-project-geoguessrbot/weights \
  --run-name effb0_B1_default
```

Results land in `results/<backbone>/<run_name>.json` and one new row in `results/<backbone>/experiment_log.csv`. All timings in this document are still estimates until someone measures B1.

---

## 0. The plan in one page

**Research question for the report:** *How much does ImageNet pretraining help country-level geolocation from a single street-view image, and does the architecture matter?*

This answers the prof's "explore other models, not just ResNet" and "extensive hyperparameter/architecture experiments". The course guidelines say a smaller model evaluated carefully beats a big model without a clear question.

| Person | Model | Role in the comparison |
|---|---|---|
| **EH** | **ResNet-18**, ImageNet-pretrained (+ our from-scratch GWTNet as reference) | Scratch vs. transfer with an *identical* architecture |
| **TT** | **EfficientNet-B0**, ImageNet-pretrained | Different family: depthwise convs, compound scaling |
| **EW** | **MobileNetV3-Large**, ImageNet-pretrained | Lightest model: speed/accuracy tradeoff (replaces the heavy ConvNeXt) |

Each person runs the **same protocol** on their own model on their own Colab account, so the three results are directly comparable and we get 3× the GPU quota:

- **10 hyperparameter runs** (Section 5), one factor changed at a time
- **3 class-scaling runs** at 10 / 20 / 50 classes (the 111-class run is already one of the 10)
- **2 extra seeds** of the best config (error bars)
- **1 final full-data run** at 224px, with the test set evaluated once
- **Grad-CAM, a failure gallery, and an architecture diagram** for their own model

That is about 16 cheap runs plus one final run per person, and at least one logged experiment per person per day.

**Why this is feasible now:** the midterm numbers (~10–13 min/epoch, 12–15 h for 70 epochs) come from decoding 640×640 JPEGs on Colab's 2 CPUs, not from the GPU. Pre-resizing every image to 256px once (Section 3, S1) should cut epoch time roughly 5× and, with pretrained models needing only 8–15 epochs, makes every run in this plan a matter of minutes to about an hour. *All timings in this document are estimates until Saturday's measured run.*

---

## 1. Where we are

| Item | Status |
|---|---|
| Logistic regression baseline | Done, but only binary FR/JP (0.73 top-1). The prof said fine to keep, don't spend more time on it |
| GWTNet (`scripts/cnn_1`), ResNet-18-style, from scratch | Implemented. **No saved results in the repo.** Needs one 111-class run |
| Transfer learning | Not started |
| Top-k metrics | Top-1/top-5 computed already. Add top-3/top-10 |
| Experiment tracking | Not started (W&B plan is in the team notes) |
| Figures (input examples, architecture diagrams, plots) | Not started. The prof asked for all of them |

The midterm called our CNN "7×7 conv + 16 3×3 convs". That is exactly ResNet-18 (1 + 16 conv layers + FC), so in the report call it **ResNet-18 (from scratch)**, not "CNN-16".

---

## 2. Final proposed architectures

### 2.1 Shared pipeline (the figure for the main architecture diagram)

```
 640×640 RGB street-view image
        │  (pre-resized once to 256×256 JPEG)
        ▼
 Train: RandomResizedCrop(224 or 160) + ColorJitter (+ optional flip / TrivialAugment)
 Eval : Resize → CenterCrop(224 or 160)
        │  Normalize with ImageNet mean/std (pretrained models)
        ▼
 ┌─────────────────────────────────────────┐
 │ Pretrained CNN backbone (feature maps)  │   ← swapped per person
 └─────────────────────────────────────────┘
        │ global average pool
        ▼
 Dropout(p) ──► Linear(feat_dim → 111)  ──► softmax ──► top-1 / 3 / 5 / 10 country guesses
                (new head, randomly initialised)

 Loss: cross-entropy with label smoothing
 Optimiser: AdamW + OneCycle (warmup then cosine decay), mixed precision (FP16)
```

### 2.2 Model specs (use these for each person's own diagram)

**ResNet-18** (`torchvision.models.resnet18`, ImageNet-1k weights). Residual blocks, skip connections.
```
conv1 7×7/2, 64 → BN → ReLU → maxpool 3×3/2
layer1: 2 × BasicBlock(64)         56×56   (skip: x + F(x))
layer2: 2 × BasicBlock(128)        28×28   (first block stride 2, 1×1 conv shortcut)
layer3: 2 × BasicBlock(256)        14×14
layer4: 2 × BasicBlock(512)         7×7
avgpool → Dropout → Linear(512 → 111)
```
~11.2M params, ~1.8 GFLOPs at 224px. *Partial freeze (B5):* freeze `conv1`, `layer1`, `layer2`; train `layer3`, `layer4`, head.

**EfficientNet-B0** (`torchvision.models.efficientnet_b0`, ImageNet-1k weights). MBConv blocks: 1×1 expand → depthwise conv → squeeze-and-excite → 1×1 project.
```
features[0]: stem conv 3×3/2, 32 → BN → SiLU
features[1]: 1 × MBConv1  k3   16 ch
features[2]: 2 × MBConv6  k3   24 ch
features[3]: 2 × MBConv6  k5   40 ch
features[4]: 3 × MBConv6  k3   80 ch
features[5]: 3 × MBConv6  k5  112 ch
features[6]: 4 × MBConv6  k5  192 ch
features[7]: 1 × MBConv6  k3  320 ch
features[8]: 1×1 conv → 1280 → BN → SiLU
avgpool → Dropout → Linear(1280 → 111)
```
~4M params, ~0.4 GFLOPs. *Partial freeze (B5):* freeze `features[0:5]`; train `features[5:]` and head.

**MobileNetV3-Large** (`torchvision.models.mobilenet_v3_large`, ImageNet weights `IMAGENET1K_V2`). Inverted residual blocks with h-swish and squeeze-and-excite on some blocks.
```
features[0]:     stem conv 3×3/2, 16 → BN → Hardswish
features[1–15]:  15 × inverted-residual bottlenecks (expand 1×1 → depthwise 3×3 or 5×5 → [SE] → project 1×1)
features[16]:    1×1 conv → 960 → BN → Hardswish
avgpool → Linear(960 → 1280) → Hardswish → Dropout → Linear(1280 → 111)
```
~4–5M params, ~0.22 GFLOPs. Note this head has an extra hidden layer, so only replace the final Linear. *Partial freeze (B5):* freeze `features[0:9]`; train `features[9:]` and the classifier.

**GWTNet (ours, from scratch).** Same as ResNet-18 above but with random init and `Dropout(0.3)`. It is TT's reference run and answers "what does pretraining buy us?".

> **Verify before trusting:** parameter counts above are approximate. Print `sum(p.numel() for p in model.parameters())` for the real numbers in the report. Check the exact `torchvision` weight enum names in the installed version.

### 2.3 Brainstorm items: what's in and out

| Idea from the board | Decision |
|---|---|
| ResNet clone → tune from there | **In** (ResNet-18, TT) |
| MobileNet | **In** (MobileNetV3-Large, EW) |
| EfficientNet-B0 | **In** (EH) |
| Transfer learning | **In**, it's the core of the study |
| Architecture diagram for EACH model | **In**, each person draws their own (Section 8) |
| Vary number of classes (10/20/50/100) | **In**, 3 extra runs per person (Section 6) |
| W&B tracking | **In** (Section 4) |
| Checkpoint to Drive every epoch + FP16 | **In**, mandatory (Section 4) |
| Data augmentation | **In** as ablations B8/B9 |
| Grad-CAM, demo | **In** (bonus, Section 8) |
| AlexNet clone | **Stretch.** Only if you finish early. Train from scratch, adds a "pre-ResNet" data point |
| "Traditional" neural net (HW3-style MLP) | **Stretch.** A cheap MLP on 32×32 or 64×64 downsampled pixels is a more credible baseline than 1.2M-feature logistic regression. One run, ~15 min |
| ConvNeXt / transformer | **Out** (too heavy for a T4 in a week) |

---

## 3. Shared setup (Thu–Fri, before anyone runs experiments)

All three of us depend on this code, so **do it once, merge to `main` by Fri night**, and everyone branches from it. Don't each patch the pipeline separately, or the runs won't be comparable.

| # | Task | Owner | Definition of done |
|---|---|---|---|
| **S1** | **Pre-resize script** (`scripts/preprocess_resize.py`): every image → 256×256 JPEG (quality 95), same folder-per-country layout. Assert images are square first. Zip to Drive as `streetview-256.zip` (expected ~2–3 GB, fits a free Drive) | TT | Zip on shared Drive; a fresh Colab can unzip it in a few minutes. Start this **tonight**, it's the long pole (~20–40 min of CPU) |
| **S2** | **Split fixes** in `scripts/common/preprocessing.py`: (a) split first, then cap only the **train** split (`--max-train-per-class 300`). Right now the cap happens *before* the split, so a capped run gets a different, smaller test set and its train images can land in a full run's test set. (b) **Nested class subsets:** shuffle all 111 countries once with seed 42 and take the first N, so the 10 ⊂ 20 ⊂ 50 ⊂ 111 sets are comparable | TT | 10-class set is inside the 20-class set, etc. Same test images regardless of cap |
| **S3** | **`transfer` pipeline** (`scripts/transfer/`, `--model transfer --backbone {resnet18,efficientnet_b0,mobilenet_v3_large}`): torchvision backbone with the new head, `--pretrained/--no-pretrained`, `--freeze {none,partial,all}` (Section 2.2 split points; **keep frozen layers' BatchNorm in eval mode**, otherwise their running stats drift), `--aug {basic,strong}`, `--head-lr-mult`, ImageNet mean/std, and `--eval-test` (default **off**, see rule below). Reuse `cnn_1/trainer.py` | EH | One-epoch smoke test passes on 10 classes for all 3 backbones |
| **S4** | **Resume.** Every epoch, save `last.pt` to Drive (model, optimizer, scheduler, scaler, epoch, history, best weights). On start, resume automatically if `last.pt` exists | EH | Kill a run mid-epoch, rerun the same command, it continues from the last epoch |
| **S5** | **W&B + experiment log.** `wandb.init(project="geoguessr", name=run_name, group=backbone, config=vars(args))`, log per-epoch train loss, val loss, val top-1/top-5, and final metrics. At the end of each run also append one row to `results/<model>/experiment_log.csv` (columns in Section 9) | EW | A run shows up in W&B and adds a CSV row |
| **S6** | **Metrics.** Add top-3/top-10. Add `scripts/analysis.py` to produce: top-k curve (k = 1..10), **top-15 most-confused country pairs**, and a **continent-level confusion matrix** (map country code → continent). The 111×111 matrix is unreadable, so don't plot it, just save it | EW | Runs on a results JSON and writes 3 PNGs |
| **S7** | **Small bug:** the AdamW no-decay filter in `run_pipeline.py` looks for `"bn"` in parameter names, but BN layers here are named like `stem.1.weight`, so BN weights still get weight decay. Use `param.ndim <= 1` | TT | No-decay group contains all BN/bias params |
| **S8** | **Cache the leakage hashes.** `check_leakage` MD5-hashes every image on every run. Hash once, save to a file, reuse | TT | Second run skips hashing |

**Rule: no test-set peeking.** Sweep runs (B1–B10, class-scaling, seeds) select by **validation** only and do **not** evaluate on test. The test set is evaluated for the **final run and the best-config runs you report**. The midterm says the test set is touched once and hyperparameters weren't tuned on it; keep that true.

**Git workflow (we already had notebook merge conflicts):**
- Pipeline code lives in `scripts/` and goes through `main`. Everyone pulls it.
- Each person works on branch `exp/<name>` and commits only `results/<model>/…` (results JSON, `experiment_log.csv`, figures).
- Don't commit notebooks you edited. Keep your own Colab notebook copy.
- Don't commit checkpoints or data (the course says not to).
- Make at least one substantive commit each week (it's a bonus item).

---

## 4. Colab survival guide

Limits from the board: **90 min idle timeout**, **12 h session lifetime**. Free-tier GPU quota isn't guaranteed and can run out. If it does, switch accounts, or try a Kaggle notebook (free T4/P100 with a weekly GPU-hour quota; verify the current limit).

**Every session, in order:**
1. Runtime → GPU (T4). Run `nvidia-smi` to confirm.
2. Mount Drive. Copy `streetview-256.zip` to `/content` (local disk) and **unzip locally**. Never read training images from Drive; it's far too slow.
3. Pull the repo (`git pull`), install `wandb`, `timm` if needed, and `wandb.login()`.
4. Set `--weights-dir` to a folder **on Drive** (`/content/drive/MyDrive/.../weights`) so checkpoints survive a disconnect.
5. Run your command. If the session dies, re-run the **same command** and it resumes (S4).

**Keep the session alive:** keep the browser tab open. Anything running in a cell counts as active, but a finished run sitting idle for 90 min gets disconnected. Chain a day's runs in one cell (`for` loop over commands) so the GPU is never idle.

**Mixed precision (FP16)** is already enabled in the trainer. Leave it on.

**Don't run two things on one T4.** Parallelism comes from having 3 people, not from stacking jobs.

---

## 5. Experiment protocol (identical for all three people)

### 5.1 Fixed settings

| Setting | Value |
|---|---|
| Data | 256px pre-resized images, all **111 countries**, seed 42, **70/15/15 split within each country** |
| Sweep training set | **300 images per country** (~33k images), capped on the **train split only** |
| Sweep resolution | **160px** crop (`--resize-to 183 --crop-size 160`), faster for a sweep |
| Final resolution | **224px** crop (`--resize-to 256 --crop-size 224`) |
| Optimiser / schedule | AdamW, OneCycle (15% warmup, cosine decay), FP16 |
| Epochs | 8 for sweeps (fixed). Keep the **best-val-loss** checkpoint. Patience only matters for the final run (as a safety stop). With a cosine schedule, stopping early cuts off the low-LR phase |
| Metrics | top-1, top-3, top-5, top-10, macro-F1 (val during sweeps) |
| Pretrained-model normalisation | ImageNet mean/std |

### 5.2 Default config (B1)

`lr 3e-4 · weight-decay 0.05 · batch size 128 · dropout 0.3 · label smoothing 0.1 · hflip OFF · aug basic · freeze none · epochs 8 · seed 42`

hflip is off by default because mirroring a street-view photo flips driving side and mirrors text, which are real country cues.

### 5.3 The 10 hyperparameter runs

Change **one thing** from B1 each time. This maps to the "at least 10 hyperparameter sets" item from the board (learning rate, regularization, batch size, augmentation, freezing, etc.).

| Run | Change from B1 | What it tests |
|---|---|---|
| **B1** | none (default) | Reference |
| **B2** | `--lr 1e-4` | LR too low? |
| **B3** | `--lr 1e-3` | LR too high? |
| **B4** | `--freeze all` (head only), `--lr 1e-3` | Pretrained features as a fixed extractor (linear probe) |
| **B5** | `--freeze partial` | Partial fine-tune (Section 2.2 split points) |
| **B6** | `--weight-decay 0.5` | Regularization strength |
| **B7** | `--batch-size 64`, `--lr 1.5e-4` | Batch size (LR scaled with it) |
| **B8** | `--use-hflip` | Does mirroring hurt? |
| **B9** | `--aug strong` (RandomResizedCrop scale 0.08–1 + `TrivialAugmentWide`) | Stronger augmentation |
| **B10** | `--dropout 0.6 --label-smoothing 0.2` | Heavier regularization |

Not in this list, deliberately: **epochs** (the final run uses 12–15 and the report notes the difference), **seed** (the 2 extra seeds in Section 6), **number of classes** (Section 6).

### 5.4 Command template

> The flags marked † are added by S2/S3/S5; the others already exist in `run_pipeline.py`. Adapt names to whatever the merged code uses.

```bash
python scripts/run_pipeline.py --model transfer --backbone resnet18 \   # † resnet18 | efficientnet_b0 | mobilenet_v3_large
  --data-dir /content/data/streetview-256 \
  --num-classes 111 --max-train-per-class 300 \                          # † 
  --resize-to 183 --crop-size 160 \
  --epochs 8 --batch-size 128 --lr 3e-4 --weight-decay 0.05 \
  --dropout 0.3 --label-smoothing 0.1 --no-use-hflip \
  --aug basic --freeze none \                                            # †
  --run-name rn18_B1_default \                                           # †
  --weights-dir /content/drive/MyDrive/geoguessr/weights --seed 42
```

**Run naming:** `<model>_<id>_<short>`, e.g. `rn18_B3_lr1e-3`, `effb0_B5_partial`, `mnv3_B9_strongaug`. Use the same ids for the three models so W&B can group them.

---

## 6. Run order for each person

### Step 0 — Fri 11/20: smoke test
Run S3's one-epoch test on `--num-classes 10 --max-train-per-class 50`. Confirm: loss goes down, W&B logs, a checkpoint appears on Drive, and killing and restarting resumes.

### Step 1 — Sat 11/21: B1 and the timing measurement
Run B1. **Write down the seconds per epoch** and the GPU utilisation (`nvidia-smi`). If utilisation is low, the CPUs are the bottleneck, so tell the team before changing anything. Update the timing estimates in this doc.

### Steps 2–5 — Sun 11/22 → Wed 11/25: B2–B10
Follow the calendar in Section 7. After each run, check the logged row and the W&B curve. Look for overfitting (train loss falling while val loss rises) and note it in the log.

### Step 6 — Wed 11/25 night: pick the best config
Best = lowest **val** top-5 (tie-break on val top-1) among B1–B10. Write it in the log and tell the team. Note that B4 (head only) and B5 are legitimate winners, since transfer learning often wins with less training.

### Step 7 — Thu 11/26 (Thanksgiving): final run
Best config, but **full training data** (no cap), **224px**, **12–15 epochs**, patience 4, `--eval-test`. Start it in the morning and let it run. Expected 30–90 min depending on the model (estimate, to be measured).

### Step 8 — Fri 11/27: class scaling and seeds
These are cheap because the training sets are small.

| Run | Config | Purpose |
|---|---|---|
| **C10** | best config, 10 classes, 300/class, 160px | Accuracy vs. number of classes |
| **C20** | same, 20 classes | |
| **C50** | same, 50 classes | |
| (111) | already done as the best of B1–B10 | |
| **S1, S2** | best config, 111 classes, seeds 43 and 44 | Mean ± std across seeds |

The nested subsets from S2 are what make C10/C20/C50/111 comparable.

### Step 9 — Sat 11/28: analysis and figures (own model)
Run `analysis.py` on the final run's results (S6). Generate Grad-CAM. Write up your section (Section 8).

**EH only:** also run the **from-scratch GWTNet** on all 111 classes with the final settings (224px, full data), around 30–40 epochs because nothing is pretrained. This is the "pretraining helps by X points" row, so it must use the same split and the same test set.

---

## 7. Day-by-day calendar

| Date | Everyone | TT | EH | EW |
|---|---|---|---|---|
| **Thu 11/19** | Read this plan, check W&B + Colab + Drive access, bring objections | **Start S1 (pre-resize) tonight** | Start S3 | Start S5/S6 |
| **Fri 11/20** | **Shared code merged by midnight.** Smoke test (Step 0) | S2, S7, S8 | S3, S4 | S5, S6 |
| **Sat 11/21** | B1 + timing (Step 1) | `rn18_B1` | `effb0_B1` | `mnv3_B1` |
| **Sun 11/22** | B2, B3 | | | |
| **Mon 11/23** | B4, B5 | | | |
| **Tue 11/24** | B6, B7 | | | |
| **Wed 11/25** | B8, B9, B10, then **pick best config** | | | |
| **Thu 11/26** | Final run (Step 7), draw your architecture diagram while it runs | | | |
| **Fri 11/27** | C10/C20/C50 + 2 seeds (Step 8) | + scratch GWTNet run | | |
| **Sat 11/28** | Analysis, Grad-CAM, push results (Step 9) | | | |
| **Sun 11/29** | Team merge: combined tables, figures, decide the story | | | |
| **Mon 11/30** | Report draft (5 pages exactly, IEEE template) | | | |
| **Tue 12/1** | Polish, README, clean-environment re-run check | | | |
| **Wed 12/2** | **Submit** | | | |
| Dec 3–6 | Slides + demo | | | |

That is at least one experiment per person per day from Saturday on. If you can fit B-runs earlier (they're short), do it and bank the time.

**If we slip, cut in this order:** B10 → B7 → C50 → seeds. **Never cut:** B1–B6, B8, the final run, the scratch-vs-pretrained comparison, C10/C20.

---

## 8. Deliverables and who produces what

### Each person (own model)
1. **Architecture diagram** for your model (draw.io or TikZ; use the block listing in Section 2.2 and mark the frozen vs. trained part for B5). Draw it yourself, don't screenshot a paper.
2. All 10 B-runs + C-runs + seed runs in **W&B** and in `experiment_log.csv`.
3. **Final run**: test top-1/3/5/10 and macro-F1, plus the results JSON.
4. **Figures** from `analysis.py`: top-k curve, top-15 confused pairs, continent confusion matrix.
5. **Grad-CAM** (`pytorch-grad-cam`; target layer: ResNet-18 `layer4[-1]`, EfficientNet/MobileNet `features[-1]`): 3 correct and 3 wrong predictions, plus a sentence on what the model looks at (road, sky, signs, vegetation?).
6. **Half a page** on your model's findings: which hyperparameters mattered, what overfit, what surprised you. This must be your own analysis.

### Team (Sun 11/29)
- **Input examples figure** (prof asked): 8–12 labeled images, include visually similar neighbors (e.g. NO/SE, AR/UY).
- **Combined results table** (skeleton below) and **comparison plots**: B-run bar chart per model, accuracy vs. number of classes (C10→111), scratch vs. pretrained learning curves.
- **Demo** (bonus): a Gradio/Streamlit page or notebook that takes an image and shows the top-5 countries from the best model. EW owns it.
- **Report outline** (5 pages exactly): Abstract, Introduction, Related Work, Data, Methods, Experiments & Results, Discussion. TT writes Data/Methods, EH writes Experiments/Results, EW writes Discussion/limitations. Everyone reviews everything. The course says each member must understand the whole project.

### Mapping to the professor's feedback

| Prof feedback | Where it's covered |
|---|---|
| Explore other models, not just ResNet | ResNet-18, EfficientNet-B0, MobileNetV3 |
| Extensive hyperparameter/architecture experiments | 10 B-runs + 3 C-runs + 2 seeds + final, per model, all tracked |
| (i) Show input examples | Team input-examples figure |
| (ii)/(v) Architecture diagram | Shared pipeline figure + one per model |
| (iii) Top-k metrics, not a 2-class confusion matrix | Top-1/3/5/10 everywhere, top-k curve, confused pairs, continent matrix |
| (iv) Logistic regression | One row in the table |
| (vi) Plots/tables | Results table, bar charts, scaling curve, training curves |

### Main results table skeleton (final runs, test set, 111 classes)

| Model | Pretraining | Regime | Top-1 | Top-3 | Top-5 | Top-10 | Macro-F1 |
|---|---|---|---|---|---|---|---|
| Logistic regression (binary FR/JP only) | — | — | | | | | |
| ResNet-18 (GWTNet) | none | scratch | | | | | |
| ResNet-18 | ImageNet | head only (B4) | | | | | |
| ResNet-18 | ImageNet | best config | | | | | |
| EfficientNet-B0 | ImageNet | head only (B4) | | | | | |
| EfficientNet-B0 | ImageNet | best config | | | | | |
| MobileNetV3-L | ImageNet | head only (B4) | | | | | |
| MobileNetV3-L | ImageNet | best config | | | | | |

Seed-variance rows (mean ± std over seeds 42/43/44) go in a second table. Chance is 0.9% top-1 and 4.5% top-5. Our proposal's 70% top-5 target is optimistic for a CNN on this task; report what we get and spend the discussion on *why* (confusable neighbors, small cues like road markings, label noise from unrepresentative street-view frames).

---

## 9. Experiment log (one row per run, also in W&B)

`results/<model>/experiment_log.csv`

| Column | Example |
|---|---|
| run_name | `effb0_B3_lr1e-3` |
| date, person | 2026-11-22, EH |
| model, pretrained, freeze | efficientnet_b0, yes, none |
| num_classes, train_per_class, img_size | 111, 300, 160 |
| lr, weight_decay, batch_size, epochs, dropout, label_smoothing, hflip, aug, seed | |
| best_epoch, train_loss, val_loss | |
| val_top1, val_top3, val_top5, val_top10, val_macro_f1 | |
| test_top1 … (final runs only) | |
| sec_per_epoch, gpu | 48, T4 |
| notes | "val loss rises from epoch 5, overfitting" |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Colab disconnects mid-run | S4 resume + Drive checkpoints; just re-run the command |
| GPU quota runs out | Spread runs across accounts; Kaggle notebooks as a fallback; cut per the order in Section 7 |
| Epochs are still slow (CPU-bound decoding) | Report it on Saturday. Options: more workers, smaller stored size (e.g. 192px), or a uint8 memmap cache |
| Our three sweeps aren't comparable | Same code, same split, same seed, same run ids, changes only through flags |
| A teammate's results are late | Per-day log + W&B make status visible; B-runs are short so work can be reassigned |
| Spatial leakage between train/test (neighboring photos) | Known limitation already in the midterm; keep it in the discussion |
| Test-set tuning | `--eval-test` is off in sweeps (Section 3) |

---

## Appendix: Professor feedback (verbatim)

**Proposal:** Project is feasible and accepted. Please explore other models not just resnet. You will require to submit extensive sets of experiments on different hyperparameters and model architectures.

**Midterm:**
- (i) Show some examples of what your input looks like.
- (ii) Include an architecture diagram of your proposed model.
- (iii) Evaluating Multiclass classification meaningfully is a tedious task. I am not sure why your confusion matrix only includes two classes. Use top-k metrics for evaluation.
- (iv) Given the resolution of your images and number of classes. I would not consider logistic regression even as a baseline. But Okay if you have already done it.
- (v) Make sure to include proposed architecture diagram.
- (vi) Make sure to use plots/tables to summarize your results.
