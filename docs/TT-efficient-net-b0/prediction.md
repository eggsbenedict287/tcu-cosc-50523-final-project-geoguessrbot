# EfficientNet-B0: predictions for B1–B6 (written BEFORE any run)

**Author:** TT · **Model:** EfficientNet-B0 (ImageNet-pretrained) · **Task:** 111-country classification

> **Rule:** these predictions were written and committed *before* launching the runs. Do not edit
> the prediction cells afterwards. Put corrections in the "After the runs" section at the
> bottom. The commit timestamp is the proof that these are real predictions.

## Setup recap (so predictions are comparable)

| Item | Value |
|---|---|
| Classes | 111 (chance top-1 = 0.9%, chance top-5 = 4.5%) |
| Train images | 300 per country (~33k), val ~16k (about 15% of each country) |
| Input | 160px crop from 183px (`--resize-to 183 --crop-size 160`) |
| Epochs | 8, AdamW + OneCycleLR (15% warmup, cosine decay), batch 128, FP16 |
| Fixed settings | weight decay 0.05, dropout 0.3, label smoothing 0.1, no horizontal flip |
| Selection metric | **val top-5** (best epoch inside a run is chosen by val loss) |
| Trainable params | `freeze none` 4.15M · `partial` 3.84M · `all` 0.14M (head only) |

Why hflip is off: driving side and text direction are geographic cues. Flipping an image
can make a right-hand-traffic country look like a left-hand-traffic one.

## B1: default (the reference every other run is compared to)

`--lr 3e-4 --aug basic --freeze none`

| Prediction | Answer |
|---|---|
| Final val top-1 | ~30% |
| Final val top-5 | ~75% |
| Final val macro-F1 | not predicted |
| Train loss vs val loss | Train loss moderately below val loss. The pretrained model will memorize some of the 300-per-country set, so the gap is real but not large. |
| Best epoch | not predicted |
| Why | The 10-country smoke test reached 79% top-1 after 2 epochs. With 111 countries (some with only ~200 images instead of ~1000) accuracy will drop a lot, then recover somewhat over 8 epochs. Only 300 images per country is little data, so I don't expect great results. |
| Confidence | Medium in these numbers, low in the model overall |

## B2–B6: one change from B1 each

### B2: lower learning rate (`--lr 1e-4`)

| Prediction | Answer |
|---|---|
| Val top-5 vs B1 | Worse, by roughly 3–5 points (top-1 too) |
| Train–val gap vs B1 | Smaller |
| Why | A lower learning rate means smaller steps. With a small dataset and only 8 epochs, the model won't move far enough toward the right answer. |
| Confidence | Medium |

### B3: higher learning rate (`--lr 1e-3`)

| Prediction | Answer |
|---|---|
| Val top-5 vs B1 | Worse, by roughly 3 points |
| Train–val gap vs B1 | Bigger. Large steps fit the training images harder and move the weights further from the pretrained ones, so unseen val images are handled worse. |
| Unstable? | Possibly. At 1e-3 the steps may be too large, so the loss can jump around, especially at the peak of the learning-rate schedule, instead of going steadily down. |
| Why | A higher learning rate may learn faster at first, but its large steps risk overwriting the useful pretrained features and add noise, so it ends worse than B1 after 8 epochs. |
| Confidence | Medium to high |

### B4: head only (`--freeze all --lr 1e-3`), the "linear probe"

| Prediction | Answer |
|---|---|
| Val top-5 vs B1 | Worse, by roughly 8 points |
| Train–val gap vs B1 | Smaller (both train and val will be mediocre) |
| Seconds per epoch vs B1 | Much faster, since there is no backward pass through the backbone. (Note: the 2-CPU image-decoding limit may shrink this saving.) |
| Why | The backbone stays at its ImageNet weights and only a tiny linear layer learns from our data. ImageNet features don't know much about the vegetation, road markings and signs that identify countries. |
| Confidence | Medium |

### B5: partial freeze (`--freeze partial`), stem + stages 1–4 frozen

| Prediction | Answer |
|---|---|
| Val top-5 vs B1 | Better, by roughly 6 points. Predicted to be the best run. |
| Val top-5 vs B4 | Above B4. B5 can adapt the later layers, which B4 can't. |
| Train–val gap vs B1 | Smaller. B1 trains the whole network, so it has the bigger gap. B5 trains 3.84M of 4.15M parameters, and its early layers can't drift. |
| Why | Trust the pretrained early layers for generic features (edges, textures) and let the later layers learn country-specific patterns. |
| Confidence | Medium to high |

### B6: strong augmentation (`--aug strong`), RandomResizedCrop (scale 0.08–1) + TrivialAugmentWide

| Prediction | Answer |
|---|---|
| Val top-5 vs B1 | Worse, by roughly 11 points |
| Train loss vs B1 | Higher. The model can't memorize heavily distorted images. |
| Train–val gap vs B1 | Smaller |
| Why | A very aggressive random crop makes the task harder, and with only 8 epochs the model can't learn enough from it. |
| Confidence | Low |

## Summary predictions

| Question | Answer |
|---|---|
| Ranking by val top-5 (best → worst) | **B5 > B1 > B3 > B2 > B4 > B6** (B2 and B3 are nearly tied) |
| Predicted winner | B5, so the final run would use `--freeze partial` |
| Biggest single effect on accuracy | **Freezing.** Predictions span from -8 (B4) to +6 (B5), a 14-point range, against about 4 points for learning rate. Augmentation (B6) is the biggest single drop at -11. |
| Least sure about | B6 (low confidence) |

## Error predictions (to compare with the confusion matrix later)

| Question | Answer |
|---|---|
| Country pairs most confused | Canada ↔ USA; Switzerland ↔ Austria; Italy ↔ Monaco |
| Easiest countries (high recall) | Spain, Ireland, Brazil |
| Hardest countries | Monaco, Liechtenstein, Poland |
| Small countries (BM, CC, MC, GI with 110–237 images) | Much harder: far fewer training images. They will mostly be confused with adjacent countries. |
| Mostly within-continent confusions? | Yes, with neighbouring countries that share road markings, language and architecture. |

> Note: the Liechtenstein entry only applies if it is in the 111-country dataset. Verify with
> `ls data/processed/streetview-256` before analysing. (Monaco = MC, 224 images, is in the dataset.)

## After the runs (fill in later, do not edit above)

| Run | Predicted val top-5 | Actual val top-5 | Right direction? | What surprised you / why were you wrong? |
|---|---|---|---|---|
| B1 | ~75% | | | |
| B2 | B1 - 4 | | | |
| B3 | B1 - 3 | | | |
| B4 | B1 - 8 | | | |
| B5 | B1 + 6 | | | |
| B6 | B1 - 11 | | | |
