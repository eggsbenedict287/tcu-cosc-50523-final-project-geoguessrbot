# EfficientNet-B0 architecture (as used in this project)

Source: `torchvision.models.efficientnet_b0` (ImageNet weights) wrapped by `TransferNet` in
`scripts/transfer/model.py`. Spatial sizes below are for our **160px sweep input**
(the final run uses 224px, so every size scales by 1.4×).

GitHub renders the Mermaid diagrams below directly. If they do not render, paste the code
into <https://mermaid.live>.

## Diagram 1: full pipeline

Colors show which parts are **frozen** in each experiment:
- **blue** = frozen in B5 (`--freeze partial`, `features.0`–`features.4`) and in B4
- **orange** = trained in B5; frozen in B4
- **green** = new head, trained in every run (random init)

```mermaid
flowchart TD
    IN["Input image<br/>3 x 160 x 160<br/>ImageNet mean/std normalized"]

    subgraph EARLY["features.0 to features.4: generic features (frozen in B4 and B5)"]
        direction TB
        S0["features.0 · Stem<br/>Conv 3x3 stride 2 + BN + SiLU<br/>32 ch · 80 x 80"]
        S1["features.1 · Stage 1<br/>1 x MBConv1, k3, stride 1<br/>16 ch · 80 x 80"]
        S2["features.2 · Stage 2<br/>2 x MBConv6, k3, stride 2<br/>24 ch · 40 x 40"]
        S3["features.3 · Stage 3<br/>2 x MBConv6, k5, stride 2<br/>40 ch · 20 x 20"]
        S4["features.4 · Stage 4<br/>3 x MBConv6, k3, stride 2<br/>80 ch · 10 x 10"]
        S0 --> S1 --> S2 --> S3 --> S4
    end

    subgraph LATE["features.5 to features.8: task-specific features (trained in B5, frozen in B4)"]
        direction TB
        S5["features.5 · Stage 5<br/>3 x MBConv6, k5, stride 1<br/>112 ch · 10 x 10"]
        S6["features.6 · Stage 6<br/>4 x MBConv6, k5, stride 2<br/>192 ch · 5 x 5"]
        S7["features.7 · Stage 7<br/>1 x MBConv6, k3, stride 1<br/>320 ch · 5 x 5"]
        S8["features.8 · Head conv<br/>Conv 1x1 + BN + SiLU<br/>1280 ch · 5 x 5"]
        S5 --> S6 --> S7 --> S8
    end

    POOL["Global average pool<br/>1280 x 5 x 5 → 1280"]

    subgraph HEAD["classifier: NEW head (trained in every run)"]
        direction TB
        DO["Dropout p = 0.3<br/>(--dropout)"]
        FC["Linear 1280 → 111<br/>one logit per country"]
        DO --> FC
    end

    OUT["Softmax → country probabilities<br/>top-1 / top-3 / top-5 / top-10"]

    IN --> S0
    S4 --> S5
    S8 --> POOL --> DO
    FC --> OUT

    classDef frozen fill:#cfe2ff,stroke:#2b5fa5,color:#000
    classDef late fill:#ffe5b4,stroke:#c77700,color:#000
    classDef head fill:#d4edda,stroke:#2e7d32,color:#000
    classDef io fill:#eeeeee,stroke:#555,color:#000
    class S0,S1,S2,S3,S4 frozen
    class S5,S6,S7,S8 late
    class DO,FC head
    class IN,POOL,OUT io
```

**Counts (111-class head):** 16 MBConv blocks in total (1 + 2 + 2 + 3 + 3 + 4 + 1).
Blocks 1–8 sit in the frozen half, blocks 9–16 in the trained half.

| `--freeze` | What trains | Trainable params |
|---|---|---|
| `none` | everything | 4.15M |
| `partial` | `features.5`–`features.8` + classifier | 3.84M |
| `all` | classifier only (1280 × 111 + 111) | 0.14M |

Most of the weights are in the late stages (wide channels), which is why `partial`
freezing still leaves 3.84M of the 4.15M trainable.

## Diagram 2: one MBConv block (the building block, 16 per network)

MBConv = **inverted bottleneck** (narrow → wide → narrow) + **squeeze-and-excite** + residual.
Example: `MBConv6` with input channels `C_in`, output channels `C_out`, expansion ×6.

```mermaid
flowchart TD
    X["Input feature map<br/>C_in x H x W"]

    subgraph EXPAND["1 · Expand (skipped when expansion = 1)"]
        E["1x1 conv, C_in → 6 x C_in<br/>BN + SiLU"]
    end

    subgraph DW["2 · Depthwise conv (spatial filtering)"]
        D["k x k depthwise conv (k = 3 or 5)<br/>one filter PER channel · stride 1 or 2<br/>BN + SiLU"]
    end

    subgraph SE["3 · Squeeze-and-Excite (channel attention)"]
        direction TB
        SQ["Global avg pool<br/>wide channels x 1 x 1"]
        F1["1x1 conv: shrink to C_in / 4<br/>SiLU"]
        F2["1x1 conv: back to wide channels<br/>Sigmoid → weights in 0..1"]
        SQ --> F1 --> F2
    end

    MUL(("x<br/>scale each<br/>channel"))

    subgraph PROJ["4 · Project (linear bottleneck)"]
        P["1x1 conv, 6 x C_in → C_out<br/>BN · NO activation"]
    end

    ADD(("+"))
    Y["Output feature map<br/>C_out x H' x W'"]

    X --> E --> D
    D --> SQ
    D --> MUL
    F2 --> MUL
    MUL --> P --> P2["Stochastic depth<br/>(randomly drops the block<br/>during training)"]
    P2 --> ADD
    X -. "skip connection<br/>only if stride 1 and C_in = C_out" .-> ADD
    ADD --> Y

    classDef expand fill:#e8eaf6,stroke:#3949ab,color:#000
    classDef dw fill:#fff3e0,stroke:#ef6c00,color:#000
    classDef se fill:#fce4ec,stroke:#c2185b,color:#000
    classDef proj fill:#e0f2f1,stroke:#00796b,color:#000
    classDef io fill:#eeeeee,stroke:#555,color:#000
    class E expand
    class D dw
    class SQ,F1,F2,MUL se
    class P,P2 proj
    class X,Y,ADD io
```

### Why each step exists (for Q&A)

| Step | What it does | Why it matters |
|---|---|---|
| 1x1 **expand** (×6) | Widens channels so the block can mix information in a rich space | The cheap 1x1 conv is where most learning happens |
| **Depthwise** k×k | Filters each channel on its own, with no mixing across channels | Costs about `k²·C` instead of `k²·C²` of a normal conv, so it is far fewer FLOPs for similar accuracy |
| **Squeeze-and-excite** | Pools each channel to one number, a tiny MLP turns that into a 0–1 weight per channel, and the weights rescale the channels | The network learns *which channels matter for this image* (e.g. "road marking" detectors for this photo) |
| 1x1 **project**, no activation | Shrinks back to a narrow `C_out` | Linear bottleneck: a nonlinearity in the narrow space would destroy information |
| **Skip connection** | Adds the input to the output when shapes match | Same reason as ResNet: easier gradient flow, so deep stacks train |
| **Stochastic depth** | Randomly skips whole blocks in training | Regularization (torchvision uses 0.2 for B0 on ImageNet) |

**Inverted bottleneck vs ResNet bottleneck:** ResNet goes wide → narrow → wide (the
skip connects the *wide* ends). MobileNetV2/EfficientNet goes narrow → wide → narrow (the
skip connects the *narrow* ends, so the thing stored between blocks is small and cheap).

## Diagram 3 (optional): where our changes plug in

```mermaid
flowchart LR
    A["Stock EfficientNet-B0<br/>classifier = Dropout 0.2 + Linear 1280 → 1000"]
    B["TransferNet<br/>classifier = Dropout 0.3 + Linear 1280 → 111"]
    C["Freeze mode<br/>none / partial / all<br/>frozen BN kept in eval()"]
    D["AdamW param groups<br/>no weight decay on biases + BN"]
    E["OneCycleLR<br/>warmup 15% then cosine"]
    A -->|"replace head (_replace_head)"| B
    B -->|"set requires_grad (_apply_freeze)"| C
    C --> D --> E
```

## Compound scaling (EfficientNet's main idea, one paragraph for the report)

Earlier CNNs scaled up only one thing (depth, width or resolution). EfficientNet scales all
three together with one coefficient. B0 is the baseline found by neural architecture search,
and B1–B7 are the same network scaled up. We use **B0** because it has about 4M parameters,
which fits the T4 and the time we have, and because it reaches ImageNet accuracy comparable
to ResNet-50 with roughly 5× fewer parameters.
