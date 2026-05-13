# Data Collection and Learning Pipeline for Goal Inference in NAMO

This document describes the data collection, mask generation, and diffusion-based goal inference training pipeline. Use this as a reference for writing the Methods section of the R:SS paper.

---

## 1. Data Collection

### 1.1 Overview

Data collection generates successful push trajectories by running a tree-structured planner on procedurally generated NAMO environments. Each episode captures the complete state before and after each push action, enabling supervised learning of goal inference.

### 1.2 Episode Structure

Each collected episode contains:

| Field | Description |
|-------|-------------|
| `state_observations` | SE(2) poses of all objects before each action: `{robot: [x,y,θ], obj1: [x,y,θ], ...}` |
| `post_action_state_observations` | SE(2) poses after each action (captures actual outcomes) |
| `action_sequence` | List of push actions: `{object_id, target: [x,y,θ]}` |
| `robot_goal` | Navigation target position `[x, y, θ]` |
| `static_object_info` | Object geometry (half-extents) and static obstacle poses |
| `reachable_objects_before_action` | Objects accessible to robot before each push |
| `region_goals_sampled` | Sampled robot goal positions used during planning |

### 1.3 Planning Algorithm

The planner uses **region-based decomposition** with the following procedure:

```
Algorithm: Region Opening Planner

Input: Environment E, robot goal G
Output: Push sequence that enables robot to reach G

1. Identify neighbor regions R_N adjacent to robot's current region
2. For each neighbor region R in R_N:
   a. Sample candidate robot goals G_R within R
   b. For each reachable object O:
      i.  Generate candidate push goals using primitive motion library
      ii. Execute push simulation for each candidate
      iii. Validate: Does pushing O to goal position enable path to G_R?
   c. If valid push found, record (O, target_pose, trajectory)
3. Return successful push chains with state observations
```

### 1.4 Trajectory Suffix Decomposition

Multi-step episodes are decomposed into multiple training examples via **trajectory suffix splitting**:

```
Original episode with n pushes:
  States: [S_0, S_1, ..., S_{n-1}]
  Actions: [A_0, A_1, ..., A_{n-1}]

Generates n training examples:
  Example 1: (S_0, [A_0, A_1, ..., A_{n-1}])  → predict all future goals
  Example 2: (S_1, [A_1, A_2, ..., A_{n-1}])  → predict remaining goals
  ...
  Example n: (S_{n-1}, [A_{n-1}])             → predict final goal
```

This augmentation ensures the model learns to predict goals from any intermediate state.

---

## 2. Mask Generation

### 2.1 Overview

Raw SE(2) pose data is converted to binary image masks for neural network training. Masks are rendered at **224×224** resolution with objects represented as filled rotated rectangles.

### 2.2 Coordinate Transformation

World coordinates are mapped to pixel coordinates via:

```
world_size = max(x_max - x_min, y_max - y_min)
scale = 224 / world_size  [pixels per meter]

pixel_x = (x - world_center_x) × scale + 112
pixel_y = (y - world_center_y) × scale + 112
```

This ensures square aspect ratio with the environment centered in the image.

### 2.3 Global Mask Channels

| Mask | Description |
|------|-------------|
| `robot` | Robot position (filled circle, r=0.15m) |
| `goal` | Robot navigation goal (filled circle, r=0.05m) |
| `static` | Static obstacles/walls (filled rotated rectangles) |
| `movable` | All movable objects (filled rotated rectangles) |
| `target_object` | The specific object to be pushed |
| `target_goal` | Target object drawn at its push destination |
| `robot_distance` | Distance field from robot position |
| `goal_distance` | Distance field from goal position |

### 2.4 Local (Object-Centered) Mask Generation

For improved spatial resolution around the manipulation target, we generate **object-centered local masks**:

```
Algorithm: Local Mask Generation

Input: Episode with target object at (x_obj, y_obj, θ_obj)
Parameters: crop_size = 5.0m, highres_size = 1024px, output_size = 224px

1. Render full scene at high resolution (1024×1024)
2. Compute scale: pixels_per_meter = 1024 / world_size
3. Locate target object center in pixel coordinates
4. Define crop window: 5m × 5m centered on target object
5. Extract crop region from high-resolution render
6. Resize crop to 224×224 using area interpolation
7. Repeat for all mask channels
```

### 2.5 Local Mask Channels

| Mask | Description |
|------|-------------|
| `local_static` | Static obstacles within crop region |
| `local_movable` | Other movable objects within crop region |
| `local_target_object` | Target object centered in crop |
| `local_target_goal` | Push target position |
| `local_robot_region` | Binary reachability from robot position (BFS) |
| `local_goal_sample_region` | Binary reachability from sampled goal position (BFS) |

### 2.6 Reachability Region Computation

Robot and goal reachability regions are computed via 8-connected BFS on an inflated obstacle grid:

```
Algorithm: Reachability BFS

Input: Start position (x, y), obstacle masks, robot half-extent
Output: Binary reachability mask

1. Build inflated obstacle grid:
   For each obstacle with half-extents (sx, sy) at position (cx, cy, θ):
     inflated_sx = sx + robot_half_x + ε  (ε = 0.005m)
     inflated_sy = sy + robot_half_y + ε
     Draw rotated rectangle at (cx, cy) with inflated size

2. Initialize visited grid to False
3. Clear 3×3 neighborhood around start position (handle discretization)
4. BFS with 8-connectivity:
   Queue.push(start_pixel)
   While Queue not empty:
     p = Queue.pop()
     For each of 8 neighbors n of p:
       If n not visited AND n not in inflated_obstacles:
         visited[n] = True
         Queue.push(n)

5. Return visited grid as reachability mask
```

### 2.7 Multi-Horizon Goal Masks

For multi-push chains, we generate goal masks for each future action:

```
For n-push trajectory suffix starting at state S_i:
  goal_mask_a1: Target object at action[0].target  (next push goal)
  goal_mask_a2: Target object at action[1].target  (second push goal)
  ...
  goal_mask_an: Target object at action[n-1].target (final push goal)
```

This enables the model to predict multiple future goals simultaneously.

---

## 3. Model Architecture

### 3.1 Overview

The goal inference model uses a **Diffusion Transformer (DiT)** with cross-attention for context conditioning. The model predicts goal positions as denoised mask images.

### 3.2 Input/Output Specification

```
Context Input: (B, 5, 64, 64) - scene representation
  Channel 0: local_static      - static obstacles
  Channel 1: local_movable     - other movable objects
  Channel 2: local_target_object - object to push (centered)
  Channel 3: local_robot_region  - robot reachability mask
  Channel 4: local_goal_sample_region - goal reachability mask

Noisy Input: (B, 1, 32, 32) or (B, 2, 32, 32) - noised goal mask(s)
  Channel 0: goal_mask_a1 - first push target
  Channel 1: goal_mask_a2 - second push target (if multi-horizon)

Output: (B, 1, 32, 32) or (B, 2, 32, 32) - denoised goal prediction
```

### 3.3 Architecture Components

#### Spatial Context Encoder (CNN)

Encodes the 64×64 context into spatial features for cross-attention:

```
Context (B, 5, 64, 64)
  → Conv2d(5, 32, k=4, s=2) + SiLU      → (B, 32, 32, 32)
  → Conv2d(32, 64, k=4, s=2) + SiLU     → (B, 64, 16, 16)
  → Conv2d(64, 128, k=4, s=2) + SiLU    → (B, 128, 8, 8)
  → Conv2d(128, D, k=3, s=1)            → (B, D, 8, 8)
  → Flatten + Transpose                  → (B, 64, D)  [64 context tokens]
```

#### Patch Embedding (for noisy input)

Tokenizes the 32×32 noisy goal mask:

```
Noisy input (B, 1, 32, 32)
  → Conv2d(1, D, k=4, s=4)              → (B, D, 8, 8)
  → Flatten + Transpose                  → (B, 64, D)  [64 input tokens]
  → Add learned positional embedding     → (B, 64, D)
```

#### Transformer Block with Cross-Attention

Each block performs:

```
Input tokens x: (B, N, D)
Context tokens ctx: (B, N_ctx, D)
Time embedding t_emb: (B, D)

1. Self-Attention:
   h = LayerNorm(x)
   x = x + MultiHeadSelfAttn(Q=h, K=h, V=h)

2. Cross-Attention to Context:
   h = LayerNorm(x)
   x = x + MultiHeadCrossAttn(Q=h, K=ctx, V=ctx)

3. MLP with Time Modulation:
   h = LayerNorm(x)
   scale, shift = Linear(t_emb).chunk(2)
   h = h × (1 + scale) + shift
   x = x + MLP(h)

Output: (B, N, D)
```

#### Un-patchify

Reconstructs the output mask from tokens:

```
Tokens (B, 64, D)
  → Reshape to (B, D, 8, 8)
  → ConvTranspose2d(D, 1, k=4, s=4)     → (B, 1, 32, 32)
```

### 3.4 Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension D | 256 |
| Transformer depth | 8 blocks |
| Attention heads | 8 |
| MLP ratio | 4 |
| Context size | 64×64 |
| Crop/output size | 32×32 |
| Patch size | 4 |

---

## 4. Training with Denoising Diffusion

### 4.1 Overview

We use **Denoising Diffusion Implicit Models (DDIM)** with a squared cosine noise schedule. The model learns to predict the noise added to goal masks, enabling iterative denoising from random noise to structured predictions.

### 4.2 Forward Diffusion Process

The forward process adds noise to data x₀ according to a variance schedule:

```
x_t = √(ᾱ_t) × x₀ + √(1 - ᾱ_t) × ε,  ε ~ N(0, I)

Where:
  ᾱ_t = cumulative product of (1 - β_t)
  β_t follows squared cosine schedule (squaredcos_cap_v2)
  t ∈ {1, 2, ..., T},  T = 1000
```

### 4.3 Training Objective

The model predicts the noise ε added to the data:

```
Algorithm: Diffusion Training Step

Input: Batch of (context, target_goal) pairs
Output: Loss value

1. Sample noise: ε ~ N(0, I), shape (B, C, H, W)
2. Sample timestep: t ~ Uniform{1, ..., 1000}
3. Compute noisy sample: x_t = √(ᾱ_t) × x₀ + √(1 - ᾱ_t) × ε
4. Forward pass: ε̂ = Model(context, x_t, t)
5. Compute loss: L = MSE(ε̂, ε)
6. Backpropagate and update weights
```

### 4.4 Noise Schedule

We use the **squared cosine** schedule which provides smoother noise levels:

```
ᾱ_t = cos²((t/T + s) / (1 + s) × π/2)

Where s = 0.008 is a small offset for numerical stability
```

This schedule adds noise more gradually than linear schedules, improving training stability.

### 4.5 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Base learning rate | 4×10⁻⁴ |
| End learning rate | 1×10⁻⁶ |
| Batch size | 64 |
| Warmup steps | 1,000 |
| Decay steps | 300,000 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 (norm) |
| Precision | Mixed FP16 |
| Training timesteps | 1,000 |
| Max epochs | 1,000 |
| Image normalization | [-1, 1] |

---

## 5. Inference (Sampling)

### 5.1 DDIM Sampling

At inference, we use DDIM for deterministic sampling with fewer steps than training:

```
Algorithm: DDIM Sampling

Input: Context, number of steps K (default 50)
Output: Generated goal mask

1. Initialize: x_T ~ N(0, I)
2. Create timestep subsequence: [t_K, t_{K-1}, ..., t_1, t_0]
   (K evenly spaced steps from T=1000 to 0)
3. For k = K to 1:
   a. Predict noise: ε̂ = Model(context, x_{t_k}, t_k)
   b. Compute predicted x₀: x̂₀ = (x_{t_k} - √(1-ᾱ_{t_k}) × ε̂) / √(ᾱ_{t_k})
   c. DDIM update (η=0 for deterministic):
      x_{t_{k-1}} = √(ᾱ_{t_{k-1}}) × x̂₀ + √(1-ᾱ_{t_{k-1}}) × ε̂
4. Return x₀ as generated goal mask
```

### 5.2 Sampling Parameters

| Parameter | Value |
|-----------|-------|
| Inference steps | 5 (vs 1000 training) |
| Sampler | DDIM |
| η (stochasticity) | 0.0 (deterministic) |
| Clip sample | False |
| Samples per inference | 32 |

### 5.4 Coordinate Recovery

To convert predicted mask pixels back to world coordinates:

```
1. Threshold predicted mask to find goal region centroid
2. Convert pixel coordinates to local crop coordinates:
   local_x = (pixel_x / 224 - 0.5) × crop_size_meters
   local_y = (pixel_y / 224 - 0.5) × crop_size_meters
3. Transform to world coordinates:
   world_x = local_x + object_center_x
   world_y = local_y + object_center_y
4. Extract rotation from predicted mask orientation
```

### 5.5 ML-Guided Planning with Voting

Rather than directly using ML-predicted coordinates, we use a **voting mechanism** to align ML predictions with kinematically-validated primitive motions:

```
Algorithm: ML-Guided Goal Selection with Voting

Input: Scene context, primitive motion library P
Parameters: N=32 samples, K=5 nearest neighbors,
            pos_tol=0.2m, angle_tol=0.2rad

1. Generate N=32 diffusion samples (5 DDIM steps each)
2. For each sample s:
   a. Threshold to binary mask, extract centroid (x_s, y_s, θ_s)
   b. Find K=5 nearest primitive slots within tolerance:
      For each primitive p in P:
        dist = ||[x_s, y_s] - [x_p, y_p]|| + w × |θ_s - θ_p|
        If dist < threshold: add p to candidates
   c. Vote for top-K nearest primitives
3. Aggregate votes across all N samples
4. Select primitive with highest vote count
5. Execute selected primitive motion

Tolerance matching:
  Position error < 0.2m AND angle error < 0.2 rad (~11°)
```

This hybrid approach combines:
- **ML predictions**: Learn scene-dependent goal distributions
- **Primitive validation**: Ensure kinematic feasibility via pre-computed motions
- **Voting robustness**: Multiple samples reduce sensitivity to individual prediction errors

### 5.6 Goal Strategy: ML-Fallback

The planner uses `ml_fallback` strategy:

```
1. Query ML model for goal predictions
2. Align predictions to primitive library via voting
3. If ML provides valid goals: use ML-ranked primitives first
4. If ML fails or no matches: fall back to geometric primitive ordering
```

| Inference Parameter | Value |
|---------------------|-------|
| Diffusion samples | 32 |
| DDIM steps | 5 |
| Voting K-nearest | 5 |
| Position tolerance | 0.2m |
| Angle tolerance | 0.2 rad |
| Random seed | 42 (reproducible) |

---

## 6. Data Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Procedural   │ → │ Region-Based │ → │ Trajectory   │       │
│  │ Environments │    │ Planner      │    │ Suffix Split │       │
│  │              │    │              │    │ (augment)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              ↓                                   │
│                    [state_obs, actions, goals]                   │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MASK GENERATION                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ High-Res     │ → │ Object-      │ → │ Reachability │       │
│  │ Rendering    │    │ Centered     │    │ BFS          │       │
│  │ (1024×1024)  │    │ Crop (5m)    │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              ↓                                   │
│              [local_static, local_movable, local_target,         │
│               robot_region, goal_region, goal_mask_a1/a2]        │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (DDIM Diffusion)                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Context      │ → │ DiT with     │ → │ Noise        │       │
│  │ Encoder      │    │ Cross-Attn   │    │ Prediction   │       │
│  │ (CNN)        │    │ (Transformer)│    │ Loss (MSE)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  Input:  context (64×64×5) + noisy_goal (32×32×1)               │
│  Output: predicted_noise ε̂ (32×32×1)                            │
│  Loss:   MSE(ε̂, ε)  with squared-cosine schedule               │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (ML-Guided Planning)                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 32 Diffusion │ → │ Primitive    │ → │ Vote         │       │
│  │ Samples      │    │ Alignment    │    │ Aggregation  │       │
│  │ (5 DDIM steps)│   │ (K=5 nearest)│    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              ↓                                   │
│              [highest-voted primitive motion executed]           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Design Decisions

### 7.1 Why Object-Centered Crops?

- **Higher effective resolution**: 5m crop at 224px = 22mm/pixel vs global scene at ~100mm/pixel
- **Translation invariance**: Object always centered, model learns relative positioning
- **Reduced scene complexity**: Focuses attention on manipulation-relevant region

### 7.2 Why Reachability Masks?

- **Encodes kinematic constraints**: Robot/goal reachability determines valid push directions
- **Provides implicit collision information**: Blocked regions appear as holes in reachability
- **Enables reasoning about connectivity**: Model learns when pushes enable new paths

### 7.3 Why DDIM Diffusion with Fast Inference?

- **Squared cosine schedule**: Smoother noise levels improve training stability
- **DDIM deterministic sampling**: Only 5 steps needed (vs 1000 training timesteps)
- **Noise prediction**: Standard ε-prediction enables well-understood sampling

### 7.4 Why Voting-Based Primitive Alignment?

- **Kinematic validation**: Ensures predicted goals correspond to feasible motions
- **Robustness**: 32 samples with K=5 voting reduces sensitivity to outliers
- **Hybrid approach**: Combines learned scene understanding with motion primitives
- **Graceful fallback**: If ML fails, system reverts to geometric primitive ordering

### 7.5 Why Cross-Attention over AdaLN?

- **Spatial context preservation**: Cross-attention maintains 2D structure of scene
- **Selective attention**: Model can focus on relevant obstacles/regions
- **Better generalization**: Decouples context encoding from denoising dynamics
