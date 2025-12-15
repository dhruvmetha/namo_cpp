# ML Inference Pipeline for Region Opening

This document explains the complete ML-guided inference pipeline used when running:
```bash
./scripts/run_region_opening_ml_sequential.sh --start-idx 0 --end-idx 100
```

## Table of Contents
1. [High-Level Overview](#1-high-level-overview)
2. [Model Architecture: Diffusion Transformer (DiT)](#2-model-architecture-diffusion-transformer-dit)
3. [Input Format: Local Masks](#3-input-format-local-masks)
4. [Output Format: Goal Masks](#4-output-format-goal-masks)
5. [Diffusion Sampling Process](#5-diffusion-sampling-process)
6. [Output Processing: Mask to SE(2)](#6-output-processing-mask-to-se2)
7. [Voting Mechanism: ML to Primitive Alignment](#7-voting-mechanism-ml-to-primitive-alignment)
8. [Search Algorithm: Region Opening BFS](#8-search-algorithm-region-opening-bfs)
9. [Failure Tracking](#9-failure-tracking)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. High-Level Overview

The pipeline predicts **where to push objects** to create navigation openings between regions. It combines:
- **Learned component**: A diffusion model predicts goal positions as image masks
- **Classical component**: Motion primitives ensure physical feasibility

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

Environment XML
      │
      ▼
┌─────────────────────┐
│ Region Connectivity │  Identify neighbors, blocking objects, region goals
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ For Each Neighbor   │  Check if already accessible, find candidate objects
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ ML Goal Generation  │  DiT diffusion model → 16 goal mask samples
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Primitive Alignment │  Vote-based mapping to 600 motion primitive slots
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ BFS Search          │  Try goals in priority order (highest votes first)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Validation          │  Check if opening was created via wavefront
└─────────────────────┘
      │
      ▼
Success/Failure + Stats
```

---

## 2. Model Architecture: Diffusion Transformer (DiT)

**File:** `sage_learning/src/model/dit/dit.py`

The model is a **Diffusion Transformer (DiT)** - a Vision Transformer adapted for diffusion-based generation.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DIFFUSION TRANSFORMER (DiT)                           │
│                                                                                │
│   Config: img_size=64, patch=4, in_ch=6, dim=256, depth=8, heads=8            │
└──────────────────────────────────────────────────────────────────────────────┘

INPUT                                           TIMESTEP
x ∈ ℝ^(B × 6 × 64 × 64)                        t ∈ [0, 1]
   │                                               │
   │  5 context + 1 noisy target                   │
   ▼                                               ▼
┌──────────────────────┐                    ┌──────────────────────┐
│    PATCH EMBEDDING   │                    │   TIME EMBEDDING     │
│  Conv2d(6→256, k=4)  │                    │  Sinusoidal + MLP    │
│  64×64 → 16×16 = 256 │                    │  t → (B, 256)        │
│  patches/tokens      │                    │                      │
└──────────────────────┘                    └──────────────────────┘
           │                                           │
           ▼                                           │
┌──────────────────────┐                               │
│ + POSITIONAL EMB     │                               │
│ Learned (1, 256, 256)│                               │
└──────────────────────┘                               │
           │                                           │
           ▼                                           │
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCKS × 8 (AdaLN-Zero)                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  For each block:                                                         │ │
│  │    1. t_emb → Linear(256→1024) → (γ₁, β₁, γ₂, β₂)  [AdaLN params]       │ │
│  │    2. x' = γ₁ · LayerNorm(x) + β₁                                        │ │
│  │    3. x = x + MultiHeadAttention(x', x', x')  [8 heads]                  │ │
│  │    4. x' = γ₂ · LayerNorm(x) + β₂                                        │ │
│  │    5. x = x + MLP(x')  [256 → 1024 → 256, GELU]                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│    FINAL LAYERNORM   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│     UN-PATCHIFY      │
│ Reshape + ConvT2d    │
│ (256→1, k=4, s=4)    │
└──────────────────────┘
           │
           ▼
OUTPUT: ε̂ ∈ ℝ^(B × 1 × 64 × 64)
        Predicted noise
```

### Key Components

| Component | Details |
|-----------|---------|
| **Patch Embedding** | Conv2d with kernel=stride=4, converts 64×64 image to 256 tokens |
| **Positional Embedding** | Learned, shape (1, 256, 256) |
| **Time Embedding** | Sinusoidal encoding → 2-layer MLP |
| **AdaLN-Zero** | Adaptive LayerNorm with zero-initialized projection (stable training) |
| **Attention** | Standard multi-head self-attention, 8 heads |
| **MLP** | 2-layer with 4× expansion ratio, GELU activation |
| **Un-patchify** | Transposed convolution to reconstruct spatial dimensions |

### Model Size

- **Parameters**: ~12M (with default config)
- **Input channels**: 6 (5 context + 1 noisy target)
- **Output channels**: 1 (denoised goal mask)

---

## 3. Input Format: Local Masks

**Files:**
- `namo/python/namo/visualization/ml_image_converter_adapter.py`
- `sage_learning/src/data/mask_diffusion_data.py`

The model operates on **local masks** - a 5m × 5m crop centered on the object being pushed.

### Channel Layout (5 Context Channels)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCAL MASKS (5m × 5m crop, 64×64 pixels)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Channel 0: local_static        Channel 1: local_movable                   │
│   ┌───────────────────┐          ┌───────────────────┐                      │
│   │ ████       ████   │          │                   │                      │
│   │ ████       ████   │          │   ▓▓▓     ▓▓▓    │                      │
│   │ ████       ████   │          │           ▓▓▓    │                      │
│   │ ████       ████   │          │                   │                      │
│   └───────────────────┘          └───────────────────┘                      │
│   Walls, fixed obstacles         Other movable objects                      │
│                                                                              │
│   Channel 2: local_target_object Channel 3: local_robot_region              │
│   ┌───────────────────┐          ┌───────────────────┐                      │
│   │                   │          │ ░░░░░░░░░░░░░░░░  │                      │
│   │       ████        │          │ ░░░░░░░░░░░░░░░░  │                      │
│   │       ████        │          │ ░░░░░░░░          │                      │
│   │                   │          │ ░░░░░░░░          │                      │
│   └───────────────────┘          └───────────────────┘                      │
│   The object being pushed        Robot's reachable area (wavefront)         │
│                                                                              │
│   Channel 4: local_goal_sample_region                                       │
│   ┌───────────────────┐                                                     │
│   │                   │                                                     │
│   │         ○○○       │          ○ = Sampled goal positions in              │
│   │         ○○○       │              the target neighbor region             │
│   │                   │                                                     │
│   └───────────────────┘                                                     │
│   Where robot wants to reach                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Preprocessing Pipeline

```python
# 1. Create local masks centered on object (5m × 5m window)
local_data = image_converter.create_local_masks(
    data_point=json_message,
    selected_object=object_id,
    robot_goal_pos=robot_goal,
    region_goals_sampled=region_goals,  # List of (x, y, θ) in target region
    crop_size_meters=5.0,
    highres_size=1024,      # Render at high res first
    output_size=224         # Then downsample
)

# 2. Stack channels in training order
input_channels = [
    local_data['local_static'],           # (224, 224, 1)
    local_data['local_movable'],          # (224, 224, 1)
    local_data['local_target_object'],    # (224, 224, 1)
    local_data['local_robot_region'],     # (224, 224, 1)
    local_data['local_goal_sample_region'],  # (224, 224, 1)
]
inp = np.concatenate(input_channels, axis=-1)  # (224, 224, 5)

# 3. Transform for model
transform = transforms.Compose([
    transforms.ToTensor(),                 # (5, 224, 224), [0, 1]
    transforms.Resize((64, 64)),           # (5, 64, 64)
    transforms.Lambda(lambda x: x * 2 - 1) # [-1, 1]
])
inp = transform(inp).unsqueeze(0)  # (1, 5, 64, 64)
```

### Why Local Masks?

1. **Translation invariance**: Model learns relative positions, generalizes better
2. **Higher resolution per object**: 5m in 64px = 7.8cm/pixel resolution
3. **Consistent scale**: Same pixel density regardless of environment size
4. **Focused context**: Only nearby obstacles matter for pushing

---

## 4. Output Format: Goal Masks

The model outputs a **probability mask** indicating where the object should be pushed.

### Raw Output

```
Output shape: (B, 1, 64, 64)
Value range: [-1, 1] (normalized)

After post-processing:
- Convert to [0, 1]: (output + 1) / 2
- Binarize: threshold at 0.5
- Result: Binary mask where 1 = predicted goal location
```

### Visual Example

```
┌────────────────────────────────────────────────────────────────────┐
│                    MODEL OUTPUT (64×64 mask)                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw probability mask          After binarization (>0.5)          │
│   ┌───────────────────┐         ┌───────────────────┐              │
│   │                   │         │                   │              │
│   │     ▒▒▓▓██▓▓▒     │         │       ████        │              │
│   │     ▓▓████▓▓▒     │    →    │       ████        │              │
│   │     ▒▒▓▓▓▓▒▒      │         │                   │              │
│   │                   │         │                   │              │
│   └───────────────────┘         └───────────────────┘              │
│   Soft probability              Hard binary mask                   │
│   (grayscale 0-1)               (object's predicted position)      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Diffusion Sampling Process

**Files:**
- `sage_learning/src/model/generative_module.py`
- `sage_learning/src/model/samplers/hf_diffusion_sampler.py`

The model uses **DDIM sampling** (Denoising Diffusion Implicit Models) for fast, deterministic generation.

### DDIM Sampling Algorithm

```
Algorithm: DDIM Sampling (20 steps)
─────────────────────────────────────

Input:
  - context: (B, 5, 64, 64) scene representation
  - num_steps: 20 (configurable)

Process:
  1. Initialize: x_T ~ N(0, I)  # Pure noise, shape (B, 1, 64, 64)

  2. For t = T, T-1, ..., 1:
       # Concatenate context with current noisy sample
       model_input = concat([context, x_t], dim=1)  # (B, 6, 64, 64)

       # Predict noise
       ε̂ = DiT(model_input, t/T)  # (B, 1, 64, 64)

       # DDIM update step (deterministic)
       x_{t-1} = DDIM_step(x_t, ε̂, t)

  3. Return: x_0  # Denoised goal mask

Output: (B, 1, 64, 64) in range [-1, 1]
```

### Why DDIM over DDPM?

| Aspect | DDPM | DDIM |
|--------|------|------|
| Steps needed | 1000 | 20 |
| Deterministic | No (stochastic) | Yes |
| Sample diversity | From noise | From noise |
| Speed | ~50x slower | Fast |
| Quality | Slightly better | Nearly same |

### Multiple Sample Generation

```python
# Generate 16 diverse samples from different noise initializations
goal_samples = model.sample_from_model(
    inp=context,        # (1, 5, 64, 64)
    samples=16,         # Generate 16 samples
    num_steps=20        # DDIM steps
)
# Output: (16, 1, 64, 64) - 16 different goal predictions
```

---

## 6. Output Processing: Mask to SE(2)

**File:** `sage_learning/ktamp_learning/goal_inference_model.py:403-457`

Each output mask is converted to an SE(2) pose (x, y, θ).

### Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      MASK → SE(2) CONVERSION                                  │
└──────────────────────────────────────────────────────────────────────────────┘

For each of 16 samples:

Step 1: Binarize
─────────────────
  mask = (output > 0.5).astype(uint8)

  ┌─────────────┐         ┌─────────────┐
  │  ▒▒▓▓██▓▓▒  │   →     │    ████     │
  │  ▓▓████▓▓▒  │         │    ████     │
  └─────────────┘         └─────────────┘
   Soft mask               Binary mask


Step 2: Validate (reject invalid predictions)
─────────────────────────────────────────────
  num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)

  if num_labels > 2:  # More than 1 connected region
      REJECT (multiple blobs = ambiguous prediction)

  ┌─────────────┐
  │  ██    ██   │   →  REJECTED (2 separate blobs)
  │  ██    ██   │
  └─────────────┘


Step 3: Extract rectangle center and angle
──────────────────────────────────────────
  corners, _, center, angle = find_rectangle_corners(mask)

  Uses cv2.minAreaRect() to fit minimum bounding rectangle:

  ┌─────────────┐
  │    ████     │   →  center = (px, py)
  │    ████     │      angle = 15° (from horizontal)
  └─────────────┘
       ↑
    Rectangle fit


Step 4: Convert pixel → world coordinates
─────────────────────────────────────────
  # Local mask is centered on object
  world_x, world_y = pixel_to_world_local(
      px=center[0], py=center[1],
      object_center=object_world_center,
      crop_size_meters=5.0,
      output_size=64
  )

  # Resolution: 5.0m / 64px = 0.078m per pixel


Step 5: Compute goal theta from angle difference
────────────────────────────────────────────────
  # Handle 180° ambiguity (rectangles look same at 180°)
  angle_diff_deg = predicted_angle - current_object_angle

  if angle_diff_deg > 90:
      angle_diff_deg -= 180
  elif angle_diff_deg < -90:
      angle_diff_deg += 180

  goal_theta = object_theta + radians(angle_diff_deg)


Output: SE(2) pose
──────────────────
  goal = {
      'x': world_x,      # meters
      'y': world_y,      # meters
      'theta': goal_theta  # radians
  }
```

### Rejection Criteria

| Condition | Reason | Action |
|-----------|--------|--------|
| Empty mask | Model predicted no goal | Skip sample |
| Multiple blobs | Ambiguous prediction | Skip sample |
| Blob too small | Noise artifact | Skip sample |
| Outside crop bounds | Invalid prediction | Skip sample |

---

## 7. Voting Mechanism: ML to Primitive Alignment

**File:** `namo/python/namo/strategies/primitive_goal_strategy.py:353-518`

ML predictions are **aligned to motion primitives** to ensure physical feasibility.

### Primitive Structure

Each object shape has **600 discrete primitive slots**:

```
Primitives per object:
─────────────────────
  60 edge points × 10 push depths = 600 slots

  Edge points: 15 contact points per edge × 4 edges = 60
  Push depths: 10 different push distances per contact point

  ┌───────────────────────────────────────┐
  │           OBJECT (top view)            │
  │    ┌─────────────────────────┐        │
  │    │ ● ● ● ● ● ● ● ● ● ● ● │←Edge 0  │
  │    │●                     ●│         │
  │    │●                     ●│←Edge 1  │
  │    │●                     ●│         │
  │    │●                     ●│         │
  │    │ ● ● ● ● ● ● ● ● ● ● ● │←Edge 2  │
  │    └─────────────────────────┘        │
  │         ↑                              │
  │      Edge 3                            │
  │                                        │
  │    ● = Contact point (15 per edge)     │
  │    Each ● has 10 depth options         │
  └───────────────────────────────────────┘
```

### Voting Algorithm

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VOTING MECHANISM                                    │
└──────────────────────────────────────────────────────────────────────────────┘

ML GOALS (16 samples)                    PRIMITIVE SLOTS (600 total)
────────────────────                     ───────────────────────────

┌─────────────────┐                      Edge 0:  [○○○○○○○○○○] (10 depths)
│ (2.1, 3.4, 0.2) │─────────┐            Edge 1:  [○○○○○○○○○○]
└─────────────────┘         │            Edge 2:  [○○○○○○○○○○]
┌─────────────────┐         │            ...
│ (2.0, 3.5, 0.1) │─────────┼───────────▶Edge 15: [○○●○○○○○○○] ← 3 votes
└─────────────────┘         │            ...
┌─────────────────┐         │            Edge 42: [○○○○○○○●○○] ← 5 votes
│ (2.05, 3.45, 0.15)│───────┤            ...
└─────────────────┘         │            Edge 59: [○○○○○○○○○○]
...                         │
┌─────────────────┐         │
│ (1.8, 2.9, -0.3)│─────────┘
└─────────────────┘

Algorithm:
──────────
For each ML goal:
    1. Find closest primitive slot:
       score = position_error + angle_weight × angle_error
       best_slot = argmin(score) over all 600 slots

    2. Increment vote count for best_slot:
       slot_votes[best_slot] += 1

Result:
───────
Sparse grid where voted slots have counts:
  aligned_goals[edge_idx][depth_idx] = Goal(x, y, θ, score=vote_count)

Example:
  Edge 15, Depth 3: Goal(x=2.02, y=3.48, θ=0.12, score=3)  ← 3 ML samples agreed
  Edge 42, Depth 8: Goal(x=1.85, y=2.95, θ=-0.28, score=5) ← 5 ML samples agreed
```

### Why Voting?

1. **Consensus**: Multiple samples agreeing = higher confidence
2. **Robustness**: Outlier predictions get few votes
3. **Priority ordering**: Search tries high-vote goals first
4. **Physical feasibility**: All executed goals are valid primitives

---

## 8. Search Algorithm: Region Opening BFS

**File:** `namo/python/namo/planners/opening/region_opening.py`

The search tries goals in **priority order** (highest votes first).

### Algorithm Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    REGION OPENING BFS SEARCH                                  │
└──────────────────────────────────────────────────────────────────────────────┘

For each neighbor region:

1. PRE-CHECK
   ──────────
   Is neighbor already accessible?
   → Check if ≥50% of region goals are reachable via wavefront
   → If YES: Record "already_accessible", skip to next neighbor

2. GET CANDIDATES
   ───────────────
   candidates = objects blocking robot↔neighbor edge
   candidates = filter to reachable objects only

3. FOR EACH CANDIDATE OBJECT:
   ──────────────────────────

   a) Generate ML goals (16 samples from diffusion model)

   b) Align to primitives (voting mechanism)

   c) Get reachable edges (which primitive edges robot can access)

   d) Build candidate list:
      candidates = [(edge_idx, depth, goal) for all aligned goals on reachable edges]

   e) Sort by priority:
      candidates.sort(key=lambda x: (-x.score, x.depth))
      # Primary: highest votes first
      # Secondary: shortest push first

   f) BFS over candidates:
      for (edge_idx, depth, goal) in candidates:
          # Execute push in simulator
          action = Action(object_id, goal.x, goal.y, goal.theta)
          env.step(action)

          # Check if opening was created
          if is_neighbor_now_accessible():
              RECORD SUCCESS
              return
          else:
              # Restore state and try next goal
              env.set_full_state(saved_state)

4. RECORD RESULT
   ──────────────
   Success: Save action sequence, observations, stats
   Failure: Save failure_reason, ML stats for debugging
```

### Search Priority

Goals are tried in this order:

```
Priority 1: Vote count (descending)
─────────────────────────────────────
  Goal with 5 votes tried before goal with 3 votes
  (Higher ML model confidence)

Priority 2: Push depth (ascending)
─────────────────────────────────────
  Among equal votes, shorter pushes first
  (More likely to succeed, less risk)

Example order:
  1. Edge 42, Depth 2, votes=5  ← Tried first
  2. Edge 15, Depth 1, votes=3
  3. Edge 15, Depth 4, votes=3
  4. Edge 28, Depth 3, votes=2
  ...
```

---

## 9. Failure Tracking

**File:** `namo/python/namo/planners/opening/region_opening.py:75-92`

The system tracks **why** each attempt succeeded or failed.

### Failure Reason Categories

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FAILURE REASON TAXONOMY                               │
└──────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │  Attempt    │
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
       ┌──────────┐           ┌──────────┐           ┌──────────┐
       │ SUCCESS  │           │ 0 PUSHES │           │ N PUSHES │
       └──────────┘           └────┬─────┘           └────┬─────┘
              │                    │                      │
              │         ┌──────────┼──────────┐          │
              │         │          │          │          │
              ▼         ▼          ▼          ▼          ▼
         "success"  "already_   "no_      "no_      "all_pushes_
                    accessible" blocking  reachable  failed"
                                objects"  objects"
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                         ▼          ▼          ▼
                    "ml_no_    "ml_goals_  "no_reachable_
                    goals_     not_        edges"
                    extracted" aligned"
```

### Failure Reason Reference

| `failure_reason` | Pushes | Meaning | Diagnostic |
|------------------|--------|---------|------------|
| `"success"` | N>0 | Found valid opening | - |
| `"already_accessible"` | 0 | Neighbor already reachable | Not an error |
| `"no_blocking_objects"` | 0 | No objects on edge | Region detection issue |
| `"no_reachable_objects"` | 0 | Can't reach blocking objects | Robot trapped |
| `"ml_no_goals_extracted"` | 0 | ML produced empty masks | Model quality issue |
| `"ml_goals_not_aligned"` | 0 | ML goals far from primitives | Model/primitive mismatch |
| `"no_reachable_edges"` | 0 | Goals on unreachable edges | Reachability issue |
| `"no_valid_goals"` | 0 | Fallback: unknown 0-push | Debug needed |
| `"all_pushes_failed"` | N>0 | Pushes didn't create opening | Hard environment |
| `"timeout"` | varies | Search exceeded time limit | Increase timeout |

### Stats Saved with Each Attempt

```python
algorithm_stats = {
    # Core stats
    'failure_reason': 'ml_goals_not_aligned',
    'pushes_total_for_neighbour': 0,
    'solutions_total_for_neighbour': 0,

    # ML debugging stats
    'ml_goals_generated': 16,      # Raw ML outputs
    'ml_goals_aligned': 0,         # After primitive alignment
    'reachable_edges_count': 45,   # Edges robot could access
    'candidate_objects_count': 3,   # Blocking objects tried

    # Other
    'neighbour_region_label': 'region_2',
    'validation_method': 'reachability_validated',
    ...
}
```

---

## 10. Configuration Reference

**File:** `namo/python/namo/data_collection/region_opening_ml_collection.yaml`

### Key Parameters

```yaml
# ═══════════════════════════════════════════════════════════════════════════
# ML MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

goal_sampler: ml                    # Use ML model (vs 'primitive' for baseline)
ml_goal_model: /path/to/checkpoint  # Model checkpoint directory
ml_device: cuda                     # GPU device
ml_samples: 16                      # Number of diffusion samples per object
ml_num_steps: 20                    # DDIM integration steps
ml_sampler_method: ddim             # Sampler: ddim (fast) or ddpm (slow)

# Primitive alignment (for stats tracking, no hard filtering)
ml_match_position_tolerance: 0.05   # meters
ml_match_angle_tolerance: 0.1       # radians (~5.7°)
ml_match_angle_weight: 1.0          # Weight of angle vs position in scoring

# ═══════════════════════════════════════════════════════════════════════════
# SEARCH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

region_max_chain_depth: 1           # Max pushes per solution (1 = single push)
region_max_solutions_per_neighbor: 1  # Solutions to find per neighbor
goals_per_region: 10                # Robot goal samples for validation

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

algorithm: region_opening           # Planner algorithm
workers: 1                          # Sequential execution
search_timeout: 300.0               # Max seconds per environment
```

### Understanding Parameter Impact

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| `ml_samples` | More diverse goals, slower | Fewer options, faster |
| `ml_num_steps` | Better quality, slower | Faster, slightly noisier |
| `goals_per_region` | More robust validation | Faster validation |
| `region_max_chain_depth` | Can solve harder cases | Faster, simpler solutions |

---

## Quick Reference: Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE INFERENCE PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

1. LOAD MODEL
   GoalInferenceModel(checkpoint_path, device='cuda', sampler='ddim', steps=20)

2. FOR EACH ENVIRONMENT:
   ├── Parse XML → identify regions, edges, blocking objects
   │
   ├── FOR EACH NEIGHBOR REGION:
   │   ├── Pre-check: is neighbor already accessible?
   │   │   └── If yes → record "already_accessible", continue
   │   │
   │   ├── Get candidate objects (blocking & reachable)
   │   │
   │   ├── FOR EACH CANDIDATE OBJECT:
   │   │   │
   │   │   ├── CREATE INPUT (5 channels, 64×64):
   │   │   │   [static, movable, target_object, robot_region, goal_region]
   │   │   │
   │   │   ├── DIFFUSION SAMPLING (DDIM, 20 steps):
   │   │   │   x_T ~ N(0,1) → denoise → x_0 (16 samples)
   │   │   │
   │   │   ├── MASK → SE(2):
   │   │   │   Binarize → find rectangle → pixel to world → 16 goals
   │   │   │
   │   │   ├── PRIMITIVE VOTING:
   │   │   │   Each ML goal votes for closest primitive slot
   │   │   │   Result: sparse aligned_goals[60][10] with vote counts
   │   │   │
   │   │   ├── BFS SEARCH (priority order):
   │   │   │   Sort by (-votes, depth) → try each goal → check if opening created
   │   │   │
   │   │   └── If success → record solution, break
   │   │
   │   └── Record attempt result (success/failure + stats)
   │
   └── Save all results to pickle file

3. OUTPUT:
   Per-neighbor results with:
   - success/failure status
   - failure_reason (for debugging)
   - action_sequence (if success)
   - ML stats (goals_generated, goals_aligned, etc.)
```
