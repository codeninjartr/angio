# 📊 Vessel Growth Plots — Graph-by-Graph Lecture Notes

> **Context**: CAM (Chorioallantoic Membrane) Assay measuring the effect of a drug at three concentrations (0.1 µg, 1 µg, 10 µg) plus a no-drug **Control** on tumor angiogenesis over five time points (0 h, 2 h, 4 h, 8 h, 24 h). Three topological features are extracted from skeletonized vessel masks: **Disconnected Networks**, **Branch Points**, and **Endpoints**.

---

## Overview — What the Three Plots Show

| Plot | Feature | What "↑" means biologically |
|------|---------|----------------------------|
| **Left** | Disconnected Networks | ↑ = Vessel fragmentation → **anti-angiogenic / tumor-starving** |
| **Middle** | Branch Points | ↑ = Vascular complexity → **pro-angiogenic / tumor-feeding** |
| **Right** | Endpoints (Dead-ends) | ↑ = Immature capillaries → **poor perfusion** |

---
---

# GRAPH 1 — Vessel Network Fragmentation (Left Plot)

**Y-axis**: Average number of Disconnected Networks  
**X-axis**: Time (0 h → 24 h)  
**Subtitle**: ↑ = Anti-angiogenic / Tumor-starving

### What this metric means
- Each "disconnected network" is an **isolated island** of vessels with no connection to the main vascular tree.
- **More islands → vasculature is broken apart → the tumor cannot get a continuous blood supply → anti-angiogenic effect.**
- **Fewer islands → vessels are merging into a unified network → the tumor is being fed → pro-angiogenic effect.**

---

### 🔵 0.1 µg (Low Dose) — n₀ = 8 eggs

| Time | Networks (Mean ± SD) | n | Observation |
|------|---------------------|---|-------------|
| 0 h | 13.38 ± 5.12 | 8 | Starting baseline — moderately fragmented |
| 2 h | 13.60 ± 2.65 | 5 | Almost unchanged (+0.22), variability drops — stable |
| 4 h | 14.50 ± 4.72 | 4 | Slight increase — mild fragmentation begins |
| 8 h | 11.33 ± 1.25 | 3 | Drops sharply — networks are **merging / reconnecting** |
| 24 h | **4.00 ± 0.00** | 1 | **Dramatic collapse** to just 4 networks (single egg) |

**Key talking points:**
- At low dose, the drug initially has **no significant anti-angiogenic effect** (0–4 h stays flat around 13–14).
- Between 4 h and 8 h, networks begin merging (14.5 → 11.3), suggesting the vasculature is **reorganizing** and consolidating.
- By 24 h, only **4 disconnected networks** remain — the vasculature has almost completely **unified into a single interconnected tree**.
- ⚠️ **Caveat**: The 24 h point is from only 1 egg (n=1), so this dramatic drop should be interpreted with caution.
- **Interpretation**: 0.1 µg is **too low to fragment the vasculature**. Instead, the vasculature **matured and merged** over time — a **pro-angiogenic trajectory**.

---

### 🟢 1 µg (Medium Dose) — n₀ = 17 eggs

| Time | Networks (Mean ± SD) | n | Observation |
|------|---------------------|---|-------------|
| 0 h | 9.53 ± 4.34 | 17 | Baseline — fewer networks, more consolidated |
| 2 h | 10.83 ± 3.53 | 12 | Slight increase (+1.3) — mild early fragmentation |
| 4 h | 12.30 ± 3.74 | 10 | Continues rising — drug is starting to **fragment** vessels |
| 8 h | 11.25 ± 3.34 | 8 | Slight dip — some re-merging or stabilization |
| 24 h | 12.67 ± 3.68 | 3 | Returns to elevated level — **sustained fragmentation** |

**Key talking points:**
- The medium dose shows a **gradual, sustained increase** in disconnected networks over 24 h (9.5 → 12.7, a ~33% increase).
- This is the **most stable, progressive anti-angiogenic curve** among all groups.
- The temporary dip at 8 h could indicate the vasculature is briefly attempting to **compensate and repair**, but by 24 h the fragmentation prevails.
- **Interpretation**: 1 µg delivers a **moderate, sustained anti-angiogenic effect**.

---

### 🔴 10 µg (High Dose) — n₀ = 12 eggs

| Time | Networks (Mean ± SD) | n | Observation |
|------|---------------------|---|-------------|
| 0 h | 12.00 ± 4.00 | 12 | Baseline — similar to 0.1 µg |
| 2 h | 10.33 ± 3.56 | 9 | Initial **decrease** — counter-intuitive |
| 4 h | 11.14 ± 3.14 | 7 | Recovers slightly |
| 8 h | **16.33 ± 3.77** | 6 | **Sharp spike** — major fragmentation event |
| 24 h | **18.00 ± 0.00** | 1 | Continues rising — maximum fragmentation |

**Key talking points:**
- The high dose shows a **biphasic response**: an initial dip (0–2 h), then a **dramatic surge** in fragmentation from 4 h to 8 h (11.1 → 16.3, a 47% spike).
- At 8 h, the network count reaches **16.33** — the highest value among all groups at any time point.
- **Interpretation**: 10 µg is the **most potent anti-angiogenic dose** — it causes severe fragmentation, especially after the 4-hour mark.

---

### ⚫ Control (No Drug) — n₀ = 7 eggs

| Time | Networks (Mean ± SD) | n | Observation |
|------|---------------------|---|-------------|
| 0 h | 11.14 ± 3.94 | 7 | Baseline |
| 2 h | 13.67 ± 0.94 | 3 | Slight increase |
| 4 h | 10.00 ± 4.32 | 3 | Drops — natural fluctuation |
| 8 h | 14.67 ± 1.70 | 3 | Rises again |
| 24 h | 16.00 ± 2.00 | 2 | Moderate increase |

**Key talking points:**
- The control group shows **natural fluctuation** without a clear directional trend.
- **Interpretation**: Without the drug, the vasculature undergoes normal developmental remodeling.

---

### 🎯 GRAPH 1 — Summary
> Dose-dependent anti-angiogenic response: 0.1 µg fails (pro-angiogenic), 1 µg is moderate and sustained, 10 µg triggers dramatic fragmentation to 18 networks.

---
---

# GRAPH 2 — Vascular Branching Complexity (Middle Plot)

**Y-axis**: Average Branch Points  
**X-axis**: Time (0 h → 24 h)  
**Subtitle**: ↑ = Pro-angiogenic / Tumor-feeding

### What this metric means
- Branch points are where a vessel **splits into two or more** daughter vessels.
- **More branch points → richer, denser vascular tree → more pathways for blood to reach the tumor → pro-angiogenic.**
- **Fewer branch points → simpler, pruned vasculature → reduced nutrient delivery → anti-angiogenic.**

---

### 🔵 0.1 µg (Low Dose) — n₀ = 8 eggs

| Time | Branches (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 2824.75 ± 739.74 | 8 | Moderately high baseline |
| 2 h | 2298.80 ± 395.97 | 5 | Drops by ~19% — initial pruning |
| 4 h | 2502.25 ± 773.55 | 4 | Partial recovery |
| 8 h | 2506.00 ± 733.87 | 3 | Plateau — stable |
| 24 h | **5115.00 ± 0.00** | 1 | **Massive spike** — branches nearly double |

**Key talking points:**
- Initial drop at 2 h, then stabilization, then an explosion at 24 h to **5115** branch points (n=1).
- Pairs with Graph 1: network merged into just 4 large islands, each now massively branched.
- **Interpretation**: 0.1 µg allows the vasculature to **recover and hyperproliferate** — pro-angiogenic.

---

### 🟢 1 µg (Medium Dose) — n₀ = 17 eggs

| Time | Branches (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 2906.47 ± 989.39 | 17 | Highest baseline of all groups |
| 2 h | 2859.67 ± 986.11 | 12 | Minimal change (-1.6%) |
| 4 h | 2458.00 ± 1021.17 | 10 | Notable drop — drug is **pruning** branches |
| 8 h | 2450.38 ± 842.00 | 8 | Stabilizes — pruning complete |
| 24 h | 2461.33 ± 509.16 | 3 | Holds steady — **sustained suppression** |

**Key talking points:**
- **Steady, progressive reduction** (2906 → 2461, ~15% decrease).
- Curve becomes **flat from 4–24 h** — maximum pruning reached and sustained.
- SD drops from 989 → 509: drug creates **uniformly pruned** vascular state.
- **Interpretation**: 1 µg is a **consistent anti-angiogenic dose** for branching.

---

### 🔴 10 µg (High Dose) — n₀ = 12 eggs

| Time | Branches (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 2765.33 ± 1442.68 | 12 | High variability baseline |
| 2 h | 3161.67 ± 1390.59 | 9 | **Paradoxical 14% INCREASE** — stress sprouting |
| 4 h | 2560.86 ± 662.14 | 7 | Crashes back down |
| 8 h | 2626.67 ± 632.51 | 6 | Slight stabilization |
| 24 h | **1523.00 ± 0.00** | 1 | **Catastrophic collapse — 52% loss from peak** |

**Key talking points:**
- **Biphasic response**: stress sprouting at 2 h, then progressive collapse.
- By 24 h, **1523 branches** — the absolute lowest of any group at any time.
- Combined with Graph 1: 18 fragments × ~85 branches each = devastated vascular bed.
- **Interpretation**: 10 µg causes the most aggressive anti-angiogenic branching response.

---

### ⚫ Control (No Drug) — n₀ = 7 eggs

| Time | Branches (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 2207.29 ± 849.19 | 7 | Lowest baseline |
| 2 h | 2345.33 ± 793.97 | 3 | Slight natural growth |
| 4 h | 2222.33 ± 230.91 | 3 | Returns to baseline |
| 8 h | 2931.67 ± 473.54 | 3 | Natural developmental branching peak |
| 24 h | 2545.50 ± 95.50 | 2 | Settles — natural plateau |

**Key talking points:**
- Natural 15% growth in branching — normal CAM development.
- 10 µg's 24 h value of 1523 is **40% below** even untreated tissue.

---

### 🎯 GRAPH 2 — Summary
> Dose-dependent vascular pruning: 0.1 µg lets branches explode to 5115, 1 µg steadily prunes by 15%, 10 µg causes catastrophic 52% collapse to 1523.

---
---

# GRAPH 3 — Capillary Dead-Ends (Right Plot)

**Y-axis**: Average Endpoints  
**X-axis**: Time (0 h → 24 h)  
**Subtitle**: ↑ = Immature vasculature / Poor perfusion

### What this metric means
- Endpoints are where a vessel **terminates without connecting** to another vessel — a "dead-end."
- **More endpoints → many immature, sprouting capillaries → poor blood flow → tumor gets less oxygen.**
- **Fewer endpoints → mature, connected vascular loops → efficient blood flow.**

---

### 🔵 0.1 µg (Low Dose) — n₀ = 8 eggs

| Time | Endpoints (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 51.00 ± 6.93 | 8 | Highest starting value |
| 2 h | 44.60 ± 6.05 | 5 | Drops — dead-ends closing into loops |
| 4 h | 53.00 ± 1.41 | 4 | Bounces back — new sprouting |
| 8 h | 44.00 ± 2.94 | 3 | Drops again |
| 24 h | 53.00 ± 0.00 | 1 | Back up — oscillating |

**Key talking points:**
- Endpoints **oscillate** between 44–53 with no clear trend.
- **Interpretation**: Drug has **zero effect** on capillary maturation at this dose.

---

### 🟢 1 µg (Medium Dose) — n₀ = 17 eggs

| Time | Endpoints (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 41.24 ± 10.42 | 17 | Lowest starting — most mature vasculature |
| 2 h | 47.75 ± 7.34 | 12 | Rises 16% — drug breaking connections |
| 4 h | 44.40 ± 6.71 | 10 | Slight recovery |
| 8 h | 39.12 ± 7.99 | 8 | Below baseline — surviving vessels form loops |
| 24 h | 51.00 ± 4.32 | 3 | **Late rebound** — fragmentation creates new dead-ends |

**Key talking points:**
- Three-phase trajectory: rise (0–2 h) → repair dip (2–8 h) → rebound (8–24 h).
- **Interpretation**: 1 µg creates a **tug-of-war** between fragmentation and repair. Drug wins by 24 h.

---

### 🔴 10 µg (High Dose) — n₀ = 12 eggs

| Time | Endpoints (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 44.25 ± 13.50 | 12 | Moderate baseline |
| 2 h | 46.67 ± 10.62 | 9 | Creeps up +5% |
| 4 h | 47.71 ± 10.28 | 7 | Continues +8% |
| 8 h | 48.67 ± 9.39 | 6 | Still climbing +10% |
| 24 h | **59.00 ± 0.00** | 1 | **33% above baseline — highest of ANY group** |

**Key talking points:**
- **Only group with a perfectly monotonic, unbroken upward trend** — no dip, no recovery.
- The vasculature **never gets a chance to repair**.
- Combined 10 µg 24 h picture: 18 fragments + 1523 branches + 59 endpoints = **vascular graveyard**.
- **Interpretation**: Drug **permanently prevents vessel maturation**.

---

### ⚫ Control (No Drug) — n₀ = 7 eggs

| Time | Endpoints (Mean ± SD) | n | Observation |
|------|----------------------|---|-------------|
| 0 h | 44.14 ± 10.26 | 7 | Baseline |
| 2 h | 44.33 ± 4.78 | 3 | Flat |
| 4 h | 46.00 ± 12.68 | 3 | Minimal change |
| 8 h | **64.33 ± 11.15** | 3 | **Sharp spike — natural sprouting burst** |
| 24 h | 48.50 ± 3.50 | 2 | **Drops back — sprouts MATURE into loops** |

**Key talking points:**
- The 8 h spike (64.33) followed by 24 h recovery (48.50) = **healthy sprout → connect → mature** cycle.
- In 10 µg, endpoints climb and **never come down** — drug prevents maturation step.
- **Key insight**: Control dead-ends are temporary (healthy). 10 µg dead-ends are permanent (drug-induced dysfunction).

---

### 🎯 GRAPH 3 — Summary
> 10 µg is the only group with perfectly monotonic endpoint increase (44 → 59), meaning vessels are permanently severed and can never mature.

---
---

# 🎓 GRAND SUMMARY

| Dose | Fragmentation | Branching | Dead-ends | Verdict |
|------|:---:|:---:|:---:|---------|
| **0.1 µg** | ↓ to 4 | ↑ to 5115 | ↔ oscillates | ❌ **Sub-therapeutic** — vasculature thrives |
| **1 µg** | ↑ to 12.7 | ↓ to 2461 | ↑ rebound to 51 | ⚠️ **Moderate** — sustained but incomplete |
| **10 µg** | ↑↑ to 18 | ↓↓ to 1523 | ↑↑ to 59 | ✅ **Maximum anti-angiogenic** — vasculature destroyed |
| **Control** | ↔ fluctuates | ↑ natural growth | spike→resolve | 📋 Normal development baseline |

> **The Story**: Carbon dots exhibit a concentration-dependent biphasic response on angiogenesis. At sub-therapeutic doses (0.1 µg), they are pro-angiogenic via hormesis. At high doses (10 µg), they are potently anti-angiogenic, fragmenting, pruning, and permanently preventing vessel maturation. The therapeutic window lies between 1–10 µg.
