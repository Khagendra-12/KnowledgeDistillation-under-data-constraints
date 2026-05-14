## Evaluation Metrics

| Metric | Baseline | SFT | SFT-Soft (SSFT) | KD + SFT (Soft) | KD + SFT |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | 0.0766 | 0.1470 | 0.0859 | 0.1771 | **0.1886** |
| ROUGE-2 | 0.0155 | 0.0528 | 0.0215 | 0.0591 | **0.0649** |
| ROUGE-L | 0.0696 | 0.1333 | 0.0786 | 0.1469 | **0.1578** |
| BLEU | 0.0096 | 0.0084 | 0.0161 | **0.0127** | **0.0127** |
| METEOR | 0.0751 | 0.1091 | 0.0743 | 0.1328 | **0.1427** |
| BERTScore (F1) | 0.7827 | 0.8076 | 0.7677 | **0.8370** | 0.8300 |
| Perplexity | 35.1653 | **31.5570** | 36.4003 | 38.6804 | 40.3721 |
| Speed (tokens/s) | **55.09** | 46.37 | 50.33 | 50.03 | 50.68 |
| Avg Latency (s/prompt) | 2.33 | 2.76 | 2.55 | 0.94 | **0.90** |
| Compression Ratio | -- | -- | -- | 15.13x | 15.13x |
| KL Divergence | -- | -- | -- | 2.1747 | **2.2088** |

---

## Percentage Change Compared to Baseline

| Metric | Baseline | SFT | SFT-Soft (SSFT) | KD + SFT (Soft) | KD + SFT |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | -- | ↑ 91.9% | ↑ 12.1% | ↑ 131.2% | **↑ 146.2%** |
| ROUGE-2 | -- | ↑ 240.6% | ↑ 38.7% | ↑ 281.3% | **↑ 318.7%** |
| ROUGE-L | -- | ↑ 91.5% | ↑ 12.9% | ↑ 111.1% | **↑ 126.7%** |
| BLEU | -- | ↓ 12.5% | ↑ 67.7% | ↑ 32.3% | ↑ 32.3% |
| METEOR | -- | ↑ 45.3% | ↓ 1.1% | ↑ 76.8% | **↑ 90.0%** |
| BERTScore (F1) | -- | ↑ 3.2% | ↓ 1.9% | **↑ 6.9%** | ↑ 6.0% |
| Perplexity | -- | **↓ 10.3%** | ↑ 3.5% | ↑ 10.0% | ↑ 14.8% |
| Speed (tokens/s) | -- | ↓ 15.8% | ↓ 8.6% | ↓ 9.2% | ↓ 8.0% |
| Avg Latency (s/prompt) | -- | ↑ 18.5% | ↑ 9.4% | ↓ 59.7% | **↓ 61.4%** |

---

## Key Findings

### 1. Knowledge Distillation is Effective Under Limited Data Constraints
Both distilled variants significantly outperform their non-distilled counterparts despite being trained on the same constrained dataset.

Compared to standard SFT:

- **KD + SFT**
  - ROUGE-L: +18.4%
  - METEOR: +30.8%
  - BERTScore: +2.8%

Compared to Soft SFT:

- **KD + SFT (Soft)**
  - ROUGE-L: +86.9%
  - METEOR: +78.7%
  - BERTScore: +9.0%

This demonstrates that logit-based knowledge distillation remains highly effective even in low-data regimes.

---

### 2. Instruction Adaptation Significantly Improves Distillation Readiness
The raw baseline model performs poorly across all quality metrics.

Baseline → KD + SFT:

- ROUGE-L: +126.7%
- METEOR: +90.0%
- BERTScore: +6.0%

This suggests that staged adaptation allows compact non-instruction-tuned language models to better absorb teacher knowledge.

---

### 3. Soft Supervision Improves Semantic Alignment
Although **KD + SFT** achieves the strongest structural performance, the soft-supervised distilled model achieves the best semantic similarity.

**KD + SFT (Soft):**
- Highest BERTScore: **0.8370**
- Lower KL divergence: **2.1747**
- Better perplexity than KD + SFT

This suggests softer supervision may improve semantic generalization and contextual alignment.

---

### 4. Efficiency Gains Are Preserved
Despite substantial quality gains:

- Compression ratio: **15.13×**
- Latency reduction:
  - KD + SFT: **61.4% faster**
  - KD + SFT (Soft): **59.7% faster**

This demonstrates practical deployability gains alongside performance improvements.

---

## KL Divergence Analysis

The KL divergence values for both distilled models are relatively close, indicating stable teacher-student distribution alignment during distillation.

| Model | KL Divergence |
|---|---:|
| KD + SFT (Soft) | **2.1747** |
| KD + SFT | 2.2088 |

Interpretation:

- **KD + SFT** achieves stronger lexical and structural overlap.
- **KD + SFT (Soft)** achieves better semantic similarity and lower distribution divergence.

This suggests a structural vs semantic optimization trade-off between hard and softened pre-distillation supervision.
