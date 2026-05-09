Baseline Locutusque/TinyMistral-248M:-
Perplexity:         35.1653
--- Structural Metrics ---
ROUGE-1:            0.0766
ROUGE-2:            0.0155
ROUGE-L:            0.0696
BLEU:               0.0096
METEOR:             0.0751
Accuracy (EM):      0.0000
--- Semantic Metrics ---
Feature Sim (F1):   0.7827 (BERTScore)
--- Performance Metrics ---
Avg Latency:        2.33 s/prompt
Speed:              55.09 tokens/s


Supervised Fine-Tuned (SFT) Locutusque/TinyMistral-248M:-
Perplexity:         31.5570
--- Structural Metrics ---
ROUGE-1:            0.1470
ROUGE-2:            0.0528
ROUGE-L:            0.1333
BLEU:               0.0084
METEOR:             0.1091
Accuracy (EM):      0.0000
--- Semantic Metrics ---
Feature Sim (F1):   0.8076 (BERTScore)
--- Performance Metrics ---
Avg Latency:        2.76 s/prompt
Speed:              46.37 tokens/s


Supervised Fine-Tuned (SFT-soft) Locutusque/TinyMistral-248M:-
Perplexity:         36.4003
--- Structural Metrics ---
ROUGE-1:            0.0859
ROUGE-2:            0.0215
ROUGE-L:            0.0786
BLEU:               0.0161
METEOR:             0.0743
Accuracy (EM):      0.0000
--- Semantic Metrics ---
Feature Sim (F1):   0.7677 (BERTScore)
--- Performance Metrics ---
Avg Latency:        2.55 s/prompt
Speed:              50.33 tokens/s


KD + SFT TinyMistral-248M:-
Perplexity:         38.0067
KL Divergence:      2.1496
--- Structural Metrics ---
ROUGE-1:            0.1770
ROUGE-2:            0.0642
ROUGE-L:            0.1435
BLEU:               0.0088
METEOR:             0.1293
Accuracy (EM):      0.0000
--- Semantic Metrics ---
Feature Sim (F1):   0.8280 (BERTScore)
--- Performance Metrics ---
Compression Ratio:  15.13x
Avg Latency:        0.88 s/prompt
Speed:              48.62 tokens/s


KD + SFT (Soft) TinyMistral-248M:-
Perplexity:         38.6804
KL Divergence:      2.1747
--- Structural Metrics ---
ROUGE-1:            0.1771
ROUGE-2:            0.0591
ROUGE-L:            0.1469
BLEU:               0.0127
METEOR:             0.1328
Accuracy (EM):      0.0000
--- Semantic Metrics ---
Feature Sim (F1):   0.8370 (BERTScore)
--- Performance Metrics ---
Compression Ratio:  15.13x
Avg Latency:        0.94 s/prompt
Speed:              50.03 tokens/s