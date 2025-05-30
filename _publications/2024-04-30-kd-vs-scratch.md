---
title: "Knowledge Distillation vs. Pretraining from Scratch under a Fixed (Computation) Budget"
collection: publications
permalink: /publication/2024-04-30-kd-vs-scratch
excerpt: ''
date: 2024-04-30
authors: Minh Duc Bui, Fabian David Schmidt, Goran Glavaš, Katharina von der Wense
conference: '5th Workshop on Insights from Negative Results in NLP @NAACL 2024'
paperurl: 'https://aclanthology.org/2024.insights-1.6/'
---

Compared to standard language model (LM) pretraining (i.e., from scratch), Knowledge Distillation (KD) entails an additional forward pass through a teacher model that is typically substantially larger than the target student model. As such, KD in LM pretraining materially slows down throughput of pretraining instances vis-a-vis pretraining from scratch. Scaling laws of LM pretraining suggest that smaller models can close the gap to larger counterparts if trained on more data (i.e., processing more tokens)-and under a fixed computation budget, smaller models are able be process more data than larger models. We thus hypothesize that KD might, in fact, be suboptimal to pretraining from scratch for obtaining smaller LMs, when appropriately accounting for the compute budget. To test this, we compare pretraining from scratch against several KD strategies for masked language modeling (MLM) in a fair experimental setup, with respect to amount of computation as well as pretraining data. Downstream results on GLUE, however, do not confirm our hypothesis: while pretraining from scratch performs comparably to ordinary KD under a fixed computation budget, more sophisticated KD strategies, namely TinyBERT (Jiao et al., 2020) and MiniLM (Wang et al., 2023), outperform it by a notable margin. We further find that KD yields larger gains over pretraining from scratch when the data must be repeated under the fixed computation budget.

[**Paper-Link**](https://aclanthology.org/2024.insights-1.6/)
