---
title: "The Trade-off between Performance, Efficiency, and Fairness in Adapter Modules for Text Classification"
collection: publications
permalink: /publication/2024-05-04-fairness-adapter-modules
excerpt: ''
authors: Minh Duc Bui, Katharina von der Wense
conference: '4th Workshop on Trustworthy Natural Language Processing (TrustNLP) @NAACL 2024'
paperurl: 'https://aclanthology.org/2024.trustnlp-1.4/'
---

Current natural language processing (NLP) research tends to focus on only one or, less frequently, two dimensions - e.g., performance, privacy, fairness, or efficiency - at a time, which may lead to suboptimal conclusions and often overlooking the broader goal of achieving trustworthy NLP. Work on adapter modules focuses on improving performance and efficiency, with no investigation of unintended consequences on other aspects such as fairness. To address this gap, we conduct experiments on three text classification datasets by either (1) finetuning all parameters or (2) using adapter modules. Regarding performance and efficiency, we confirm prior findings that the accuracy of adapter-enhanced models is roughly on par with that of fully finetuned models, while training time is substantially reduced. Regarding fairness, we show that adapter modules result in mixed fairness across sensitive groups. Further investigation reveals that, when the standard fine-tuned model exhibits limited biases, adapter modules typically do not introduce extra bias. On the other hand, when the finetuned model exhibits increased bias, the impact of adapter modules on bias becomes more unpredictable, introducing the risk of significantly magnifying these biases for certain groups. Our findings highlight the need for a case-by-case evaluation rather than a one-size-fits-all judgment.

[**Paper-Link**](https://aclanthology.org/2024.trustnlp-1.4/)

