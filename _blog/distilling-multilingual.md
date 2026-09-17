---
title: "[Project]  Distilling Multilingual Encoders into Monolingual Components"
date: 2022-08-04 12:30:47 +01:00
tags: [project, nlp]
description: Master Thesis
image: "/assets/img/cross-lingual/introduction.jpg"
---
*In this blog post, we motivate and lay out my main contributions of my Master Thesis. To read the full thesis, 
download my thesis <a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>. To 
read the related work sections as blog posts, 
see [Cross-Lingual Representation](https://minhducbui.github.io/cross-lingual-representation/), [Parameter-Efficient Fine-Tuning in NLP](https://minhducbui.github.io/efficient-tuning/)
and [Knowledge Distillation](https://minhducbui.github.io/knowledge-distillation/).*


<figure>
<img src="/assets/img/distilling-multilingual/introduction.jpg" alt="interview-img">
</figure>


# Table of Contents
1. [Motivation](#1-motivation)
2. [Research Objective & Contribution](#2-research-objective---contribution)


# 1. Motivation

Natural language processing (NLP) has made significant progress in
recent years, achieving impressive performances across diverse tasks.
However, these advances are focused on just a tiny fraction of the 7000
languages in the world, e.g., English, where sufficient amounts of text
in the respective language are available (high-resource languages).
Nevertheless, when the situation arises where text data in a language is
scarce, language technologies fail. These languages are called
low-resource languages, e.g., Swahili, Basque, or Urdu; see the next Figure for a more apparent distinction between
high-, mid-, and low-resource languages:

<figure>
<img src="/assets/img/master-thesis/nlp_resource_hierarchy.png" alt="interview-img">
<em>A conceptual view of the NLP resource hierarchy categorised in
availability of task-specific labels and availability of unlabeled
language-specific text. Taken from (<a href="#ref-ruder2019unsupervised">Ruder et al. (2019)</a>).</em>
</figure>

This leaves low-resource languages and, therefore, most languages
understudied, which further increases the digital language divide[^1] on
a technological level. Being able to develop technologies for
low-resource languages is vital for scientific, social, and economic
reasons, e.g., Africa and India are the hosts of around 2000
low-resource languages and are home to more than 2.5 billion inhabitants
(<a href="#ref-magueresse2020lowresource">Magueresse et al. (2020)</a>). \"Opening\" the newest NLP technologies
for low-resource languages can help bridge the gap, e.g., digital
assistants, or help reduce the discrimination against speakers of
non-English languages
(<a href="#ref-tatman-2017-gender">Tatman (2017)</a>; <a href="#ref-rabinovich-etal-2018-native">Rabinovich et al. (2018)</a>; <a href="#ref-Zhiltsova2019MitigationOU">Zhiltsova et al. (2019)</a>).

To improve language technologies for low-resource languages, the field
of Cross-Lingual Representation Learning is focused on creating
high-quality representations for these languages by gaining benefit from
abundant data in another language via a shared representation space. As
static word representations gained in popularity, many multilingual
embedding methods have been presented
(<a href="#ref-mikolov2013exploiting">Mikolov et al. (2013)</a>; <a href="#ref-hermann-blunsom-2014-multilingual">Hermann et al. (2014)</a>; <a href="#ref-hu2020xtreme">Hu et al. (2020)</a>).
The idea behind these methods is to induce embeddings[^2] such that the
embeddings for two languages are aligned, i.e., word translations, e.g.,
*cat* and *Katze*, have similar representations. Recently, however,
large pre-trained language models, the so-called transformer models,
took static word embedding methods over in virtually every aspect,
partly because these models induce context-dependent word
representations, capturing the rich meaning of a word better
(<a href="#ref-peters2018deep">Peters et al. (2018)</a>; <a href="#ref-howard2018universal">Howard et al. (2018)</a>; <a href="#ref-Radford2018ImprovingLU">Radford et al. (2018)</a>; <a href="#ref-devlin2019bert">Devlin et al. (2019)</a>).
E.g., mBERT and XLM-R, transformer-based multilingual masked language
models pre-trained on text in (approximately) 100 languages, can obtain
impressive performances for a variety of cross-lingual transfer tasks
(<a href="#ref-pires-etal-2019-multilingual">Pires et al. (2019)</a>; <a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>). Even though
these models were not trained with any cross-lingual objectives, they
still produce representations that can generalize well across languages
for a wide range of downstream tasks
(<a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>; <a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>). To analyze the
cross-lingual transfer ability of multilingual models, the model is
first fine-tuned on annotated data of a downstream task and then
evaluated in the zero or few-shot scenario, i.e., evaluated with the
fine-tuned models in the target language (<a href="#ref-hu2020xtreme">Hu et al. (2020)</a>)  with no or few
additional labeled target language data.

As impressive as these multilingual transformers might seem,
low-resource languages still perform sub-par to high-resource languages
(<a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>; <a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>), partly due to the fact
of a smaller pre-training corpus (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) , the curse
of multilinguality (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>)  and the importance of
vocabulary curation and size
(<a href="#ref-chung-etal-2020-improving">Chung et al. (2020)</a>; <a href="#ref-artetxe-etal-2020-call">Artetxe et al. (2020)</a>). E.g., the curse
of multilinguality argues assuming that the model capacity stays
constant that adding more languages leads to better cross-lingual
performance on low-resource languages up until a point where the overall
performance on monolingual and cross-lingual benchmarks degrades.
Intuitively explained, adding more languages to the model has two
effects: (1) Positive cross-lingual transfer, especially for
low-resource languages, and (2) lower per-language capacity, which then,
in turn, can degrade the overall model performance. These two effects of
*capacity dilution* and positive transfer need to be carefully traded
against each other. The model either needs to have a large model
capacity[^3] or is specialized (constrained) towards a subset of
languages beforehand. For these reasons, it is hard to create a single
model that can effectively represent a diverse set of languages. One
solution is to create language-specific models (monolingual models) with
language-specific vocabulary and model parameters
(<a href="#ref-virtanen2019multilingual">Virtanen et al. (2019)</a>; <a href="#ref-antoun-etal-2020-arabert">Antoun et al. (2020)</a>), but in return,
monolingual models need enough text to pre-train the model on the
language modeling task, which is typically not available for
low-resource languages. Additionally, we can not benefit from any
cross-lingual transfer from related languages, making it harder to
create an adequate representation for low-resource languages
(<a href="#ref-pires-etal-2019-multilingual">Pires et al. (2019)</a>; <a href="#ref-lauscher-etal-2020-zero">Lauscher et al. (2020)</a>).

# 2. Research Objective & Contribution

In this thesis, we explore how one can alleviate the issues of big
multilingual transformers for low-resource languages, especially the
curse of multilinguality.
Specifically, our two main objectives are: (1) Improving the
cross-lingual alignment for low-resource languages and (2) improving
cross-lingual downstream task performance for low-resource languages. We
utilize Knowledge Distillation (KD) by distilling the multilingual model
into language-specialized (also called monolingual) language models. We
make the following contributions:

- We propose a novel setup to distill multilingual transformers into monolingual components. Based on the setup, we 
propose two KD strategies: One for improving the alignment between two languages and one to improve cross-lingual 
downstream task performance. We call the former `MonoAlignment` and the latter `MonoShot`.
- `MonoAlignment` uses a distillation strategy to distill multilingual transformer models into smaller monolingual 
  components which have an improved aligned representation space between a high-resource language and a low-resource 
  language. We demonstrate the effectiveness by distilling XLM-R and experimenting with aligning English with 
  Turkish, Swahili, Urdu, and Basque.
- We compare `MonoAlignment` to other Knowledge Distillation
    strategies showing that it outperforms them in the retrieval task
    for low-resource languages.
- Our work suggests that an increase in the cross-lingual alignment of
    a multilingual transformer model does not necessarily translate into
    an increase in cross-lingual downstream task performance.
- Therefore, we propose `MonoShot`, another Knowledge Distillation
    strategy to distill multilingual transformer models into smaller
    monolingual components but which have a strong cross-lingual
    downstream performance in the zero- and few-shot settings.
- We show that `MonoShot` performs best among many different Knowledge
    Distillation strategies, albeit still lacks behind the teacher
    performance. However, it outperforms models built upon the teacher
    architecture but is trimmed down to the same size as the distilled
    components and initialized from parts of the teacher.
- We demonstrate an effective fine-tuning strategy for the zero-shot
    scenario for aligned monolingual models and compare it against many
    other strategies.

To conduct our research, we will draw inspiration from the field of
Cross-Lingual Representation Learning, Knowledge Distillation, and
Parameter-Efficient Fine-tuning. Following different Knowledge
Distillation strategies, such as from DistilBert (<a href="#ref-sanh2020distilbert">Sanh et al. (2020)</a>) 
or TinyBert (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>) , we distill the aligned cross-lingual
representation space of the multilingual transformer model XLM-R
(<a href="#ref-translation_conneau_2017">Conneau et al. (2017)</a>)  into smaller monolingual students. To
fine-tune aligned monolingual models in a zero-shot scenario, we study
the field of parameter-efficient fine-tuning, i.e., Adapters
(<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>; <a href="#ref-pfeiffer2021adapterfusion">Pfeiffer et al. (2021)</a>), BitFit
(<a href="#ref-bitfit_2021">Zaken et al. (2021)</a>)  and Sparse Fine-Tuning (<a href="#ref-guo_2020">Guo et al. (2020)</a>). Finally, we evaluate
the general-purpose cross-lingual representation of our monolingual
models in the retrieval, classification, structured prediction, and
question-answering task.






# References


<ol class="bibliography">
  <li id="ref-ruder2019unsupervised">Sebastian Ruder, Anders S{\o}gaard, Ivan Vuli{\'c}. <em>Unsupervised Cross-Lingual Representation Learning</em>. Proceedings of ACL 2019, Tutorial Abstracts. 2019</li>
  <li id="ref-magueresse2020lowresource">Alexandre Magueresse, Vincent Carles, Evan Heetderks. <em>Low-resource Languages: A Review of Past Work and Future Challenges</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2006.07264">link</a></li>
  <li id="ref-tatman-2017-gender">Rachael Tatman. <em>Gender and Dialect Bias in {Y}ou{T}ube{'}s Automatic Captions</em>. Proceedings of the First {ACL} Workshop on Ethics in Natural Language Processing. 2017. <a href="https://aclanthology.org/W17-1606">link</a></li>
  <li id="ref-rabinovich-etal-2018-native">Ella Rabinovich, Yulia Tsvetkov, Shuly Wintner. <em>Native Language Cognate Effects on Second Language Lexical Choice</em>. Transactions of the Association for Computational Linguistics. 2018. <a href="https://aclanthology.org/Q18-1024">link</a></li>
  <li id="ref-Zhiltsova2019MitigationOU">Alina Zhiltsova, Simon Caton, Catherine Mulway. <em>Mitigation of Unintended Biases against Non-Native English Texts in Sentiment Analysis</em>. AICS. 2019</li>
  <li id="ref-mikolov2013exploiting">Tomas Mikolov, Quoc V. Le, Ilya Sutskever. <em>Exploiting Similarities among Languages for Machine Translation</em>. arXiv. 2013. <a href="https://arxiv.org/abs/1309.4168">link</a></li>
  <li id="ref-hermann-blunsom-2014-multilingual">Karl Moritz Hermann, Phil Blunsom. <em>Multilingual Models for Compositional Distributed Semantics</em>. Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2014. <a href="https://aclanthology.org/P14-1006">link</a></li>
  <li id="ref-hu2020xtreme">Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, Melvin Johnson. <em>XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2003.11080">link</a></li>
  <li id="ref-peters2018deep">Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. <em>Deep contextualized word representations</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1802.05365">link</a></li>
  <li id="ref-howard2018universal">Jeremy Howard, Sebastian Ruder. <em>Universal Language Model Fine-tuning for Text Classification</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1801.06146">link</a></li>
  <li id="ref-Radford2018ImprovingLU">Alec Radford, Karthik Narasimhan. <em>Improving Language Understanding by Generative Pre-Training</em>. 2018</li>
  <li id="ref-devlin2019bert">Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. <em>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1810.04805">link</a></li>
  <li id="ref-pires-etal-2019-multilingual">Telmo Pires, Eva Schlinger, Dan Garrette. <em>How Multilingual is Multilingual {BERT}?</em>. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. <a href="https://aclanthology.org/P19-1493">link</a></li>
  <li id="ref-conneau2020unsupervised">Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. <em>Unsupervised Cross-lingual Representation Learning at Scale</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1911.02116">link</a></li>
  <li id="ref-wu-dredze-2019-beto">Shijie Wu, Mark Dredze. <em>Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of {BERT}</em>. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019. <a href="https://aclanthology.org/D19-1077">link</a></li>
  <li id="ref-chung-etal-2020-improving">Hyung Won Chung, Dan Garrette, Kiat Chuan Tan, Jason Riesa. <em>Improving Multilingual Models with Language-Clustered Vocabularies</em>. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. <a href="https://aclanthology.org/2020.emnlp-main.367">link</a></li>
  <li id="ref-artetxe-etal-2020-call">Mikel Artetxe, Sebastian Ruder, Dani Yogatama, Gorka Labaka, Eneko Agirre. <em>A Call for More Rigor in Unsupervised Cross-lingual Learning</em>. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020. <a href="https://aclanthology.org/2020.acl-main.658">link</a></li>
  <li id="ref-virtanen2019multilingual">Antti Virtanen, Jenna Kanerva, Rami Ilo, Jouni Luoma, Juhani Luotolahti, Tapio Salakoski, Filip Ginter, Sampo Pyysalo. <em>Multilingual is not enough: BERT for Finnish</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1912.07076">link</a></li>
  <li id="ref-antoun-etal-2020-arabert">Wissam Antoun, Fady Baly, Hazem Hajj. <em>{A}ra{BERT}: Transformer-based Model for {A}rabic Language Understanding</em>. Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection. 2020. <a href="https://aclanthology.org/2020.osact-1.2">link</a></li>
  <li id="ref-lauscher-etal-2020-zero">Anne Lauscher, Vinit Ravishankar, Ivan Vuli{\'c}, Goran Glava{\v{s}}. <em>From Zero to Hero: {O}n the Limitations of Zero-Shot Language Transfer with Multilingual {T}ransformers</em>. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. <a href="https://aclanthology.org/2020.emnlp-main.363">link</a></li>
  <li id="ref-sanh2020distilbert">Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf. <em>DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1910.01108">link</a></li>
  <li id="ref-jiao2020tinybert">Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. <em>TinyBERT: Distilling BERT for Natural Language Understanding</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1909.10351">link</a></li>
  <li id="ref-translation_conneau_2017">Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. <em>Word Translation Without Parallel Data</em>. arXiv. 2017. <a href="https://arxiv.org/abs/1710.04087">link</a></li>
  <li id="ref-houlsby2019parameterefficient">Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly. <em>Parameter-Efficient Transfer Learning for NLP</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1902.00751">link</a></li>
  <li id="ref-pfeiffer2021adapterfusion">Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych. <em>AdapterFusion: Non-Destructive Task Composition for Transfer Learning</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2005.00247">link</a></li>
  <li id="ref-bitfit_2021">Elad Ben Zaken, Shauli Ravfogel, Yoav Goldberg. <em>BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2106.10199">link</a></li>
  <li id="ref-guo_2020">Demi Guo, Alexander M. Rush, Yoon Kim. <em>Parameter-Efficient Transfer Learning with Diff Pruning</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2012.07463">link</a></li>
</ol>



[^1]: <http://labs.theguardian.com/digital-language-divide/>

[^2]: Word embeddings and word representation are interchangeable in our
    thesis.

[^3]: Here: Measured in the number of free parameters in the model.




