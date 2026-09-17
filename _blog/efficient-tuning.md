---
title: "[Paper] Survey: Parameter-Efficient Fine-Tuning in NLP"
date: 2022-06-28 11:30:47 +01:00
tags: [paper, nlp]
description: Master Thesis
image: "/assets/img/cross-lingual/introduction.jpg"
---
*Part of the related work section in my Master Thesis. Download the full version
<a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>.*


<figure>
<img src="/assets/img/efficient-tuning/introduction.jpg" alt="interview-img">
</figure>


# Table of Contents
1. [Motivation](#1-motivation)
2. [Adapters](#2-adapters)
3. [Sparse Fine-Tuning](#3-sparse-fine-tuning)



# 1. Motivation

The standard approach to solving a new task with a
pre-trained transformer is by adding a task-head (e.g., a linear
classification layer) on top of the pre-trained transformer (encoder)
and minimizing the task loss end-to-end. However, this approach results
in a completely new unique, large model making it harder to track what
significantly changed during fine-tuning and therefore making it also
hard to transfer acquired task-specific knowledge (*modularity*). The
latter is important in our monolingual setup as we want to transfer the
acquired task-specific knowledge by our source student into our target
student. Ideally, transferring the
acquired task-specific knowledge still matches the results of fully
fine-tuning one model.

The first work that we explore is Adapters
(<a href="#ref-rebuffi2017learning">Rebuffi et al. (2017)</a>; <a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>) which inserts a
small subset of trainable task-specific parameters between layers of a
model and only changes these during fine-tuning, keeping the original
parameters frozen. We then discuss sparse fine-tuning, which only
changes a subset of the pre-trained model parameters. Specifically, we
consider Diff-Pruning (<a href="#ref-guo_2020">Guo et al. (2020)</a>), which adds a sparse, task-specific
difference-vector to the original parameters, and BitFit (<a href="#ref-bitfit_2021">Zaken et al. (2021)</a>),
which enforces sparseness by only fine-tuning the bias terms and the
classification layer.

# 2. Adapters

Adapters were initially proposed for computer vision to adapt to
multiple domains (<a href="#ref-rebuffi2017learning">Rebuffi et al. (2017)</a>) but were then used as an
alternative lightweight training strategy for pre-trained transformers
in NLP (<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>). Adapters introduce additional
parameters to a pre-trained transformer, usually small, bottleneck
feed-forward networks inserted at each transformer layer. Adapters
enable us to keep the pre-trained parameters of the model fixed and only
fine-tune the newly introduced parameters on either a new task
(<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>; <a href="#ref-stickland2019bert">Stickland et al. (2019)</a>; <a href="#ref-pfeiffer2021adapterfusion">Pfeiffer et al. (2021)</a>)
or new domains (<a href="#ref-bapna2019simple">Bapna et al. (2019)</a>). Adapters perform either on par or
slightly below full fine-tuning
(<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>; <a href="#ref-stickland2019bert">Stickland et al. (2019)</a>; <a href="#ref-pfeiffer2021adapterfusion">Pfeiffer et al. (2021)</a>).
Importantly, adapters learn task-specific representations which are
compatible with subsequent transformer layers (<a href="#ref-adapterhub_2020">Pfeiffer et al. (2020)</a>).


**Placement & Architecture.** Most work insert adapters at each layer of
the transformer model, the architecture and placement of adapters are,
however, non-trivial: (<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>) experiment with
different adapter architectures and empirically validated that using a
two-layer feed-forward neural network with a bottleneck worked well:

<figure>
<img src="/assets/img/master-thesis/placement.PNG" alt="interview-img">
</figure>

This simple down- and
up-projection with a non-linearity has become the common adapter
architecture. The placement and number of adapters within each
transformer block are still debated. (<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>)
place the adapter at two positions: One after the multi-head attention
and one after the feed-forward layers. (<a href="#ref-stickland2019bert">Stickland et al. (2019)</a>) just use one
adapter after the feed-forward layers, which (<a href="#ref-bapna2019simple">Bapna et al. (2019)</a>) adopted
and extends by including a layer norm (<a href="#ref-ba2016layer">Ba et al. (2016)</a>) after the adapter.
(<a href="#ref-pfeiffer2021adapterfusion">Pfeiffer et al. (2021)</a>) test out different adapter positions and
adapter architectures jointly and came to the conclusion to use the same
adapter architecture as (<a href="#ref-houlsby2019parameterefficient">Houlsby et al. (2019)</a>) but only places
the adapter after the feed-forward neural network:

<figure>
<img src="/assets/img/master-thesis/architecture_adapter.JPG" alt="interview-img">
</figure>

**Modularity of Representations.** One important property of Adapters is
that they learn task-specific knowledge within each adapter component.
The reason is that they are placed within a frozen transformer block
layer, forcing the adapters to learn an output representation compatible
with the subsequent layer of the transformer model. This results in
modular adapters, meaning that they can be either stacked on top of each
other or replaced dynamically (<a href="#ref-mad_x">Pfeiffer et al. (2020)</a>). This modularity of adapters can
be used to fine-tune multiple monolingual aligned students on
cross-lingual downstream tasks: Instead of fine-tuning the whole student
model on the source language, we insert and fine-tune the adapters,
which then can be inserted into the monolingual student corresponding to
the target language. 

# 3. Sparse Fine-Tuning

Sparse fine-tuning (SFT) only fine-tunes a small subset of the original
pre-trained model parameters at each step, effectively fine-tuning in a
parameter efficient way. The fine-tuning procedure can be describes as

$$\begin{aligned}
    \Theta_{\text{task}} =  \Theta_{\text{pretrained}} + \delta_{\text{task}}
\end{aligned}$$ 

where $\Theta_{\text{task}}$ is the task-specific
parameterization of the model after fine-tuning,
$\Theta_{\text{pretrained}}$ is the set of pretrained parameters which
is fixed and $\delta_{\text{task}}$ is called the task-specific diff
vector. We call the procedure sparse if $\delta_{\text{task}}$ is
sparse. As we only have to store the nonzero positions and weights of
the diff vector, the method is parameter efficient. Method generally
differ in the calculation of $\delta_{\text{task}}$ and its induced
sparseness.

**Methods.** (<a href="#ref-guo_2020">Guo et al. (2020)</a>) introduce Diff Pruning, which determines
$\delta_{\text{task}}$ by adaptively pruning the diff vector during
training. To induce this sparseness, they utilize a differentiable
approximation of the $L_0$-norm penalty (<a href="#ref-louizos_2017">Louizos et al. (2017)</a>). (<a href="#ref-bitfit_2021">Zaken et al. (2021)</a>)
induce sparseness by only allowing non-zero differences in the bias
parameters (and the classification layer) of the transformer model. The
method, called BitFit, and Diff-Pruning, both can match the performance
of fine-tuned baselines on the GLUE benchmark (<a href="#ref-guo_2020">Guo et al. (2020)</a>; <a href="#ref-bitfit_2021">Zaken et al. (2021)</a>).
(<a href="#ref-ansell_2021">Ansell et al. (2021)</a>) learn sparse, real-valued masks based on a simple variant
of the Lottery Ticket Hypothesis (<a href="#ref-frankle_2018">Frankle et al. (2018)</a>): First, they fine-tune
a pre-trained model for a specific task or language, then select the
subset of parameters that change the most which correspond to the
non-zero values of the diff vector $\delta_{\text{task}}$. Then, the
authors set the model to its original pre-trained initialization and
re-tune the model again by only fine-tuning the selected subset of
parameters. The diff vector $\delta_{\text{task}}$ is therefore sparse.

**Comparison to Adapters.** In contrast to Adapters, Sparse fine-tuning
(SFT) does not modify the architecture of the model but restricts its
fine-tuning to a subset of model parameters (<a href="#ref-guo_2020">Guo et al. (2020)</a>; <a href="#ref-bitfit_2021">Zaken et al. (2021)</a>).
As a result, SFT is much more expressive, as they are not constricted to
just modifying the output of Transformer layers with shallow MLP but can
directly modify the pre-trained model's embedding and attention layers
(<a href="#ref-ansell_2021">Ansell et al. (2021)</a>). Similar to Adapters, (<a href="#ref-ansell_2021">Ansell et al. (2021)</a>) show that their sparse
fine-tuning technique has the same concept of modality found in Adapters
(<a href="#ref-mad_x">Pfeiffer et al. (2020)</a>). Again this modularity can be used in our monolingual setup to
fine-tune on cross-lingual downstream tasks.


# References


<ol class="bibliography">
  <li id="ref-rebuffi2017learning">Sylvestre-Alvise Rebuffi, Hakan Bilen, Andrea Vedaldi. <em>Learning multiple visual domains with residual adapters</em>. arXiv. 2017. <a href="https://arxiv.org/abs/1705.08045">link</a></li>
  <li id="ref-houlsby2019parameterefficient">Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly. <em>Parameter-Efficient Transfer Learning for NLP</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1902.00751">link</a></li>
  <li id="ref-guo_2020">Demi Guo, Alexander M. Rush, Yoon Kim. <em>Parameter-Efficient Transfer Learning with Diff Pruning</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2012.07463">link</a></li>
  <li id="ref-bitfit_2021">Elad Ben Zaken, Shauli Ravfogel, Yoav Goldberg. <em>BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2106.10199">link</a></li>
  <li id="ref-stickland2019bert">Asa Cooper Stickland, Iain Murray. <em>BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1902.02671">link</a></li>
  <li id="ref-pfeiffer2021adapterfusion">Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych. <em>AdapterFusion: Non-Destructive Task Composition for Transfer Learning</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2005.00247">link</a></li>
  <li id="ref-bapna2019simple">Ankur Bapna, Naveen Arivazhagan, Orhan Firat. <em>Simple, Scalable Adaptation for Neural Machine Translation</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1909.08478">link</a></li>
  <li id="ref-adapterhub_2020">Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, Iryna Gurevych. <em>AdapterHub: A Framework for Adapting Transformers</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2007.07779">link</a></li>
  <li id="ref-ba2016layer">Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. <em>Layer Normalization</em>. arXiv. 2016. <a href="https://arxiv.org/abs/1607.06450">link</a></li>
  <li id="ref-mad_x">Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, Sebastian Ruder. <em>MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2005.00052">link</a></li>
  <li id="ref-louizos_2017">Christos Louizos, Max Welling, Diederik P. Kingma. <em>Learning Sparse Neural Networks through $L_0$ Regularization</em>. arXiv. 2017. <a href="https://arxiv.org/abs/1712.01312">link</a></li>
  <li id="ref-ansell_2021">Alan Ansell, Edoardo Maria Ponti, Anna Korhonen, Ivan Vulić. <em>Composable Sparse Fine-Tuning for Cross-Lingual Transfer</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2110.07560">link</a></li>
  <li id="ref-frankle_2018">Jonathan Frankle, Michael Carbin. <em>The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1803.03635">link</a></li>
</ol>





