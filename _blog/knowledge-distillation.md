---
title: "[Paper] Survey: Knowledge Distillation in NLP"
date: 2022-06-10 11:30:47 +01:00
tags: [paper, nlp]
description: Master Thesis
image: "/assets/img/cross-lingual/introduction.jpg"
---
*Part of the related work section in my Master Thesis. Download the full version
<a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>.*


<figure>
<img src="/assets/img/knowledge-distillation/introduction.jpg" alt="interview-img">
</figure>


# Table of Contents
1. [Motivation & Preliminaries](#1-motivation---preliminaries)
2. [Brief History: Knowledge Distillation for Transformers](#2-brief-history--knowledge-distillation-for-transformers)
3. [Transformer Components Distillation](#3-transformer-components-distillation)
4. [Distillation Setup Strategies](#4-distillation-setup-strategies)
5. [Challenges](#5-challenges)

Knowledge Distillation is a technique to transfer knowledge from a
well-trained teacher model to a student model, typically smaller than
the teacher. During training, Knowledge Distillation allows the student
model to access the learned knowledge from the teacher model, providing
more information to the student. Thus, the student can achieve similar
or even better performance than the teacher.

As our thesis is based on multilingual transformers, we restrict our
review to Knowledge Distillation for transformers.

# 1. Motivation & Preliminaries

**Motivation.** (<a href="#ref-Bucila2006ModelC">Bucila et al. (2006)</a>) first introduced the concept of
Knowledge Distillation as a way to compress models by transferring the
learned knowledge to a smaller model. In their work, the primary goal
was to compress a large complex ensemble into smaller, faster models
without significant performance loss. First, they train the ensemble
model (also called *teacher*) and then transfer the acquired knowledge
to a smaller model (also called *student*) by mimicking the teacher's
behavior. Thus the idea of Knowledge Distillation[^1] was born.

Knowledge Distillation is especially beneficial in the case of complex,
overparameterized teachers. From an optimization perspective,
(<a href="#ref-du2018power">Du et al. (2018)</a>; <a href="#ref-soltanolkotabi2018theoretical">Soltanolkotabi et al. (2018)</a>) show that high capacity
models (i.e., the teacher) can find a good local minimum due to
over-parameterization. Therefore, using these over-parameterized models
to guide a lower capacity model during training can facilitate
optimization. The state-of-the-art multilingual models are massive
pre-trained transformers consisting of millions of parameters which some
argue are over-parameterized but need the capacity during pre-training
(<a href="#ref-Hao_2019">Hao et al. (2019)</a>; <a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>; <a href="#ref-dufter2021identifying">Dufter et al. (2021)</a>). Our
approach uses Knowledge Distillation to precisely distill these
transformer models to induce cross-lingual knowledge into students.

**Vanilla Knowledge Distillation.** The most straightforward approach to
Knowledge Distillation is to distill from the output layer by matching
the output of the teacher and student
(<a href="#ref-Bucila2006ModelC">Bucila et al. (2006)</a>; <a href="#ref-ba_deep_2013">Ba et al. (2013)</a>; <a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>). Initially,
(<a href="#ref-Bucila2006ModelC">Bucila et al. (2006)</a>) used the teacher to make predictions on an unlabeled
dataset producing labels for the student to mimic. The created labels
are referred to as *hard labels*, and the dataset used to train the
student is called *transfer dataset*. As hard labels only transfer
information about the highest class probability, (<a href="#ref-ba_deep_2013">Ba et al. (2013)</a>) propose
to use the logits of the teacher, called *soft labels*. The idea is that
the soft labels of a well-trained teacher provide additional supervisory
signals of the inter-class similarities to the student. They directly
used the logits $z$ by minimizing the squared difference between the
logits produced by the teacher and the logits produced by the student:

$$\begin{equation} \tag{1}\label{eq:logits_mse} 
     L_{\text{logits}} = \frac{1}{2N} \sum_{x \in X} \Vert z_{x}^{TE} - z_{x}^{ST}\Vert_2^2 ,
\end{equation}$$ 

where $N$ is the number of examples, $z_{x}^{TE}$ the
logit output of the teacher and $z_{x}^{ST}$ of the logit output of the
student.

One problem that may arise in well-trained teachers is that they are
overconfident and thus almost always predict classes (tokens) with very
high confidence. The learned similarities between similar classes
(tokens) reside in the ratios of very small probabilities in the soft
targets - These, however, have very little influence on the cost
function $\eqref{eq:logits_mse}$ during distillation. To leverage the
information residing in these small probabilities (<a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>)
propose to *soften* the distribution by modifying the predicted
probability distribution of the teacher. (<a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>) introduce
a variable called temperature into the final softmax: 

$$\begin{equation} \tag{2}\label{eq:temperature}
    \sigma (z_{x}; T)\_i := p_i = \frac{\exp{(z_{x, i}/T)}}{\sum_j \exp{(z_{x, j}/T)}},
\end{equation}$$

where $T$ denotes the temperature, which is normally set
to $1$ for the softmax function. A higher temperature $T$ causes a
softer probability distribution over tokens, resulting in a higher
entropy in the probability distribution. (<a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>) then use
the same temperature $T$ when training the student to match these soft
targets. Furthermore, (<a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>) showed that matching logits
is a special case of the modified softmax in $\eqref{eq:temperature}$. Using the Kullback-Leibler divergence
between teacher and student, we get the *(unscaled) Hinton loss
function*: 

$$\begin{aligned}
     \sum_{x \in X} D_{KL} \left(\sigma(z_{x}^{TE}; T) \Vert  \sigma(z^{ST}_x; T)\right).
\end{aligned}$$ 

To compensate for the changed magnitude of the gradients
caused by the soft targets scale, (<a href="#ref-hinton2015distilling">Hinton et al. (2015)</a>) multiply the
loss function with $T^2$ to get the *Hinton loss function*:

$$\begin{equation} \tag{3}\label{eq:hinton_loss}
    L_{H} = T^2 \cdot \sum_{x \in X} D_{KL} \left(\sigma (z_{x}^{TE}; T) \Vert \sigma (z_{x}^{ST};
T) \right).
\end{equation}$$ 

This allows us to change the temperature while
experimenting without changing the relative contributions of the soft
targets. The Hinton loss $\eqref{eq:hinton_loss}$ is the vanilla setup that is mainly used and
referred to as (vanilla) Knowledge Distillation.

The main difference between different Knowledge Distillation strategies
is how one mimics the teacher's behavior. Formally, we can define a
transformation $f^{TE}$ and $f^{ST}$ of the teacher and student network
inputs, respectively, to some informative representation for transfer.
(<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>) calls these transformations *behavior functions*.
Knowledge Distillation can then be defined in general as minimizing the
objective function 

$$\begin{aligned}
    L_{\text{KD}} = \sum_{x \in X} L \left(f^{TE}(x), f^{ST}(x) \right),
\end{aligned}$$ 

where $L(\cdot)$ is a loss function evaluating
the difference between a teacher and student networks, $x$ the (text)
input, and $X$ denotes the training set. Behavior functions
can be defined for any type of representation of the input, e.g., the
logits of the final output or some intermediate representations in the
network, which the student then mimics.

# 2. Brief History: Knowledge Distillation for Transformers

Knowledge Distillation methods first found their use in the area of
Computer Vision. As AlexNet (<a href="#ref-Krizhevsky_2012">Krizhevsky et al. (2012)</a>), one of the largest
neural networks at that time, completely outperformed other methods on
ImageNet (<a href="#ref-deng2009imagenet">Deng et al. (2009)</a>), the trend towards bigger models started in
that area. To keep the models to a relatively acceptable size while
still having the same performance as the big models, a variety of model
compression techniques were developed, such as Knowledge Distillation,
see the survey of (<a href="#ref-Gou_2021">Gou et al. (2021)</a>) for a more in-depth look into Knowledge
Distillation in Computer Vision.

**Task-Specific Distillation.** (<a href="#ref-sun2019patient">Sun et al. (2019)</a>) was one of the first
known successes at using Knowledge Distillation for BERT at the
fine-tuning stage. They introduce *Patient Knowledge Distillation (PKD)*
which uses the intermediate representations of BERT for a more effective
distillation (additional to the Hinton Loss). This is motivated by
previous findings of (<a href="#ref-romero2015fitnets">Romero et al. (2015)</a>) showing that distilling
intermediate representations can serve as hints for the student during
training, improving the final performance. Similar to PKD, XtremeDistil
(<a href="#ref-mukherjee2020xtremedistil">Mukherjee et al. (2020)</a>) also distills from intermediate
representations but additionally utilizes parameter projection that is
agnostic of teacher architecture. Subsequently, inspired by the findings
that attention weights of BERT capture linguistic knowledge and that
BERT becomes more complex in higher layers (<a href="#ref-clark2019does">Clark et al. (2019)</a>), *Stacked
Internal Distillation (SID)* (<a href="#ref-aguilar2020knowledge">Aguilar et al. (2020)</a>) additionally
distills the attention probabilities of the teacher and from lower
layers first. However, these works used distillation for fine-tuned
models on specific downstream tasks (*task-specific distillation*). In
our thesis, we want to produce a general-purpose student that can be
fine-tuned on any downstream task. While we are not focusing on
task-specific distillation
(<a href="#ref-liu2019improving">Liu et al. (2019)</a>; <a href="#ref-turc2019wellread">Turc et al. (2019)</a>; <a href="#ref-tang2019distilling">Tang et al. (2019)</a>; <a href="#ref-kaliamoorthi2021distilling">Kaliamoorthi et al. (2021)</a>)
or multi-task distillation
(<a href="#ref-tan2019multilingual">Tan et al. (2019)</a>; <a href="#ref-clark2019bam">Clark et al. (2019)</a>; <a href="#ref-liu2020mkd">Liu et al. (2020)</a>), we still want to
mention important work in this field.

**General-Purpose Students.** To produce general-purpose students,
distillation happens during the pre-training tasks, mainly the masked
language modeling task. In this task, the transformer model is trained
to predict the original token for each masked token by maximizing the
estimated probability for this token. The standard loss function is to
minimize the cross-entropy between the transformer's predicted
distribution and the one-hot distribution of all tokens. We denote the
loss as the masked language modeling loss $L_{\text{MLM}}$
(<a href="#ref-devlin2019bert">Devlin et al. (2019)</a>) as 

$$\begin{equation} \tag{4}\label{eq:ce}
    L_{\text{MLM}}= - \sum_{x \in X} \left( \sum_i l_{x, i} \log(p_{x, i}) \right),
\end{equation}$$ 

where $l_{x, i}$ is the ground-truth and $p_{x, i}$ the
predicted probability for $i$-th token in the text input $x$. Typically
a transformer applies a softmax function to obtain $p_{x, i}$, denoted
by $\sigma (z_{x})=p_{x} = (p_{x, 1}, ..., p_{x, C})$, where $\sigma $ is
the softmax function and $z_{x} = (z_{x, 1}, ..., z_{x, C})$ the logit
output of the text input $x$.

*DistilBert* (<a href="#ref-sanh2020distilbert">Sanh et al. (2020)</a>) extends the work of (<a href="#ref-sun2019patient">Sun et al. (2019)</a>)
by distilling during pre-training on a large-scale corpus with a
soft-label distillation loss and a cosine embedding loss to construct a
general-purpose student. Furthermore, they initialize the student from
the teacher by taking one layer out of two. Similar to SID
 (<a href="#ref-aguilar2020knowledge">Aguilar et al. (2020)</a>), TinyBert (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>)  introduce
additional attention distillation to produce general-purpose students.
Specifically, they distill self-attention distributions
$\text{Attention}(Q, K, V)$. TinyBert further uses task
distillation with data augmentation to strengthen performance on
downstream tasks. They show that their $6$ layer model performs on par
with its teacher BERT on the GLUE benchmark (<a href="#ref-wang2019glue">Wang et al. (2019)</a>) . Instead of
using shallower students, MobileBERT (<a href="#ref-sun2020mobilebert">Sun et al. (2020)</a>)  uses thinner
students by introducing bottleneck structures which have been shown to
be more effective (<a href="#ref-turc2019wellread">Turc et al. (2019)</a>) . Compared to TinyBert, they
outperform it on GLUE and SQuAD with a similar-sized MobileBERT while
not utilizing any task-distillation or data augmentations during
fine-tuning. However, MobileBERT constructs another teacher network and
many architectural modifications to help facilitate training. MiniLM
 (<a href="#ref-wang2020minilm">Wang et al. (2020)</a>) only distills the self-attention module with an added
loss function: They align the relation between values in the
self-attention module, which is calculated via the multi-head scaled
dot-product between values $V$. Furthermore, they only distill from the
last transformer layer of the teacher. Compared to TinyBert and
MobileBert, MiniLM alleviates the difficulties in layer mapping between
the teacher and student models, and the layer number of our student
model can be more flexible. (<a href="#ref-khanuja2021mergedistill">Khanuja et al. (2021)</a>)introduce
MergeDistil, which uses multiple mono- or multilingual teachers to
distill into one general-purpose student to leverage language-specific
LM.

To further dive deeper into previous works and build a foundation for
further exploration for our thesis, we distinguish previous works into
which parts of the transformer they distill from.

# 3. Transformer Components Distillation

This section will explore different behavior functions and their
corresponding loss function to transfer knowledge from a transformer
teacher into a transformer student. We categorize the possible parts of
the transformer that we can distill from into: *Final output layer*,
*hidden layer representations*, *attention* and *embedding
distillation*.

We summarize our findings in the next Table:

<figure>
<img src="/assets/img/master-thesis/table_1.png" alt="interview-img">
<em>Categorizing of previous works into different transformer parts
distillation during pre-training. Notice that TinyBert does not utilize
soft or hard targets during pre-training, only in their second
distillation stage (task-distillation). Furthermore, we specify the loss
function (KL: Kullback Leibler Loss; MSE: Mean-squared error;
COS: Cosine Embedding Loss; CE: Cross-entropy loss; SVD:
Singular Value Decomposition.</em>
</figure>

**Final Output Layer.** One simple idea to induce knowledge from a
teacher during pre-training is to use the predicted distribution over
all tokens of the teacher model as *soft targets* for training the
student. The hope is that soft targets are much more informative than
hard targets as they can reveal similarities between tokens, e.g., the
masked token is \"dog\" and our well-trained teacher model would give a
high probability to \"dog\", but also to \"cat\" as both could make
sense in the masked sentence. This information about similarities
between tokens can then be transferred to our students. One problem that
may arise in well-trained teachers is that they almost always predict
the correct token with very high confidence.

Almost all works use the hard and soft targets to distill knowledge from
the teacher. PKD, SID, and DistilBERT use the cross-entropy loss to
distill from the soft and hard targets, while XtremeDistil uses the
mean-squared error to align soft targets. In some implementations, the
Kullback Leibler loss is used to align soft targets. During
pre-training, TinyBert does not use any output layer distillation as
(<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>) argue that their goal is to primarily learn the
intermediate structures of BERT. Furthermore, they conduct experiments
showing that output layer distillation does not bring additional
improvements on downstream tasks.

**Hidden Layer Representation Distillation.** Using only logits to
transfer cross-lingual knowledge from the teacher can be problematic as
they only serve as one \"task-specific\" knowledge, in this case, masked
language modeling. (<a href="#ref-romero2015fitnets">Romero et al. (2015)</a>) show that additional hints from
the *intermediate layers* can improve the training process and the final
performance of the student.

As one has to decide which layer of the teacher to distill from
(typically, the student is smaller than the teacher), we have to define
a mapping function. For this purpose, we assume that the teacher model
has $N$ Transformer layers and the student $M$ Transformer layers, where
$M \leq N$. The index $0$ corresponds to the embedding layer, $M+1$ the
index of the student's prediction layer, and $N+1$ the index of the
teacher's prediction layer. The mapping function is then defined as

$$\begin{equation} \tag{5}\label{eq:mapping-function}
    n = g(m), \qquad 0 < m \leq M, 0 < g(m) \leq N
\end{equation}$$ 

between indices from student layers to teacher layers,
where the teacher's $g(m)$-th layer is distilled into the $m$-th layer
of the student model. Typically, there are four different mapping
strategies:

- *Uniform*: We distill uniformly over the teacher layers. E.g. assume 12 layer teacher and 6 layer student, then 
  the mapping function is $g(m) = m * 2$.

<figure>
<img src="/assets/img/master-thesis/uniform.PNG" alt="interview-img" width="400">
</figure>

- *Top*: We distill from the top layers of the teacher: $g(m) = m + N - M$.

<figure>
<img src="/assets/img/master-thesis/top.PNG" alt="interview-img" width="400">
</figure>

- *Bottom*: We distill from the bottom layers of the teacher: $g(m) = m$.

<figure>
<img src="/assets/img/master-thesis/bottom.PNG" alt="interview-img" width="400">
</figure>

- *Last*: We distill only from the last layer of the teacher into the  last layer of the student: $g(M) = N$.


PKD (<a href="#ref-sun2019patient">Sun et al. (2019)</a>)  focuses on extracting intermediate representations
but only for the `[CLS]` token, reasoning that BERT's original
implementation (<a href="#ref-devlin2019bert">Devlin et al. (2019)</a>)  mostly uses the output from the last
layer's `[CLS]` token to perform predictions for downstream tasks. The
hope is that if the student can imitate the representation of `[CLS]` in
the teacher's intermediate layers for any given input, the
generalization ability can be similar to the teacher. The additional
loss is then defined as the mean squared error between the normalized
hidden states of the `[CLS]` token in all layers: 

$$\begin{equation} \tag{6}\label{eq:msecls}
    L_{\text{MSECLS}} = \sum_{x \in X} \sum_{m=1}^{M} \text{MSE} \left(\text{CLS}_m^{ST}(x), \text{CLS}_{g(m)}^{TE}(x) \right)^2,
\end{equation}$$ 

where $\text{CLS}^{ST}\_m(x) \in \mathbb{R}^{d}$ and
$\text{CLS}^{TE}\_{g(m)}(x) \in \mathbb{R}^{d}$ extracts the `[CLS]`
token for the input $x$ in the layer $m$ for student and $g(m)$ teacher
respectively. PKD was tested with the mapping function *top* and
*uniform*, where the strategy *uniform* showed better performance.
Similar to PKD, SID (<a href="#ref-aguilar2020knowledge">Aguilar et al. (2020)</a>) also uses hidden vector
representations for the `[CLS]` token to additionally align internal
representations, but opted to use cosine similarity 

$$\begin{aligned}
    L_{\text{cos}CLS}(x) = 1 - cos\left(\text{CLS}_m^{ST}(x), \text{CLS}_{g(m)}^{TE}(x) \right)
\end{aligned}$$ 

with the *uniform* mapping strategy. The obvious problem
with focusing on the `[CLS]` is that it only works for tasks that use
the `[CLS]` for prediction such as tasks in the GLUE dataset
(<a href="#ref-wang2019glue">Wang et al. (2019)</a>).

(<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>) extended the idea of distilling hidden layer
representations to student representations with any arbitrary number of
hidden dimensions with their proposed model *TinyBert*. Let
$H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$ refer to the hidden states
in layer $m$ of student and teacher networks respectively, $L$ the
sequence length and the scalars $d', d$ denote the hidden sizes of the
student and teacher models respectively. Instead of assuming that the
hidden dimension is the same for the teacher and student $d' = d$, they
allow for $d \neq d'$ in general. This allows for creating thinner
models by using a smaller hidden dimension $d' < d$ than the teacher.
They achieve this by introducing a learnable linear transformation
matrix $W \in \mathbb{R}^{d' \times d}$ to transform the hidden states
of the student network into the same space as the teacher network's
states, i.e., $H^{ST}\_m W \in \mathbb{R}^{L \times d}$. TinyBert
transfers the knowledge which resides in the hidden layers output
representation from *each token* in the sequence by minimizing the MSE
across all tokens: 

$$\begin{equation} \tag{7}\label{eq:msehidn}
    L_{\text{Hid}\_MSE}(\ \cdot \ ; m) = \text{MSE}(H^{ST}_m W_m, H^{TE}_{g(m)}),
\end{equation}$$ 

where the matrices $H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$ can have different hidden
sizes $d \neq d'$. Furthermore, they experimented with all three mapping
functions and concluded that *uniform* works best in their case. Notice
that we can also align hidden representations with the cosine loss

$$\begin{equation} \tag{8}\label{eq:coshidn}
    L_{\text{COS}hidn} = 1 - \text{cos}(H^{ST}_m W_m, H^{TE}_{g(m)}), 
\end{equation}$$ 

with the matrices
$H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$.

*XtremeDistil* also uses a projection to make all output spaces
compatible. The projection however is non-linear and they use the
KL-divergence: 

$$\begin{aligned}
    L_{\text{KL}hidn}(\ \cdot \ ; m) = D_{KL}\left(\textit{Gelu}\left(H^{ST}_m W_m + b_m\right) \Vert H^{TE}_{g(m)} \right),
\end{aligned}$$ 

where $W_m$ is the learnable projection matrix, $b_m$ the learnable bias term and *Gelu* (Gaussian Error Linear Unit) is the
non-linear projection function.

As we have seen, different mapping functions were used across previous
work. It might not be useful to distill from intermediate
representations as in (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>; <a href="#ref-mukherjee2020xtremedistil">Mukherjee et al. (2020)</a>; <a href="#ref-aguilar2020knowledge">Aguilar et al. (2020)</a>), especially when the
student is a network of lower representational capacity.
(<a href="#ref-yang2021knowledge">Yang et al. (2021)</a>) showed that using only the last hidden layer (*last*
strategy) for matching representation results in the best performance,
albeit experiments were conducted with Convolutional Neural Networks. To
further align the last hidden representation, (<a href="#ref-yang2021knowledge">Yang et al. (2021)</a>) use the
teacher's pre-trained Softmax Regression (SR) classifier to first pass
the input $x$ to the teacher network resulting in the output

$$\begin{aligned}
    \sigma (z_x^{TE}) = \sigma (W^{TE}_{N+1} H_N^{TE}),
\end{aligned}$$ 

where $W^{TE}\_{N+1}$ is the weight matrix of the softmax
classifier head. Moreover, they feed the same input to the student to
get the last hidden representation $H_M^{ST}$ and input the
representation to the teacher's SR classifier to obtain

$$\begin{aligned}
    \sigma (z_x^{ST}) = \sigma (W^{TE}_{N+1} H_M^{ST}).
\end{aligned}$$ 

They then optimize the loss 

$$\begin{aligned}
    L_{SR} = - \sigma  \left(W^{TE}_{N+1} H_M^{TE} \right) \log \left( \sigma (W^{TE}_{N+1} H_M^{ST}) \right),
\end{aligned}$$ 

while keeping the classifier's parameter $W^{TE}_{N+1}$
frozen. This loss is aligning the last hidden representations since if
$\sigma (z_x^{TE}) = \sigma (z_x^{ST})$ (which the loss is optimizing to)
then this implies that $H_N^{TE} = H_M^{ST}$.

**Attention Distillation.** (<a href="#ref-clark2019does">Clark et al. (2019)</a>) found that the attention
weights of the transformer model BERT can capture linguistic knowledge,
which can be used to transfer linguistic knowledge into our student.



Motivated by this, (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>) additionally uses attention
distillation for TinyBert, which uses the mean squared error to align
the teacher and students matrices of the multi-head attention. Let the
matrices $Q, K, V \in \mathbb{R}^{L \times d_k}$ denote the queries,
keys and values, where $d_k$ is the dimension of keys. As already
discussed, the attention $\text{Attention}(Q, K, V)$ is calculated with

$$\begin{equation} \tag{8}\label{eq:attention}
    A = \frac{QK^{T}}{\sqrt{d_k}} \in \mathbb{R}^{L \times L}, \quad \text{Attention}(Q, K, V) = \sigma (A)V \in \mathbb{R}^{L \times d_k},
\end{equation}$$ 

where $d_k$ acts as a scaling factor. We then calculate
the attention loss with 

$$\begin{equation} \tag{9}\label{eq:lossatt}
    L_{\text{MSEAtt}} = \frac{1}{h} \sum_{i=1}^h \text{MSE}(A_i^{ST}, A_i^{TE}),
\end{equation}$$ 

where $h$ is the number of attention heads,
$A_i \in \mathbb{R}^{L \times L}$ refers to the attention matrix before
applying the softmax, the index $i$ corresponds to the $i$-th head of
teacher or student, $L$ is the input text length, and MSE$(\cdot)$ means
the mean squared error loss function. They argue that using $A$ instead
of $\text{Attention}(Q, K, V)$  show faster convergence rate and
better performances. We visualize the TinyBert attention and
intermediate hidden representation distillation in the next Figure:

<figure>
<img src="/assets/img/master-thesis/layer_distill.PNG" alt="interview-img" width="500">
<em>Visualization of a attention and intermediate hidden representation
distillation. Taken from (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>).</em>
</figure>

Nonetheless (<a href="#ref-aguilar2020knowledge">Aguilar et al. (2020)</a>) make use of the attention matrix
$\text{Attention}(Q, K, V)$ and chose the KL-divergence loss.
For a given head in a transformer, the loss can be formulated as

$$\begin{aligned}
    L_{\text{KLAtt}} = \frac{1}{L} \sum_i^L {\tt Att}(Q^{TE}, K^{TE}, V^{TE})_i \log \left(\frac{Att(Q^{TE}, 
K^{TE}, V^{TE})_i}{Att(Q^{ST}, K^{ST}, V^{ST})_i} \right),
\end{aligned}$$ 

where $Att(Q, K, V)_i = \text{Attention}(Q, K, V)_i \in  \mathbb{R}^{d_k}$
describe the $i$-th row of the attention probability matrix.

MiniLM (<a href="#ref-wang2020minilm">Wang et al. (2020)</a>)  additionally aligns the relation between values
in the self-attention module, which is calculated via the multi-head
scaled dot-product between values $V^{ST}$ and $V^{TE}$.

**Embedding Distillation.** Multiple lines of work indicate that the
embedding layer is important for a successful cross-lingual
representation (<a href="#ref-pires-etal-2019-multilingual">Pires et al. (2019)</a>; <a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>; <a href="#ref-dufter2021identifying">Dufter et al. (2021)</a>).
Therefore, it is also important to distill from our multilingual
teacher's embedding layer to our students.

DistilBert uses the cosine embedding loss to align the embeddings of
teacher and student: 

$$\begin{aligned}
    L_{\text{COS}embed} = 1 - \text{cos}(E^{ST}_m, E^{TE}).
\end{aligned}$$ 

where the matrices $E^{ST}$ and $E^{TE}$ refer to the
embeddings of student and teacher networks, respectively. Instead of
using the cosine loss, TinyBert uses the MSE: 

$$\begin{equation}
    L_{\text{Emb}\_MSE} = \text{MSE}(E^{ST} W_e, E^{TE}).
\end{equation}$$ 

Again, TinyBert allows for a mismatch between dimensions
of $E^{ST}$ and $E^{TE}$ by using a learnable linear transformation
$W_e$.

XtremeDistil (<a href="#ref-mukherjee2020xtremedistil">Mukherjee et al. (2020)</a>)  uses Singular Value
Decomposition (SVD) to project the word embeddings of the teacher to a
lower-dimensional space for the student since they use the same
WordPiece vocabulary.

# 4. Distillation Setup Strategies

This thesis introduces a novel distillation setup to induce aligned
monolingual students, which is why we review different distillation
setup strategies for transformers in this section. We restrict our
literature review to a \"fixed\" teacher at the distillation time and
not an, e.g., jointly trained teacher-student setup, such as in
(<a href="#ref-jin2019knowledge">Jin et al. (2019)</a>). Furthermore, we focus on the general distillation
stage, where we only perform distillation on the Masked Language
Modeling objective. Finally, the task is to distill knowledge from the
multilingual teacher(s) to our student while MLM pre-training on
multiple monolingual text corpora, i.e., multilingual corpus, obtaining
a general-purpose student.

**One Teacher, One student.** In this setup, only one teacher exists and
is distilled into one student. (<a href="#ref-reimer_sbert">Reimers et al. (2020)</a>) use parallel data to
facilitate strong cross-lingual sentence representations by training the
student model such that (1) identical sentences in different languages
are close and (2) the original source language from the teacher model
SBERT (<a href="#ref-sbert_reimers_2019">Reimers et al. (2019)</a>)  are adopted and transferred to other
languages.


They do so by minimizing the mean squared error of (1) the source
sentence embedding of the teacher with the target sentence embedding
using parallel data and (2) the source sentence embedding of the teacher
and the student, see the next figure for a visualization:

<figure>
<img src="/assets/img/master-thesis/sbert_kd.PNG" alt="interview-img">
<em>Visualization of the distillation strategy of (<a href="#ref-reimer_sbert.">reimer_sbert.</a>) Given
parallel data (here: English and German), the student model is trained
such that the sentence embeddings for the English and German sentences
are close to the teacher English sentence vector. Adopted from
(<a href="#ref-reimer_sbert">Reimers et al. (2020)</a>).</em>
</figure>

Furthermore, all discussed monolingual distillation approaches can be
extended to create a multilingual student straightforwardly: Distill a
multilingual teacher into a student during the MLM task on a
multilingual corpus. Since the teacher is already multilingual and the
representations aligned (<a href="#ref-pires-etal-2019-multilingual">Pires et al. (2019)</a>; <a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>), the cross-lingual
knowledge can then be distilled into one student. Consequently, every
monolingual distillation approach can be directly applied, e.g. PKD (<a href="#ref-sun2019patient">Sun et al. (2019)</a>), DistilBert (<a href="#ref-sanh2020distilbert">Sanh et al. (2020)</a>)  or TinyBert
 (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>). We can generalize the distillation loss as a
*convex combination* or a *linear combination* of all chosen loss
functions. E.g. DistilBert uses a convex combination of the masked
language modeling loss $L\_{\text{MLM}}$, the
Hinton Loss $L_H$ \eqref{eq:hinton_loss} and the cosine embedding loss
$L_{\text{COS}embed}$ \eqref{eq:coshidn}: 

$$\begin{aligned}
    L_{\text{DB}} = \alpha \cdot L_H + \beta \cdot L_{MLM} + \gamma \cdot L_{\text{COS}embed}(\ \cdot \ ; M),
\end{aligned}$$ 

where $M$ is the index of the last hidden layer, $\alpha, \beta, \gamma \in [0, 1]$ with $\alpha + \beta + \gamma = 1$.


**Multiple Teacher, One Student.** Language-specific language models may
perform better given a sizable pre-training data volume than
multilingual teachers in their respective language but are, in turn,
monolingual and do not use any positive language transfer. Distilling
multiple task-agnostic language-specific teachers into one task-agnostic
student can help the student be competitive or outperform the individual
language-specific LMs while still being multilingual.
(<a href="#ref-khanuja2021mergedistill">Khanuja et al. (2021)</a>) merges multiple monolingual or multilingual
pre-trained LMs into a single task-agnostic multilingual student using
task-agnostic Knowledge Distillation, which they call *MergeDistill*.
The difficulty is that each LM can have its own vocabulary.
(<a href="#ref-khanuja2021mergedistill">Khanuja et al. (2021)</a>) use the union of all teacher LM vocabularies
for the student vocabulary. They use a vocab mapping step *teacher
$\rightarrow$ student*, converting each teacher token index to its
corresponding student token index. They first tokenize and predict for
each language using their respective teacher LM and get the top-$k$
logits for each masked word. For distillation, they then use the Hinton
Loss $L_H$ $\eqref{eq:hinton_loss}$ and MLM loss $L_\text{MLM}$.
Interestingly, their experiments show that due to the shared
multilingual representations, the student is able to perform in a
zero-shot manner on related languages that the teacher does not cover.

**One Teacher, Multiple Students.** In this setup, we have one
multilingual teacher and want to distill into multiple mono-, bi- or
multilingual students, for which the representations for each language
are aligned across students. In this work, we will focus on distilling
into multiple monolingual students. To the best of our knowledge, this
is the first work that investigates distilling from a multilingual
teacher into monolingual students sharing a representation space.

# 5. Challenges

The trend towards bigger and bigger models in NLP is fueled by the high
generalization power of the learned representations. Smaller models lack
the inductive biases to learn these representations from the training
data alone but may have the capacity to represent these solutions
 (<a href="#ref-ba_deep_2013">Ba et al. (2013)</a>; <a href="#ref-stanton2021does">Stanton et al. (2021)</a>). We discussed several methods such as
DistilBert (<a href="#ref-sanh2020distilbert">Sanh et al. (2020)</a>)  and TinyBert (<a href="#ref-jiao2020tinybert">Jiao et al. (2020)</a>)  that
achieve similar performances on some downstream tasks as the teacher
with just a fraction of the total parameter number. In the previous
sections, we discussed different Knowledge Distillation strategies to
induce knowledge into the student, e.g., the effects of distilling
different transformer components into the student. However, we highlight
two open challenges in regards to Knowledge Distillation in general
(Fidelity vs. Generalization) and our thesis (KD for Monolingual
Students).

**Fidelity vs. Generalization.** Recently, (<a href="#ref-stanton2021does">Stanton et al. (2021)</a>) show that
while Knowledge Distillation can improve the generalization abilities of
students, there often remains a low fidelity, i.e., the ability of a
student to match a teacher's predictions. Previous works
 (<a href="#ref-furlanello_2018">Furlanello et al. (2018)</a>; <a href="#ref-mobahi_2020">Mobahi et al. (2020)</a>) already show that in self-distillation,
the student can improve generalization. This can only happen by virtue
of failing at the distillation procedure: The student fails to match the
teacher:

<figure>
<img src="/assets/img/master-thesis/fidelity.PNG" alt="interview-img">
<em>The effect of enlarging the CIFAR-100 distillation dataset with
GAN-generated samples. The shaded region corresponds to
$\mu \pm \sigma $, estimated over three trials. (a) The teacher and
student have the same model capacity. Student fidelity increases as the
dataset grows, but the test accuracy decreases. (b) The teacher has a
larger model capacity than the student. Student fidelity again increases
when the dataset grows, but the test accuracy now also slightly
increases. Figure taken from
(<a href="#ref-stanton2021does">Stanton et al. (2021)</a>).</em>
</figure>


These experiments, however, hold true for students that have the *same
model capacity* as the teacher. Often there is a significant disparity
in generalization between large teacher models and smaller students.
Importantly, (<a href="#ref-stanton2021does">Stanton et al. (2021)</a>) then show that for these larger teacher
models, improvements in fidelity translate into improvements in
generalization (Figure (b)).

**KD for Monolingual Students.** As this is the first work that explores
distilling multilingual encoders into monolingual components (to the
best of our knowledge), the question remains which parts of the teacher
are important to distill from to improve (1) alignment and (2)
cross-lingual downstream task performance. Another open question is
whether sharing between students and weight initialization from the
teacher improves (1) and (2). Finally, as we have multiple students, we
can not utilize the default approach to solve cross-lingual downstream
tasks: Fine-tuning one multilingual model in the source language and
evaluating/train with the same model in the target language. We must
explore fine-tuning strategies for multiple (monolingual) students for
cross-lingual downstream tasks.

# References


<ol class="bibliography">
  <li id="ref-Bucila2006ModelC">Cristian Bucila, Rich Caruana, Alexandru Niculescu-Mizil. <em>Model compression</em>. KDD '06. 2006</li>
  <li id="ref-du2018power">Simon S. Du, Jason D. Lee. <em>On the Power of Over-parametrization in Neural Networks with Quadratic Activation</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1803.01206">link</a></li>
  <li id="ref-soltanolkotabi2018theoretical">Mahdi Soltanolkotabi, Adel Javanmard, Jason D. Lee. <em>Theoretical insights into the optimization landscape of over-parameterized shallow neural networks</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1707.04926">link</a></li>
  <li id="ref-Hao_2019">Yaru Hao, Li Dong, Furu Wei, Ke Xu. <em>Visualizing and Understanding the Effectiveness of BERT</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1908.05620">link</a></li>
  <li id="ref-conneau2020unsupervised">Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. <em>Unsupervised Cross-lingual Representation Learning at Scale</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1911.02116">link</a></li>
  <li id="ref-dufter2021identifying">Philipp Dufter, Hinrich Schütze. <em>Identifying Necessary Elements for BERT's Multilinguality</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2005.00396">link</a></li>
  <li id="ref-ba_deep_2013">Lei Jimmy Ba, Rich Caruana. <em>Do Deep Nets Really Need to be Deep?</em>. arXiv. 2013. <a href="https://arxiv.org/abs/1312.6184">link</a></li>
  <li id="ref-hinton2015distilling">Geoffrey Hinton, Oriol Vinyals, Jeff Dean. <em>Distilling the Knowledge in a Neural Network</em>. arXiv. 2015. <a href="https://arxiv.org/abs/1503.02531">link</a></li>
  <li id="ref-jiao2020tinybert">Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. <em>TinyBERT: Distilling BERT for Natural Language Understanding</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1909.10351">link</a></li>
  <li id="ref-Krizhevsky_2012">Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton. <em>ImageNet Classification with Deep Convolutional Neural Networks</em>. Neural Information Processing Systems. 2012. <a href="https://doi.org/10.1145/3065386">link</a></li>
  <li id="ref-deng2009imagenet">Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei. <em>Imagenet: A large-scale hierarchical image database</em>. 2009 IEEE conference on computer vision and pattern recognition. 2009</li>
  <li id="ref-Gou_2021">Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao. <em>Knowledge Distillation: A Survey</em>. International Journal of Computer Vision. 2021. <a href="https://doi.org/10.1007%2Fs11263-021-01453-z">link</a></li>
  <li id="ref-sun2019patient">Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu. <em>Patient Knowledge Distillation for BERT Model Compression</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1908.09355">link</a></li>
  <li id="ref-romero2015fitnets">Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio. <em>FitNets: Hints for Thin Deep Nets</em>. arXiv. 2015. <a href="https://arxiv.org/abs/1412.6550">link</a></li>
  <li id="ref-mukherjee2020xtremedistil">Subhabrata Mukherjee, Ahmed Awadallah. <em>XtremeDistil: Multi-stage Distillation for Massive Multilingual Models</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2004.05686">link</a></li>
  <li id="ref-clark2019does">Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. <em>What Does BERT Look At? An Analysis of BERT's Attention</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1906.04341">link</a></li>
  <li id="ref-aguilar2020knowledge">Gustavo Aguilar, Yuan Ling, Yu Zhang, Benjamin Yao, Xing Fan, Chenlei Guo. <em>Knowledge Distillation from Internal Representations</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1910.03723">link</a></li>
  <li id="ref-liu2019improving">Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao. <em>Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1904.09482">link</a></li>
  <li id="ref-turc2019wellread">Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. <em>Well-Read Students Learn Better: On the Importance of Pre-training Compact Models</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1908.08962">link</a></li>
  <li id="ref-tang2019distilling">Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy Lin. <em>Distilling Task-Specific Knowledge from BERT into Simple Neural Networks</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1903.12136">link</a></li>
  <li id="ref-kaliamoorthi2021distilling">Prabhu Kaliamoorthi, Aditya Siddhant, Edward Li, Melvin Johnson. <em>Distilling Large Language Models into Tiny and Effective Students using pQRNN</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2101.08890">link</a></li>
  <li id="ref-tan2019multilingual">Xu Tan, Yi Ren, Di He, Tao Qin, Zhou Zhao, Tie-Yan Liu. <em>Multilingual Neural Machine Translation with Knowledge Distillation</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1902.10461">link</a></li>
  <li id="ref-clark2019bam">Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D. Manning, Quoc V. Le. <em>BAM! Born-Again Multi-Task Networks for Natural Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1907.04829">link</a></li>
  <li id="ref-liu2020mkd">Linqing Liu, Huan Wang, Jimmy Lin, Richard Socher, Caiming Xiong. <em>MKD: a Multi-Task Knowledge Distillation Approach for Pretrained Language Models</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1911.03588">link</a></li>
  <li id="ref-devlin2019bert">Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. <em>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1810.04805">link</a></li>
  <li id="ref-sanh2020distilbert">Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf. <em>DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1910.01108">link</a></li>
  <li id="ref-wang2019glue">Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman. <em>GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1804.07461">link</a></li>
  <li id="ref-sun2020mobilebert">Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou. <em>MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2004.02984">link</a></li>
  <li id="ref-wang2020minilm">Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou. <em>MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2002.10957">link</a></li>
  <li id="ref-khanuja2021mergedistill">Simran Khanuja, Melvin Johnson, Partha Talukdar. <em>MergeDistill: Merging Pre-trained Language Models using Distillation</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2106.02834">link</a></li>
  <li id="ref-yang2021knowledge">Jing Yang, Brais Martinez, Adrian Bulat, Georgios Tzimiropoulos. <em>Knowledge distillation via softmax regression representation learning</em>. ICLR2021. 2021</li>
  <li id="ref-pires-etal-2019-multilingual">Telmo Pires, Eva Schlinger, Dan Garrette. <em>How Multilingual is Multilingual {BERT}?</em>. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. <a href="https://aclanthology.org/P19-1493">link</a></li>
  <li id="ref-wu-dredze-2019-beto">Shijie Wu, Mark Dredze. <em>Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of {BERT}</em>. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019. <a href="https://aclanthology.org/D19-1077">link</a></li>
  <li id="ref-jin2019knowledge">Xiao Jin, Baoyun Peng, Yichao Wu, Yu Liu, Jiaheng Liu, Ding Liang, Junjie Yan, Xiaolin Hu. <em>Knowledge Distillation via Route Constrained Optimization</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1904.09149">link</a></li>
  <li id="ref-reimer_sbert">Nils Reimers, Iryna Gurevych. <em>Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation</em>. 2020. <a href="https://doi.org/10.18653/v1/2020.emnlp-main.365">link</a></li>
  <li id="ref-sbert_reimers_2019">Nils Reimers, Iryna Gurevych. <em>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1908.10084">link</a></li>
  <li id="ref-reimer_sbert.">reimer_sbert.</li>
  <li id="ref-stanton2021does">Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson. <em>Does Knowledge Distillation Really Work?</em>. arXiv. 2021. <a href="https://arxiv.org/abs/2106.05945">link</a></li>
  <li id="ref-furlanello_2018">Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar. <em>Born Again Neural Networks</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1805.04770">link</a></li>
  <li id="ref-mobahi_2020">Hossein Mobahi, Mehrdad Farajtabar, Peter L. Bartlett. <em>Self-Distillation Amplifies Regularization in Hilbert Space</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2002.05715">link</a></li>
</ol>


[^1]: Knowledge Distillation can also be referred to as teacher-student
    Knowledge Distillation.



