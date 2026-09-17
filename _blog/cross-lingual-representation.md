---
title: "[Paper] Survey: Cross-Lingual Representation Learning"
date: 2022-05-26 11:30:47 +01:00
tags: [paper, nlp]
description: Master Thesis
image: "/assets/img/cross-lingual/introduction.jpg"
---
*Part of the related work section in my Master Thesis. Download the full version
<a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>.*


<figure>
<img src="/assets/img/cross-lingual/introduction.jpg" alt="interview-img">
</figure>


# Table of Contents
1. [Motivation](#motivation)
2. [Constructing Word Representations](#constructing-word-representations)
3. [Evolution of Architectures](#evolution-of-architectures)
4. [Cross-Lingual Static Word Representations](#cross-lingual-static-word-representations)
5. [Cross-Lingual Contextualized Word Representations](#cross-lingual-contextualized-word-representations)
6. [Challenges in Multilingual Transformers](#challenges-in-multilingual-transformers) 


# Motivation

The latest NLP technology relies on pre-training on massive amounts of
text in the respective language in an unsupervised fashion, producing
fixed-size sequence or word representations that can be used to
fine-tune on a task with sufficient labeled data. In most cases, both
data sources are needed to meet the performance of state-of-the-art NLP
approaches. However, the lack of both data sources will highly degrade
the performance of these methods, posing a fundamental problem in
scaling low-resource languages. Cross-lingual representation techniques
try to alleviate the issue of data scarcity for low-resource languages
by inducing an aligned representation across languages, i.e.,
language-agnostic language representations. The idea is to transfer
lexical, syntactic, and semantic knowledge across languages that can be
used for cross-lingual downstream tasks. This gives rise to two
advantages: (1) Transferring lexical knowledge across languages enables
us to reason about the semantics of words in multilingual contexts and
is a vital source of knowledge for multilingual systems such as machine
translation (<a href="#ref-artexe_mt qi-etal-2018-pre lample-etal-2018-phrase">artexe_mt qi-etal-2018-pre lample-etal-2018-phrase</a>),
multilingual search, and question answering (<a href="#ref-10.1145/2766462.2767752">Vuli\'{c} et al. (2015)</a>).
(2) More importantly, given a downstream task, models can utilize the
joint representation space by training on a high-resource language such
as English, where labeled data exists, to then use the acquired task
knowledge to transfer it to other (low-resource) languages. The hope is
that the model can generalize lexical properties and relations across
languages (<a href="#ref-10.1162/089120102762671990">Plath (2002)</a>). Ultimately, cross-Lingual
representations can also be seen as a type of transfer learning which
can help us understand why transferring knowledge across languages
works.


**Relation to Transfer Learning.** Transfer Learning is a sub-field in
Machine Learning that focuses on reusing the acquired knowledge from
past related tasks to help the learning process of solving a new task.
Cross-Lingual Representation Learning, therefore, is a type of transfer
learning, specifically similar to domain adaption; see Figure
<a href="#trans_learning" data-reference-type="ref"
data-reference="trans_learning">1</a> for a taxonomy of transfer
learning in NLP.

<figure>
<img src="/assets/img/master-thesis/transfer_learning.PNG" id="trans_learning"  alt="A taxonomy 
for 
transfer learning for NLP. Given a downstream task, Cross-Lingual Representation Learning utilizes the joint representation space by fine-tuning on a high-resource language to then use the acquired task knowledge to transfer it to other (low-resource) languages. Adopted from ." />
<figcaption aria-hidden="true">A taxonomy for transfer learning for NLP.
Given a downstream task, Cross-Lingual Representation Learning utilizes
the joint representation space by fine-tuning on a high-resource
language to then use the acquired task knowledge to transfer it to other
(low-resource) languages. Adopted from <span class="citation" data-cites="noauthororeditor"></span>.</figcaption>
</figure>

Viewing cross-lingual representation as a form of transfer learning can
help us understand in which cases knowledge from one to another language
can be transferred:
- Transfer Learning works well when the *underlying structures are
    similar to each other*. Languages share many aspects on many
    different levels, e.g., on a lexical level, languages can
    incorporate words from another language (loanwords) and have words
    from the same origin (cognate). On a syntactical level, languages
    might have a similar structure of sentences, and on a semantic level
    languages, languages are built upon a so-called natural semantic
    metalanguage, see (<a href="#ref-Goddard2006NaturalSM">Goddard (2006)</a>) for a more in-depth
    analysis.
- Transfer Learning fails when the source and target settings are
    vastly different. In our cross-lingual setting, transferring any
    knowledge from languages that are not related in any way is hard,
    i.e., languages that are not typologically or etymologically
    related.

In summary, languages must be in some way related; otherwise, we can not
transfer knowledge across languages. Languages in the same language
family share much more than unrelated languages. This is also reflected
in the performance of cross-lingual methods
(<a href="#ref-lample2019crosslingual  lauscher-etal-2020-zero">lample2019crosslingual  lauscher-etal-2020-zero</a>). Furthermore, even
when languages come from different language families,
(<a href="#ref-Goddard2006NaturalSM">Goddard (2006)</a>) argues that, on a semantic level, languages are
built upon a natural semantic metalanguage, therefore, share a
connection.

# Constructing Word Representations

The question remains how one can exploit shared structures across
languages to build a cross-lingual representation. As word
representations, i.e., words represented as real-valued vectors, are the
basis of modern NLP and cross-lingual representations, we first discuss
different approaches to built word representations.

**Bag-of-Words.** The simplest solution to represent words in a vector
form is to use the bag-of-words approach, which describes the occurrence
of words within a document. However, the bag-of-word approach has major
problems such as losing information about word order and semantics,
being highly dimensional (curse of dimensionality), and can not
represent out-of-vocabulary tokens.

**Distributed Word Representation.** To improve upon bag-of-words,
distributed word representations were utilized, which represent words
(or more generally, tokens[^1]) as distributed representations of lower
dimensionality, trained to capture syntactic and semantic relationships
between words. The approach is motivated by the underlying idea that \"a
word is characterized by the company it keeps\", called the
*distributional hypothesis* (<a href="#ref-doi:10.1080/00437956.1954.11659520">Harris (1954)</a>). This
means that words that occur in the same context are semantically
related. The general approach to generating distributed word embeddings
is by computing word co-occurrence statistics based on unlabeled free
text (unsupervised).

**Language Models.** The most prominent way to make use of the
distributional hypothesis to create distributed word representations is
by using language models (LMs), which is *the* backbone of modern NLP.
LMs first gained momentum when (<a href="#ref-collobert_2008">Collobert et al. (2008)</a>) showed the effectiveness
of applying neural models to language modeling to create semantic word
embeddings for downstream tasks. It was the start point of the modern
approach to NLP: Pre-train neural models to create word representations
for downstream tasks. Formally, an LM, given a sequence of tokens
$\{t_1, ..., t_m\}$ with length $m$, outputs a probability distribution
over the sequence of tokens, i.e., the likelihood of the tokens
occurring jointly:

$$
\begin{equation} \label{eq:lm_likelihood}
    P(t_1, ..., t_m) \stackrel{\text{(chain rule)}}{=} P(t_1) \cdot P(t_2 \vert t_1) \cdot ... \cdot P(t_m\vert t_1,...,t_{m-1}) = \prod_{i=1}^m P (t_i \vert t_{i:i-1}).
\end{equation}
$$ 

The chain rule allows us to calculate the joint
probability of the tokens in the sequence as conditional probabilities
of a token given their respective previous tokens. Even then, this
becomes quickly intractable due to combinatorial explosion of the
possible number of previous tokens. To avoid this problem, typically we
leverage the Markov assumption, i.e., it is assumed that the probability
of $P(t_m\vert t_1,...,t_{m-1})$ depends only on the previous $n-1 << m$
tokens: 
$$
P(t_m\vert t_1,...,t_{m-1}) \approx P(t_m\vert t_{m-(n-1)},...,t_{m-1}) \implies P(t_1, ..., t_m) & \approx \prod_{i=1}^m P (t_i \vert t_{i-(n-1):i-1}).
$$ 

$$
\implies P(t_1, ..., t_m) \approx \prod_{i=1}^m P (t_i \vert t_{i-(n-1):i-1}).
$$ 

As the joint probability of tokens now only depend on
the product of probabilities of the form 

$$
\begin{equation} \label{eq:lm}
    P(t_i \vert t_{i-(n-1)}, ..., t_{i-1}),
\end{equation}
$$

called $n$-grams, we need to estimate the $n$-grams
which can be done with the maximum likelihood estimate (MLE) on the
training corpus. In practice, models are trained to either predict the
subsequent tokens (directional) or to predict the missing token given
the surrounding context of the word (bidirectional).

To evaluate the performance of a language model, the usual metric is the
*Perplexity* (<a href="#ref-Jelinek1977PerplexityaMO">Jelinek et al. (1977)</a>), which is defined as the
inverse probability of sequences on a validation set normalized by the
number of tokens: 

$$\begin{equation}
    PP(t_1,...,t_{m-1}) = \sqrt[m]{\frac{1}{P(t_1, t_2, ...,t_{m})}} \stackrel{\text{(chain rule)}}{=} \sqrt[m]{\frac{1}{\prod_{i=1}^m P (t_i \vert t_{i:i-1})}}.
\end{equation}$$ 

A lower perplexity indicates a better model.

The following subsection will explore different model architectures that
utilize language modeling to create powerful word embeddings.

# Evolution of Architectures

We outline the evolution of (neural) architectures in NLP to induce
strong word representations that can be utilized for downstream tasks,
such as Natural Language Understanding. For now, we restrict ourselves
to inducing monolingual word representations using the language modeling
task on *monolingual* text corpus.

**Feedforward Neural Networks.**
(<a href="#ref-mikolov2013exploiting">Mikolov et al. (2013)</a>) introduce an efficient way of learning
high-quality word vectors for millions of words in the vocabulary from a
large amount of unlabeled text data in an unsupervised fashion. The
released word embeddings not only capture semantic and syntactic
information but also learn relationships between words[^2], e.g.,
*Paris - France + Italy = Rome*. They dub their approach word2vec and
give two novel model architectures: Skip-Gram (SG) and Continuous
Bag-of-Words (CBOW).

Both CBOW and SG architectures are based on a simple feedforward neural
network. The CBOW method computes the probability of the current token
based on the context tokens within a window $W$ of $k$ neighboring
tokens. On the other hand, the SG computes the probability of
surrounding tokens within a window $W$ of $k$ neighboring tokens given
the current token. The network encodes each token $t_i$ into a center
token $e_i$ and context token $c_i$ which correspond to the $i-$th row
of the center token matrix $E^{\vert V \vert \times d}$ and context
token matrix $E^{\vert V \vert \times d}$, where the $V$ is the size of
the vocabulary and $d$ the token vector embedding size. Given a center
token $t$, SG estimates the likelihood of seeing a context token $w$
conditioned on the given center token with the softmax function:

$$\begin{equation} \label{eq:word2vec_likelihood}
    P(c_w \vert e_t) = \frac{\exp{e_t^Tc_w}}{\sum_{i=1}^N \exp{e_t^T c_i} },
\end{equation}
$$ 

where $e_t$ denote the embedding for the center token
$t$ and $c_w$ the embedding for the context token $w$ in the window
$c_w \in W$. Given a text corpus $\{t_1, ..., t_T \}$ of length $T$ and
assuming that the context words are independently generated given any
center word, we learn the model parameters (token embeddings) by
maximizing the likelihood function over the text corpus which is
eqivalent to minimizing the negative log-likelihood: 

$$
\begin{equation} \label{eq:word2vec}
    \max_{e,c}  \prod_{t=1}^T \prod_{w \in W_t} P(c_w \vert e_t)  \Leftrightarrow \min_{e,c} - \sum_{t=1}^T \sum_{w \in W_t} \log \left( P(c_w \vert e_t) \right).
\end{equation}
$$ 

For a downstream task, the final embedding for token $t$
is either the center token or calculated as the element-wise average or
the sum of its center and context representations. Therefore, the
word2vec objective directly uses the language modeling task to
generate effective word embeddings.

Even though word2vec is very effective in creating powerful word
representations, there are some considerable drawbacks: First, the
denominator in the Equation of the word2vec likelihood sums over the entire vocabulary,
slowing down the calculating of the softmax. There are some approaches
such as hierarchical softmax and negative sampling to overcome this
(<a href="#ref-mikolov2013exploiting">Mikolov et al. (2013)</a>). Still, there are two major conceptually
disadvantages of word2vec representations: First, they can not embed any
tokens outside the vocabulary, and second, they do not account for the
linguistic morphology of a word, e.g., the representations of \"eat\"
and \"eaten\" are learned separately (no parameter-sharing) based on its
context they appear on.

To solve the above issues, (<a href="#ref-fasttext_2016">Bojanowski et al. (2016)</a>) introduce FastText, a new
embedding method. FastText extends the idea of word2vec by using the
internal structure of a word to improve the word representations of
word2vec. Instead of constructing representations for words, FastText
learns representations for n-grams of characters which are then used to
build word representations by summing the bag-of-character n-grams up.
E.g., for $n=3$, the word *\"artificial\"* is represented by $<$*ar,
art, rti, tif, ifi, fic, ici, ial, al*$>$ where the angular brackets
indicate the beginning and end of the word. This allows us to better
represent and capture the meaning of suffixes and prefixes. Furthermore,
words that do not appear during training can be represented by breaking
the word into $n$-grams to get its representation.

The released representations of FastText and word2vec became famous
because of their ease of use and effectiveness in a variety of NLP
problems (<a href="#ref-lample_2016  kiros_2015 Kusner2015FromWE">lample_2016  kiros_2015 Kusner2015FromWE</a>). Furthermore,
these (monolingual) representations can be used to construct a
cross-lingual representation space by mapping representations of
multiple languages into a shared space (see Section 3).

However, word2vec and FastText have several drawbacks: Each word has a
static word representation. Consequently, both methods can not correctly
capture phrases and polysemy of words. Furthermore, during training, we
only consider the context of a word, leading to a similar representation
of a word and its anatomy since both appear in a similar context.
Another drawback is that we only consider a fixed-size window of context
words for conditioning the language model. A more natural way to learn
representation is to allow a variable amount of context words.


**Recurrent Neural Network (RNN).** A RNN is a class of artificial
neural networks that are specialized to process sequential data, e.g.,
natural language text. RNNs are capable of conditioning the model to an
arbitrary number of words in the sequence. 
![image](/assets/img/master-thesis/rnn.PNG)
The above Figure 
depicts the architecture of an uni-directional RNN where each vertical
box is a hidden layer at a time-step $t$. At each time step $t$, the
hidden layer gets two inputs: the output of the previous layer $h_{t-1}$
and the input at that time-step $x_t$.

<img src="/assets/img/master-thesis/rnn_input_output.PNG" alt="drawing" width="500"/>

To produce the output features $h_t$ and to obtain a
prediction output $\hat{y}$ of the next word, we utilize the weight
matrices $W^{(hh)}, W^{(hx)}, W^{(S)}$ as follows: 

$$
    h_t = \sigma\left(W^{(hh)} h_{t-1} + W^{(hx)} x_{[t]} 
    \right) 
$$

$$
    \hat{y}_t = \text{softmax}\left(W^{(S)} h_{t} \right)

$$ 

Notice that the weights $W^{(hh)}, W^{(hx)}$ are applied
repeatedly at each time step, therefore sharing the weights across time
steps. This allows the model to process sequences of arbitrary length.
Furthermore, the model size does not increase with longer input
sequences. In theory, RNNs can use information from any steps from the
past. However, in practice, this is difficult as the vanishing and
exploding gradients become a big issue with long sequences
(<a href="#ref-vanishing_1998 Bengio1994LearningLD">vanishing_1998 Bengio1994LearningLD</a>) which then makes the model
insensitive to past inputs. To alleviate these issues, we mention some
heuristic solutions: Clipping the gradient to a small number whenever
they explode (<a href="#ref-pascanu_2021">Pascanu et al. (2012)</a>) , initialization of $W^{(hh)}$ as the
identity matrix since it helps avoid the vanishing gradients
(<a href="#ref-Le2015ASW">Le et al. (2015)</a>)  and using the Rectified Linear Units (ReLU) instead of the
sigmoid function (<a href="#ref-relu_2018">Agarap (2018)</a>) . However, one of the most important
extensions to solve the vanishing gradient problem is the so-called
long-short term memory (LSTM) (<a href="#ref-lstm_1997 gers_2000">lstm_1997 gers_2000</a>) , which is a
sub-architecture for the hidden layer of an RNN. The LSTM unit
introduces a gating mechanism that selectively propagates only a subset
of relevant information across time steps and consequently mitigates the
vanishing gradient problem.

RNNs and LSTMs started to dominate NLP, either performing competitively
or outperforming existing state-of-the-art on various tasks
(<a href="#ref-sutskever_2011 mikolov_rnn_2011 Sutskever2014SequenceTS">sutskever_2011 mikolov_rnn_2011 Sutskever2014SequenceTS</a>) . One
particular interesting architecture emerged to address tasks where a
output sequence was needed such as machine translation: The
*encoder-decoder* architecture. The architecture was first proposed by
(<a href="#ref-Hinton1993AutoencodersMD">Hinton et al. (1993)</a>) and was then later used in the context of NLP
(<a href="#ref-Kalchbrenner2013RecurrentCT Sutskever2014SequenceTS">Kalchbrenner2013RecurrentCT Sutskever2014SequenceTS</a>).

<figure>
<img src="/assets/img/master-thesis/encoder_decoder2.png" id="undirectedGraph" style="width:14cm" alt="For big 
data, deep learning methods perform much better than traditional machine learning algorithms." />
<figcaption aria-hidden="true">Encoder-decoder architecture of an RNN on a machine translation task.
The encoder produces a thought vector $C$, representing the source
sentence. The decoder then unfolds state $C$ to produce an output
sequence.</figcaption>
</figure>

The encoder part takes a sequence as an input and outputs a single
vectorized representation of the whole sequence (called thought vector),
which the decoder takes as an input to generate a sequence.

However, since the thought vector has a fixed size representation form,
and the decoder only depends on the thought vector, the representative
power of the (uni-directional) RNN encoder-decoder architecture for
sequences is naturally limited. All the information about the input has
to be encoded into the fixed-size thought vector, which becomes
increasingly difficult for long sequences.

**Attention.** To improve upon the above described shortcoming of
encoder-decoders, (<a href="#ref-attention_2014">Bahdanau et al. (2014)</a>) introduces the concept of *attention*.
The attention module allows the decoder to re-access and select encoder
hidden states at decoding time.
<figure>
<img src="/assets/img/master-thesis/attention_module.PNG" alt="interview-img" width="800">
<em>An encoder-decoder model for machine translation with added attention
mechanism.</em>
</figure>

Following the numbers in the above Figure, the attention module can be explained
by the following: (1) The decoder's hidden state and (2) the
intermediate hidden states of the encoder are being fed into the
attention module. (3) The attention module then selects relevant
information from the hidden states of the encoder based on the decoder's
hidden state and calculates the context vector. Finally, (4) the decoder
takes the context vector and the last output word (\"cómo\") as the
input and outputs the next word. {\cite Galassi_2021 } gives an exhaustive
overview of different attention implementations. However, we restrict
ourselves to the common attention mechanism used in transformers,
explained in the next paragraph.

First, the decoder state $\mathbf{h}_D \in \mathbb{R}^{1 \times d}$ is embed
to a *query* $\mathbf{q} \in \mathbb{R}^{1 \times d_k}$ using a learnable
weight matrix $W_Q  \in \mathbb{R}^{d \times d_k}$: 

$$
    \mathbf{q} = \mathbf{h}_D W_Q
$$ 

and each encoder state $\mathbf{h}_E^{(i)}$, where $i$
denotes the encoder time-step, is stacked to a encoder state matrix
$\mathbf{H}_E$ and is used to produce the *key* matrix $\mathbf{K}$ and *value*
matrix $\mathbf{V}$: 

$$
    \mathbf{K} =  \mathbf{H}_E W_K, \quad \mathbf{V} = \mathbf{H}_E W_V,
$$ 

where
$W_K \in \mathbb{R}^{d \times d_k}, W_V \in \mathbb{R}^{d \times d_v}$
are learnable weights. We then calculate the *attention weights* which
computes how relevant a single key $\mathbf{k}^{(i)}$ vector is for the
query $\mathbf{q}$: 

$$
    w =  \mathbf{q} \mathbf{K}^T.
$$ 

Commonly, the weights are normalized to a probability
distribution using the softmax function which are then used to create
the *context vector* $\mathbf{c}$ by taking the weighted average of the
values: 

$$
    \mathbf{c} = \sum_i a^{(i)} \mathbf{v}^{(i)} \quad \text{ where } \quad a^{(i)} = \text{softmax}(w)_i.
$$ 

During training, we optimize the weights $W_Q, W_K, W_V$
which then improves the selective focus of the attention module. We can
summarize the (dot-product) Attention with: 

$$
    \text{Attention}(Q, K, V) = \text{softmax}\left( Q K^T \right) V.
$$ 

Notice that we extended the vector query to a matrix
query $Q$ where each row represents one query vector.

**Transformers.** Even though attention solves the issue of restrictive
expressiveness, RNNs have another main architectural drawback: RNNs are
slow since they are sequential and therefore hard to parallelize. A new
model architecture based solely on attention mechanisms and fully
parallelised was proposed by (<a href="#ref-vaswani2017attention">Vaswani et al. (2017)</a>), called
*Transformers*, an encoder-decoder model. In this thesis, our models
only rely on the encoder part of the model, which is why we omit the
description of the decoder. We visualize the architecture of the encoder:.

<figure>
<img src="/assets/img/master-thesis/encoder_transformer.png" alt="interview-img" width="300">
<em>A transformer encoder block, adopted from (<a href="#ref-vaswani2017attention">Vaswani et al. (2017)</a>).</em>
</figure>

First, the sequence is tokenized, then the model embeds each token in
the input sequence with a token embedding layer, then adds a positional
encoding[^5] depending on the position of the token in the sequence.
These representations are then routed $N$ times through separate
(self-)attention and feedforward sub-networks. The core difference
between the attention module described above and self-attention is that
the query matrix $\mathbf{Q}$ is generated from tokens of the input sequence
and can attend to all other tokens of the same sequence, including
itself, to generate its new representation. Furthermore, they do not use
the dot-product attention but the *scaled dot-product attention*:

$$
    \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V.
$$ 

Additionally they utilize *multiple heads* (multiple
self-attention layers), which split the queries, keys, and values
matrices $Q, K, V$ along the embedding dimensions with
$d_k = d_v = d / h$ where $h$ is the number of heads. Subsequently, they
apply the self-attention independently, each having its own parameters.
The advantage of multi-heads is that tokens can jointly attend to
multiple tokens in the sequence. Each head produces its own output and
gets concatenated, once again projected, resulting in the final values

$$
    \text{MultiHead}(Q, V, K) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O 
$$ 

where $W^O \in \mathbb{R}^{hd_v \times d}$ is the
projection matrix. Finally, the result is fed into a feedforward neural
network. Additionally, the architecture uses dropout, residual
connections, and layer normalization to stabilize and improve training.

Using the transformer architecture has improved upon RNNs in many ways:
Through multi-heads, the total computational complexity per layer is
much lower, and through their ability to parallelize many computations,
the scalability of transformers far exceeds RNNs. Therefore, stacking
transformer blocks to increase the representative model capacity can be
done efficiently. Furthermore, the path length between long-range
dependencies in the network is reduced from $O(n)$ to $O(1)$ as
self-attention allows access to the input directly.

Another critical aspect of transformers is the pre-training and
fine-tuning paradigm: The general procedure is to pre-train on a
language modeling task on huge training text, which is possible because
of the high parallelizability of transformers, e.g.,
(<a href="#ref-Radford2018ImprovingLU">Radford et al. (2018)</a>) train on the next word prediction task on a
corpus with over $7000$ books. Given a downstream task, the whole
pre-trained model (coupled with a task head) is then fine-tuned on the
task dataset.

# Cross-Lingual Static Word Representations

In the previous Section 2, we outlined different architectures
to induce word representations. However, we restricted ourselves to
inducing monolingual word representations by pre-training on monolingual
text corpora (with the language modeling task, i.e., predicting the next
word). Since monolingual word embeddings pre-train in each language
independently, therefore only learning monolingual distributional
information, they can not capture semantic relations between words
across languages. The fundamental idea behind cross-lingual word
representations is to create an aligned representation space for word
representations from across multiple languages. This section briefly
discusses how to extend static word embeddings (induced by, e.g.,
word2vec and FastText) to create a cross-lingual space.

FastText and word2vec induce static word embeddings, i.e., they do not
consider the context of a word in their representation. Therefore
phrases and polysemy of words can not be correctly captured, which
prohibits the effectiveness of an aligned space across languages with
static embeddings. Nonetheless, we discuss two popular approaches: (1)
Projection-based models and (2) Bilingual Embeddings.

**Projection-based methods.** Projection-based methods rely on
independently trained monolingual word vector spaces in the source
language $\mathbf{X}\_{L1}$ and target language $\mathbf{X}\_{L2}$, that post-hoc
align the monolingual spaces into one cross-lingual word embedding
$\mathbf{X}\_{CL}$:

<figure>
<img src="/assets/img/master-thesis/projection_based.png" alt="interview-img">
<em>Illustration of projection-based methods. $X$ and $Y$ are monolingual
spaces and $W$ the projection matrix. Adopted from
(<a href="#ref-translation_conneau_2017">Conneau et al. (2017)</a>).</em>
</figure>

The alignment is based on word translation pairs $D$, which can be
obtained by existing dictionaries or by inducing it automatically. The
former is a supervised method and the latter an unsupervised approach
that usually assumes (approximately) isomorphism between monolingual
spaces. Typically, a supervised projection-based method uses a
pre-obtained dictionary $D$ containing word pairs for finding the
alignment (<a href="#ref-mikolov2013exploiting huang-etal-2015-translation">mikolov2013exploiting huang-etal-2015-translation</a>) .
Unsupervised methods induce the dictionary using different strategies
such as adversarial learning (<a href="#ref-translation_conneau_2017">Conneau et al. (2017)</a>) ,
similarity-based heuristics (<a href="#ref-artetxe-etal-2018-robust">Artetxe et al. (2018)</a>) , PCA
(<a href="#ref-hoshen-wolf-2018-non">Hoshen et al. (2018)</a>) , and optimal transport
(<a href="#ref-alvarez-melis-jaakkola-2018-gromov">Alvarez-Melis et al. (2018)</a>) .

Projection-based methods construct an aligned monolingual subspace
$\mathbf{X}\_S$ and $\mathbf{X}\_T$, where the aligned rows are translations of
each other. (<a href="#ref-mikolov2013exploiting">Mikolov et al. (2013)</a>) learns the projection $\mathbf{W}_{L1}$,
by minimizing the Euclidean distance between the linear projection of
$\mathbf{X}\_S$ onto $\mathbf{X}\_T$: 

$$
    \mathbf{W}\_{L1} = argmin  \vert \mathbf{X}\_{S} \mathbf{W} - \mathbf{X}\_{T} \vert
$$ 

which can be further improved by constraining
$\mathbf{W}\_{L1}$ to be an orthogonal matrix (<a href="#ref-xing-etal-2015-normalized">Xing et al. (2015)</a>).

The induced cross-lingual space performs well for related languages on
the BLI task but degrades when the language pair is distant
(<a href="#ref-vulvic_2019_bli">Vulić et al. (2019)</a>). Furthermore, (<a href="#ref-glavas2019properly">Glavas et al. (2019)</a>) show that the BLI
is not necessarily correlated to downstream performance.

**Bilingual Embeddings.** Bilingual Embeddings induce the cross-lingual
space by jointly learning representations from scratch. In general, the
general joint objective can be expressed as: 

$$
    \alpha(Mono_1 + Mono_2) + \beta Bi
$$ 

where $Mono_1$ and $Mono_2$ are monolingual models,
aiming to capture the clustering structure of each language, whereas the
bilingual component, $Bi$, encodes the information that ties the two
monolingual spaces together
(<a href="#ref-klementiev-etal-2012-inducing luong-etal-2015-bilingual">klementiev-etal-2012-inducing luong-etal-2015-bilingual</a>) . The
hyperparameters $\alpha$ and $\beta$ weight the influence of the
monolingual components and the bilingual component.

One popular choice is the BiSkip-gram model which extends the Skip-gram
model (see Section 2) by predicting words crosslingually
rather than just monolingually:

<figure>
<img src="/assets/img/master-thesis/bigram.PNG" alt="interview-img">
<em>The BiSkip-gram predicts within language and cross-lingually based on
the alignment information. Image taken from
(<a href="#ref-luong-etal-2015-bilingual">Luong et al. (2015)</a>).</em>
</figure>

However, the approach is expensive in terms of supervision as the
BiSkip-gram approach is based on a parallel corpus. Furthermore, for
low-resource languages, this level of supervision is, in some cases,
impossible to acquire.

# Cross-Lingual Contextualized Word Representations

Transformers and RNNs produce contextualized word embeddings, i.e., they
encode the same word differently depending on its context. We already
discussed in Section 2 why transformers improve upon RNNs
from an architectural standpoint and how it benefits the pre-training.
This section will first dive into one of the most popular transformer
models and its pre-training tasks. We then explore multilingual
transformers and, more importantly, the XLM-R model as it builds the
foundation of our thesis.

**BERT.** Before exploring multilingual transformer models, we introduce
the perhaps most popular transformer: *BERT* (**B**idirectional
**E**ncoder **R**epresentations from **T**ransformers)
(<a href="#ref-devlin2019bert">Devlin et al. (2019)</a>) . BERT is a sequence encoder-only model, which slightly
modifies the Transformer architecture (<a href="#ref-vaswani2017attention">Vaswani et al. (2017)</a>)  in the
following ways: As it only consists of an encoder, the model allows
information to flow bidirectionally, creating bidirectional
representations.

Specifically, the BERT encoder is composed of $L$ layers, each layer $l$
with $M$ self-attention heads, where a self-attention head $(m,l)$ has a
*key*, *query* and *value* encoders. Each one is calculated by a linear
layer: 

$$\begin{aligned}
    \mathbf{Q}^{m,l}(\mathbf{x}) & = \mathbf{W}_q^{m,l} (\mathbf{x}) + \mathbf{b}_q \\
    \mathbf{K}^{m,l}(\mathbf{x}) & = \mathbf{W}_k^{m,l} (\mathbf{x}) + \mathbf{b}_k \\
    \mathbf{V}^{m,l}(\mathbf{x}) & = \mathbf{W}_v^{m,l} (\mathbf{x}) + \mathbf{b}_v
\end{aligned}$$ 

where the input $\mathbf{x}$ is the output representation
embedding layer for the first encoder layer, and for the rest of the
layers, $\mathbf{x}$ is the output representation of the former encoder
layer. These are then fed into the scaled dot-product attention (which does not introduce any new
parameters) and concatenated by the projection, outputting $\mathbf{h}\_1^l$. The output is then
again fed into the MLP with the layer norm component of the transformer
block: 

$$\begin{aligned}
    \mathbf{h}\_2^{l} & = \text{Dropout}(\mathbf{W}\_{m_1}^{l} \cdot \mathbf{h}\_1^l + \mathbf{b}\_{m_1}^{l}) \\
    \mathbf{h}\_3^{l} & = \mathbf{g}^l\_{LN_1} \odot \frac{(\mathbf{h}\_2^l + \mathbf{x}) - \mu}{\sigma} + \mathbf{b}^l\_{m_1} \\
    \mathbf{h}\_4^{l} & = \text{GELU}(\mathbf{W}\_{m_2}^{l} \cdot \mathbf{h}\_3^l + \mathbf{b}\_{m_2}^{l}) \\
    \mathbf{h}\_5^{l} & = \text{Dropout}(\mathbf{W}\_{m_3}^{l} \cdot \mathbf{h}\_4^l + \mathbf{b}\_{m_3}^{l}) \\
    \text{out}^{l} & = g^l\_{LN_2} \odot \frac{(\mathbf{h}\_5^l + \mathbf{h}\_3^l) - \mu}{\sigma} + \mathbf{b}^l\_{LN_2}. 
\end{aligned}$$ 

The model parameters $\Theta$ are therefore constructed
from the weight matrices $\mathbf{W}^{l,(\cdot)}\_{(\cdot)}$, bias vectors
$\mathbf{b}\_{(\cdot)}^{l, (\cdot)}$ and vectors $\mathbf{g}^l\_{(\cdot)}$.

Furthermore, the pre-training task is not the next word prediction
anymore but consists of two novel pre-training tasks: (1) The masked
language modeling task (MLM) and the (2) next-sentence prediction (NSP)
task. MLM masks tokens of the input sequence, and the task is to predict
the original token based on the masked sequence:

<figure>
<img src="/assets/img/master-thesis/Masked-Language-Modelling-MLM-task-used-to-train-autoencoding-contextualised-word.png" alt="interview-img">
<em>Masked Language Modeling task in BERT. Masked tokens get replaced with
a special token [MASK]. Taken from (<a href="#ref-torregrossa_2021">Torregrossa et al. (2021)</a>).</em>
</figure>

The masking happens by selecting $15\%$ tokens, then $80\%$ are masked,
or $10\%$ replaced by a random token or $10\%$ left unchanged. On the
other hand, NSP asks the model to predict whether the input, which
consists of two concatenated sentences, if they are consecutive to one
another. Specifically, they construct sentence pairs by taking $50\%$
actual sentence pairs that are consecutive and $50\%$ artificial
sentence pairs that are not consecutive. Additionally, while tokenizing
the input sequence, BERT inserts special tokens, such as the `[CLS]`
token and `[SEP]`, where the former is always inserted at the start of a
sequence, and the latter separates two sentences allowing to process
sentence pairs. The pre-trained BERT model can then be fine-tuned
end-to-end by adding one additional output layer, which either takes the
token representations for token level tasks or the `[CLS]`
representation for classification tasks as an input. BERT achieves
state-of-the-art for a wide range of tasks, such as question answering
and language inference (<a href="#ref-devlin2019bert">Devlin et al. (2019)</a>)  and marks the start of modern
NLP with transformers. A noteworthy modification of the BERT model is
*RoBERTa* (Robustly Optimized BERT Pretraining Approach)
(<a href="#ref-liu2019roberta">Liu et al. (2019)</a>) : First, they remove the NSP task, use bigger batch
sizes & longer sequences, and use a dynamic masking strategy, i.e., the
masking pattern is generated every time a sequence is fed to the model.
RoBERTa outperforms BERT on GLUE, RACE, and SQuAD.

**mBERT.** The idea to extend BERT to multiple languages is to
concatenate multiple monolingual corpora to then jointly pre-train on
it. As this massive multilingual corpus from many languages has an
enormous vocabulary size and a large number of out-of-vocabulary tokens,
BERT/mBERT uses a *subword-based tokenizer*. The idea is to split rare
words into smaller meaningful subwords, e.g., \"papers\" is split into
\"paper\" and \"s\". The model then learns that the word \"papers\" is
formed using the word \"paper\" with a slightly different meaning but
the same root word. There are many different implementations of this
idea, such as WordPiece (<a href="#ref-wordpiece_2016">Wu et al. (2016)</a>) , BPE (<a href="#ref-bpe_2015">Sennrich et al. (2015)</a>)  and
SentencePiece (<a href="#ref-sentencepiece_2018">Kudo et al. (2018)</a>) . The original BERT/mBERT
implementation uses WordPiece. To encode text from multiple languages,
the subword tokenizer creates its vocabulary on the concatenated text.
The multilingual version of BERT, dubbed *mBERT*, pre-trains on $104$
languages and surprisingly learns strong cross-lingual representations
that generalize well to other languages via zero-shot transfer
(<a href="#ref-pires-etal-2019-multilingual wu_emerging_2019">pires-etal-2019-multilingual wu_emerging_2019</a>)  without any explicit
supervision.

This ability can be explained by three factors
(<a href="#ref-pires-etal-2019-multilingual">Pires et al. (2019)</a>) : (1) The subword tokenizer maps common
subwords across languages which act as anchor points for learning an
alignment, e.g., \"DNA\" [^6] has a similar meaning even in distantly
related languages. The anchor points are similar to the seed dictionary
in the projection-based approach (see Section 3). (2) This effect is then reinforced and
distributed to other non-overlapping tokens by jointly training across
multiple languages forcing co-occurring tokens also to be mapped to a
shared space. (3) mBERT learns cross-lingual representations deeper than
simple vocabulary memorization, generalizing across languages. However,
recent works (<a href="#ref-wu-dredze-2019-beto k2020crosslingual">wu-dredze-2019-beto k2020crosslingual</a>)  show that a
shared vocabulary is not required to create a strong cross-lingual
representation. (<a href="#ref-k2020crosslingual">K et al. (2020)</a>)  additionally demonstrate that word
order plays an important role.

**XLM.** (<a href="#ref-lample2019crosslingual">Lample et al. (2019)</a>) introduce a new unsupervised method for
learning cross-lingual representations, called XLM (cross-lingual
language models). XLM builds upon BERT and makes the following
adjustment: First, they include the Translation Language Modeling (TLM)
task into the pre-training. Each training sample consisting of pairs of
parallel sentences (source and target sentence) is randomly masked. To
predict a masked word, the model is then allowed to either attend to the
surrounding source words or the target translation, encouraging the
model to align the source and target representations:

<figure>
<img src="/assets/img/master-thesis/tlm_xlm.PNG" alt="interview-img">
<em>Cross-lingual language model pretraining. Image taken from
(<a href="#ref-lample2019crosslingual">Lample et al. (2019)</a>).</em>
</figure>

They then choose to drop the NSP task and only alternate training
between MLM and TLM. Furthermore, the model receives a language ID to
its input (similar to positional encoding), helping the model learn the
relationship between related tokens in different languages. XLM uses the
subword tokenizer BPE (<a href="#ref-bpe_2015">Sennrich et al. (2015)</a>)  which learns the splits on the
concatenation of sentences sampled randomly from the monolingual
corpora. Furthermore, XLM samples according to a multinomial
distribution with probabilities: 

$$\begin{aligned}
    q_i = \frac{p_i^\alpha}{\sum_{j=1}^Np_i^\alpha} \quad \text{  with  } \quad p_i = \frac{n_i}{\sum_{k=1}^N n_k}
\end{aligned}$$ 

where $i$ denotes the index of the language and $n_i$
the the number of sentences in the text corpora of the language with the
index $i$. XLM uses $\alpha = 0.5$.

XLM outperforms mBERT on XNLI (<a href="#ref-Conneau2018XNLIEC">Conneau et al. (2018)</a>)  in 15 languages.
However, XLM handles fewer languages than mBERT, is based on a larger
model, and uses a high amount of supervision as it needs parallel
sentences during pre-training. Therefore the difference may not be so
significant in reality. Furthermore, acquiring parallel sentences for
low-resource languages is problematic, making the model unsuitable for
such scenarios.

**XLM-R.** (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) propose to take a step back and drop
the TLM task and only pre-train in RoBERTa fashion with the MLM task on
a huge, multilingual dataset. They dub their multilingual model
XLM-RoBERTa (XLM-R). They crawled a massive amount of text, over 2.5TB
of data in 100 languages. Additionally, they changed the vocabulary size
of RoBERTa to $250 000$ tokens compared to RoBERTa's $50 000$ tokens.
They employ the subword tokenizer SentencePiece (<a href="#ref-sentencepiece_2018">Kudo et al. (2018)</a>) 
with an unigram language model (<a href="#ref-subword_unigram_2018">Kudo (2018)</a>) . They use the
same sampling strategy as XLM, but utilize $\alpha = 0.3$. Furthermore,
XLM-R does not use language IDs, which will allow XLM-R to better deal
with code-switching. (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) provide two models:
XLM-R~Base~ ($L = 12, H = 768, A = 12, 270$M params) and XLM-R
($L = 24, H = 1024, A = 16, 550$M params).

(<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) show that XLM-R sets a new State-of-the-Art on
numerous cross-lingual tasks. Compared to mBERT and XLM, XLM-R provides
substantial gains in classification, sequence labeling, and question
answering *without any explicit cross-lingual supervision*.

# Challenges in Multilingual Transformers

As XLM-R and the concept of multilingual transformers build the basis of
our thesis, we will further analyze weaknesses and how our approach
tries to alleviate them.

**Low-Resource Languages.** Even though multilingual LMs do not use any
explicit cross-lingual signals, they still create multilingual
representations (<a href="#ref-pires-etal-2019-multilingual wu-dredze-2019-beto">pires-etal-2019-multilingual wu-dredze-2019-beto</a>) ,
which can be used for cross-lingual downstream tasks. By releasing a new
multi-task benchmark for evaluating the cross-lingual generalization,
called XTREME[^7], which covers nine tasks and 40 languages,
(<a href="#ref-hu2020xtreme">Hu et al. (2020)</a>) showed that even though models achieve human performance
on many tasks in *English*, there is a sizable gap in the performance of
cross-lingually transferred models, especially in *low-resource
languages*. Additionally, (<a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>)
(<a href="#ref-lauscher-etal-2020-zero">Lauscher et al. (2020)</a>)  showed that multilingual transformers
pre-trained via language modeling significantly underperform in
resource-lean scenarios and for distant languages. Furthermore, the
literature
(<a href="#ref-pires-etal-2019-multilingual wu-dredze-2019-beto k2020crosslingual">pires-etal-2019-multilingual wu-dredze-2019-beto k2020crosslingual</a>) 
focused on evaluating languages that were from the same language family
or with large corpora in pre-training, languages such as German, Spanish
or French. For example, (<a href="#ref-k2020crosslingual">K et al. (2020)</a>) investigate Hindi, Spanish,
and Russian, which are from the same language family, Indo-European, and
have a large corpus from Wikipedia. This concern is raised by multiple
sources (<a href="#ref-lauscher-etal-2020-zero wu-dredze-2019-beto">lauscher-etal-2020-zero wu-dredze-2019-beto</a>) , which show
that the performance drops huge for distant target languages and target
languages that have small pre-training corpora. Furthermore,
(<a href="#ref-lauscher-etal-2020-zero">Lauscher et al. (2020)</a>) show empirically that for massively
multilingual transformers, pre-training corpora sizes affect the
zero-shot performance in higher-level tasks. In contrast, the results in
lower-level tasks are more impacted by typological language proximity.

As multilingual transformers struggle with low-resource languages, we
investigate if our approach is capable of improving in these scenarios.
To strengthen the performance of low-resource languages, we try to avoid
the curse of multilinguality.

**Curse of Multilinguality.** (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) experiment with
different settings for the XLM-R model, showing that scaling the batch
size, training data, training time and shared vocabulary improve
performance on downstream task.

<figure>
<img src="/assets/img/master-thesis/curse_multi.PNG" alt="interview-img">
<img src="/assets/img/master-thesis/curse_multi_2.PNG" alt="interview-img">
</figure>

More importantly, they show that for a fixed model capacity, adding more
languages to the pre-training lead to better cross-lingual performance
on low-resource languages till the per-language capacity is too low
(*capacity dilution*), after which the overall performance on
monolingual and cross-lingual benchmarks degrades. They call this the
*curse of multilinguality*. 

<figure>
<img src="/assets/img/master-thesis/curse_multi.PNG" alt="interview-img">
<img src="/assets/img/master-thesis/curse_multi_2.PNG" alt="interview-img">
</figure>

Adding more languages to the pre-training
ultimately has two important effects: (1) Positive cross-lingual
transfer, especially for low-resource languages, and (2) lower
per-language capacity, which then, in turn, can degrade the overall
model performance. Multilingual transformer models have to carefully
balance these two effects of *capacity dilution* and positive transfer.
Adding more capacity to the model can alleviate some of the curse but is
not a solution for moderate-size models.

Furthermore, the allocation of the model capacity across languages is
influenced by the training set size, the size of the shared subword
vocabulary, and the rate at which the model samples training instances
from each language during pre-training. (<a href="#ref-conneau2020unsupervised">Conneau et al. (2020)</a>) show
that sampling batches of low-resource languages improve performance on
low-resource languages and vice-versa. XLM-R uses a rate of
$\alpha = 0.3$, which still leaves some room for improvement on
low-resource languages.

In my Master Thesis, we introduces language-specific students that allocate $100\%$
of model parameters to one language, avoiding the capacity dilution
while still benefiting from the acquired cross-lingual knowledge from
XLM-R by distilling from the XLM-R model.


# References

[^1]: A tokenizer splits the text into smaller units called tokens.
    Tokens can be words, characters, or subwords. In our thesis, we
    mostly use the term word representations to illustrate concepts
    better. Only when necessary do we explicitly state tokens.
    Nevertheless, all presented approaches can be generalized to any
    token-level.

[^2]: Relationships are defined by subtracting two words vectors, and
    the result is added to another word.

[^3]: Figure taken from
    <https://www.baeldung.com/cs/nlp-encoder-decoder-models>

[^4]: Figure taken from
    <https://project-archive.inf.ed.ac.uk/ug4/20201880/ug4_proj.pdf>

[^5]: Notice that without the positional encoding, the transformer has
    no notion of word order.

[^6]: \"DNA\" is indeed a subword in mBERT (<a href="#ref-wu-dredze-2019-beto">Wu et al. (2019)</a>) .

[^7]: <https://sites.research.google/xtreme>



<ol class="bibliography">
  <li id="ref-artexe_mt qi-etal-2018-pre lample-etal-2018-phrase">artexe_mt qi-etal-2018-pre lample-etal-2018-phrase</li>
  <li id="ref-10.1145/2766462.2767752">Ivan Vuli\'{c}, Marie-Francine Moens. <em>Monolingual and Cross-Lingual Information Retrieval Models Based on (Bilingual) Word Embeddings</em>. Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2015. <a href="https://doi.org/10.1145/2766462.2767752">link</a></li>
  <li id="ref-10.1162/089120102762671990">Warren J. Plath. <em>{Early Years in Machine Translation: Memoirs and Biographies of Pioneers}</em>. Computational Linguistics. 2002. <a href="https://doi.org/10.1162/089120102762671990">link</a></li>
  <li id="ref-Goddard2006NaturalSM">Cliff Goddard. <em>Natural Semantic Metalanguage</em>. 2006</li>
  <li id="ref-lample2019crosslingual  lauscher-etal-2020-zero">lample2019crosslingual  lauscher-etal-2020-zero</li>
  <li id="ref-doi:10.1080/00437956.1954.11659520">Zellig S. Harris. <em>Distributional Structure</em>. <i>WORD</i>. 1954. <a href="https://doi.org/10.1080/00437956.1954.11659520">link</a></li>
  <li id="ref-collobert_2008">Ronan Collobert, Jason Weston. <em>A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning</em>. Proceedings of the 25th International Conference on Machine Learning. 2008. <a href="https://doi.org/10.1145/1390156.1390177">link</a></li>
  <li id="ref-Jelinek1977PerplexityaMO">Frederick Jelinek, Robert L. Mercer, Lalit R. Bahl, J. Baker. <em>Perplexity—a measure of the difficulty of speech recognition tasks</em>. Journal of the Acoustical Society of America. 1977</li>
  <li id="ref-mikolov2013exploiting">Tomas Mikolov, Quoc V. Le, Ilya Sutskever. <em>Exploiting Similarities among Languages for Machine Translation</em>. arXiv. 2013. <a href="https://arxiv.org/abs/1309.4168">link</a></li>
  <li id="ref-fasttext_2016">Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov. <em>Enriching Word Vectors with Subword Information</em>. arXiv. 2016. <a href="https://arxiv.org/abs/1607.04606">link</a></li>
  <li id="ref-lample_2016  kiros_2015 Kusner2015FromWE">lample_2016  kiros_2015 Kusner2015FromWE</li>
  <li id="ref-vanishing_1998 Bengio1994LearningLD">vanishing_1998 Bengio1994LearningLD</li>
  <li id="ref-pascanu_2021">Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. <em>On the difficulty of training Recurrent Neural Networks</em>. arXiv. 2012. <a href="https://arxiv.org/abs/1211.5063">link</a></li>
  <li id="ref-Le2015ASW">Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton. <em>A Simple Way to Initialize Recurrent Networks of Rectified Linear Units</em>. ArXiv. 2015</li>
  <li id="ref-relu_2018">Abien Fred Agarap. <em>Deep Learning using Rectified Linear Units (ReLU)</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1803.08375">link</a></li>
  <li id="ref-lstm_1997 gers_2000">lstm_1997 gers_2000</li>
  <li id="ref-sutskever_2011 mikolov_rnn_2011 Sutskever2014SequenceTS">sutskever_2011 mikolov_rnn_2011 Sutskever2014SequenceTS</li>
  <li id="ref-Hinton1993AutoencodersMD">Geoffrey E. Hinton, Richard S. Zemel. <em>Autoencoders, Minimum Description Length and Helmholtz Free Energy</em>. NIPS. 1993</li>
  <li id="ref-Kalchbrenner2013RecurrentCT Sutskever2014SequenceTS">Kalchbrenner2013RecurrentCT Sutskever2014SequenceTS</li>
  <li id="ref-attention_2014">Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. <em>Neural Machine Translation by Jointly Learning to Align and Translate</em>. arXiv. 2014. <a href="https://arxiv.org/abs/1409.0473">link</a></li>
  <li id="ref-vaswani2017attention">Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. <em>Attention Is All You Need</em>. arXiv. 2017. <a href="https://arxiv.org/abs/1706.03762">link</a></li>
  <li id="ref-Radford2018ImprovingLU">Alec Radford, Karthik Narasimhan. <em>Improving Language Understanding by Generative Pre-Training</em>. 2018</li>
  <li id="ref-translation_conneau_2017">Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. <em>Word Translation Without Parallel Data</em>. arXiv. 2017. <a href="https://arxiv.org/abs/1710.04087">link</a></li>
  <li id="ref-mikolov2013exploiting huang-etal-2015-translation">mikolov2013exploiting huang-etal-2015-translation</li>
  <li id="ref-artetxe-etal-2018-robust">Mikel Artetxe, Gorka Labaka, Eneko Agirre. <em>A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings</em>. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018. <a href="https://aclanthology.org/P18-1073">link</a></li>
  <li id="ref-hoshen-wolf-2018-non">Yedid Hoshen, Lior Wolf. <em>Non-Adversarial Unsupervised Word Translation</em>. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. <a href="https://aclanthology.org/D18-1043">link</a></li>
  <li id="ref-alvarez-melis-jaakkola-2018-gromov">David Alvarez-Melis, Tommi Jaakkola. <em>{G}romov-{W}asserstein Alignment of Word Embedding Spaces</em>. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. <a href="https://aclanthology.org/D18-1214">link</a></li>
  <li id="ref-xing-etal-2015-normalized">Chao Xing, Dong Wang, Chao Liu, Yiye Lin. <em>Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation</em>. Proceedings of the 2015 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies. 2015. <a href="https://www.aclweb.org/anthology/N15-1104">link</a></li>
  <li id="ref-vulvic_2019_bli">Ivan Vulić, Goran Glavaš, Roi Reichart, Anna Korhonen. <em>Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1909.01638">link</a></li>
  <li id="ref-glavas2019properly">Goran Glavas, Robert Litschko, Sebastian Ruder, Ivan Vulic. <em>How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1902.00508">link</a></li>
  <li id="ref-klementiev-etal-2012-inducing luong-etal-2015-bilingual">klementiev-etal-2012-inducing luong-etal-2015-bilingual</li>
  <li id="ref-luong-etal-2015-bilingual">Thang Luong, Hieu Pham, Christopher D. Manning. <em>Bilingual Word Representations with Monolingual Quality in Mind</em>. Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing. 2015. <a href="https://aclanthology.org/W15-1521">link</a></li>
  <li id="ref-devlin2019bert">Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. <em>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1810.04805">link</a></li>
  <li id="ref-torregrossa_2021">François Torregrossa, Allesiardo Robin, Vincent Claveau, Nihel Kooli, Guillaume Gravier. <em>A survey on training and evaluation of word embeddings</em>. International Journal of Data Science and Analytics. 2021. <a href="https://doi.org/10.1007/s41060-021-00242-8">link</a></li>
  <li id="ref-liu2019roberta">Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. <em>RoBERTa: A Robustly Optimized BERT Pretraining Approach</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1907.11692">link</a></li>
  <li id="ref-wordpiece_2016">Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean. <em>Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation</em>. arXiv. 2016. <a href="https://arxiv.org/abs/1609.08144">link</a></li>
  <li id="ref-bpe_2015">Rico Sennrich, Barry Haddow, Alexandra Birch. <em>Neural Machine Translation of Rare Words with Subword Units</em>. arXiv. 2015. <a href="https://arxiv.org/abs/1508.07909">link</a></li>
  <li id="ref-sentencepiece_2018">Taku Kudo, John Richardson. <em>SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1808.06226">link</a></li>
  <li id="ref-pires-etal-2019-multilingual wu_emerging_2019">pires-etal-2019-multilingual wu_emerging_2019</li>
  <li id="ref-pires-etal-2019-multilingual">Telmo Pires, Eva Schlinger, Dan Garrette. <em>How Multilingual is Multilingual {BERT}?</em>. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. <a href="https://aclanthology.org/P19-1493">link</a></li>
  <li id="ref-wu-dredze-2019-beto k2020crosslingual">wu-dredze-2019-beto k2020crosslingual</li>
  <li id="ref-k2020crosslingual">Karthikeyan K, Zihan Wang, Stephen Mayhew, Dan Roth. <em>Cross-Lingual Ability of Multilingual BERT: An Empirical Study</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1912.07840">link</a></li>
  <li id="ref-lample2019crosslingual">Guillaume Lample, Alexis Conneau. <em>Cross-lingual Language Model Pretraining</em>. arXiv. 2019. <a href="https://arxiv.org/abs/1901.07291">link</a></li>
  <li id="ref-Conneau2018XNLIEC">Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk, Veselin Stoyanov. <em>XNLI: Evaluating Cross-lingual Sentence Representations</em>. EMNLP. 2018</li>
  <li id="ref-conneau2020unsupervised">Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. <em>Unsupervised Cross-lingual Representation Learning at Scale</em>. arXiv. 2020. <a href="https://arxiv.org/abs/1911.02116">link</a></li>
  <li id="ref-subword_unigram_2018">Taku Kudo. <em>Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates</em>. arXiv. 2018. <a href="https://arxiv.org/abs/1804.10959">link</a></li>
  <li id="ref-pires-etal-2019-multilingual wu-dredze-2019-beto">pires-etal-2019-multilingual wu-dredze-2019-beto</li>
  <li id="ref-hu2020xtreme">Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, Melvin Johnson. <em>XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization</em>. arXiv. 2020. <a href="https://arxiv.org/abs/2003.11080">link</a></li>
  <li id="ref-wu-dredze-2019-beto">Shijie Wu, Mark Dredze. <em>Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of {BERT}</em>. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019. <a href="https://aclanthology.org/D19-1077">link</a></li>
  <li id="ref-lauscher-etal-2020-zero">Anne Lauscher, Vinit Ravishankar, Ivan Vuli{\'c}, Goran Glava{\v{s}}. <em>From Zero to Hero: {O}n the Limitations of Zero-Shot Language Transfer with Multilingual {T}ransformers</em>. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. <a href="https://aclanthology.org/2020.emnlp-main.363">link</a></li>
  <li id="ref-pires-etal-2019-multilingual wu-dredze-2019-beto k2020crosslingual">pires-etal-2019-multilingual wu-dredze-2019-beto k2020crosslingual</li>
  <li id="ref-lauscher-etal-2020-zero wu-dredze-2019-beto">lauscher-etal-2020-zero wu-dredze-2019-beto</li>
</ol>


