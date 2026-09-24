---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
<style>
/* Tighten spacing */
li > blockquote {
  margin-top: 0.1em !important;
  margin-bottom: 0.1em !important;
  margin-left: 1em !important; /* 👈 add this line to indent */
  border-left: 3px solid #ccc; /* optional: thinner, subtler quote bar */
  padding-left: 0.8em;         /* keeps text away from the bar */
}

/* Optional: make nested list inside quote compact */
li > blockquote > ul {
  margin-top: 0.1em !important;
  margin-bottom: 0.1em !important;
  padding-left: 1em !important;
}

li > blockquote p {
  margin-top: 0 !important;
  margin-bottom: 0.1em !important;
}
</style>

Hi, I'm Duc, a PhD student at JGU Mainz (Germany), supervised by **Prof. Katharina von der Wense**. I work on **Human-Centered NLP**, with a focus on **cultural and linguistic diversity**: designing and evaluating language technologies that **prioritize human needs** and consider the ethical and social implications of these systems. I was fortunate to do research visits with **Prof. Anne Lauscher (University of Hamburg)** and **Prof. Diyi Yang (Stanford)**.


Broadly, my research groups into:

- **Cultural Variation:** How models fail people across cultural contexts, conventions, and norms
  > - [*Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision-Language Models*](https://aclanthology.org/2025.naacl-long.490/) @ *NAACL 2025*
  > - [*Perspectives on Cross-Lingual Consistency in LLMs for Medical Questions*](https://arxiv.org/pdf/2609.07687) @ *EMNLP 2026*
  > - [*On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures*](https://aclanthology.org/2025.acl-long.1032/) @ *ACL 2025*
  > - [*Findings of the AmericasNLP 2026 Shared Task on Cultural Image Captioning for Indigenous Languages*](https://aclanthology.org/2026.americasnlp-6.27.pdf) @ *AmericasNLP 2026*

- **Linguistic Variation:**  How models disadvantage speakers of different dialects and language varieties
  > - [*Large Language Models Discriminate Against Speakers of German Dialects*](https://arxiv.org/abs/2509.13835) @ *EMNLP 2025*
  > - [*Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect*](https://www.arxiv.org/abs/2602.16852) @ *LREC 2026*

- **Deployment & Real-World Harm:** How social harms surface when these systems reach real users
  > - [*Greater accessibility can amplify discrimination in generative AI*](https://arxiv.org/abs/2603.22260) *(Preprint)*
  > - [*From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation*](https://arxiv.org/abs/2604.21716) @ *ACL 2026 Findings*


Latest News ([See all](/news/))
------
{% assign news_items = site.data.news %}
<table style="border-collapse: collapse; border:none; font-size:18px;">
  {% for item in news_items limit:5 %}
    <tr>
      <td style="width:20%; border: none; vertical-align:top;">
        <b>{{ item.date }}</b>
      </td>
      <td style="width:80%; border: none; vertical-align:top;">
        {{ item.news }}
      </td>
    </tr>
  {% endfor %}
</table>

---


Recent Publications ([See all](/publications/))
------
{% assign reversed_publications = site.data.publications %}
{% for post in reversed_publications limit:7 %}
{% include paper.html %}
{% endfor %}
