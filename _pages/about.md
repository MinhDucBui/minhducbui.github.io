---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

My name is **Minh Duc Bui** (you can call me **Duc**). I am a **PhD student in Natural Language Processing (NLP)** at the **Johannes Gutenberg University Mainz (Germany)**, supervised by **Katharina von der Wense (née Kann)**. My research examines the **cross-cultural aspects of large language models (LLMs)** — specifically, how they represent and respond to human diversity across three key dimensions:

---

#### **Linguistic Variation**
Investigating how language form and variation (e.g., dialects, syntax, orthography) influence model performance.

- [*Large Language Models Discriminate Against Speakers of German Dialects*](https://arxiv.org/abs/2509.13835) — EMNLP 2025  
- *Upcoming:* Work on **Meenzerisch** (the dialect of Mainz, Germany)

---

#### **Cultural Variation**
Exploring how cultural context, conventions, and norms shape model behaviour and interpretation.

- [*Multi3Hate*](https://aclanthology.org/2025.naacl-long.490/) — NAACL 2025  
- [*On Generalization across Measurement Systems*](https://aclanthology.org/2025.acl-long.1032/) — ACL 2025  
- *Upcoming:* Work on **Korean Honorific Translations**

---

#### **Socio-demographic Variation**
Examining how social attributes such as gender, identity, and ethnicity are encoded or amplified by LLMs.

- *Upcoming:* Work on **AudioLLMs**

---

Together, these projects aim to advance our understanding of how LLMs internalize **linguistic, cultural, and social variation**, and how we can develop **more equitable and inclusive language technologies** for diverse user populations.





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
