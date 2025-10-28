---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, my name is **Minh Duc Bui**, but you can call me **Duc**. I am a PhD student in **Natural Language Processing (NLP)** at **Johannes Gutenberg University Mainz (Germany)**, supervised by **Katharina von der Wense (née Kann)**.  

My research focuses on the **cross-cultural aspects of LLMs**, examining how they represent and respond to human diversity across three key dimensions:  

- **Linguistic Variation:** how language form and variation (e.g., dialects, syntax, orthography) affect model performance
   - [Large Language Models Discriminate Against Speakers of German Dialects](https://arxiv.org/abs/2509.13835) at EMNLP 2025  
- **Cultural Variation:** how cultural context, conventions, and norms shape model interpretation and reasoning
   - [Multi3Hate](https://aclanthology.org/2025.naacl-long.490/) at NAACL 2025
   - [On Generalization across Measurement Systems](https://aclanthology.org/2025.acl-long.1032/) at ACL 2025
- **Socio-demographic Variation:** how social attributes such as gender, class, and ethnicity are encoded or amplified by LLMs
   - Upcoming work on AudioLLMs

Together, these studies aim to understand how LLMs internalize linguistic, cultural, and social variation and how we can design more equitable systems for users across diverse backgrounds.




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
