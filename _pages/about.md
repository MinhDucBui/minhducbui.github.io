---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, my name is **Minh Duc Bui**, but you can call me **Duc**. I am a **PhD student** in Natural Language Processing (NLP) at Johannes Gutenberg University Mainz (Germany), supervised by **Katharina von der Wense (née Kann)**.  

My research focuses on the **cross-cultural aspects of large language models (LLMs)**, examining how they represent and respond to human diversity across three key dimensions:  
- **Linguistic variation** — how language form and variation (e.g., dialects, syntax, orthography) affect model performance  
  <blockquote>
  <ul>
    <li><a href="https://arxiv.org/abs/2509.13835"><em>Large Language Models Discriminate Against Speakers of German Dialects</em></a> — <em>EMNLP 2025</em></li>
    <li><em>Upcoming:</em> Meenzerisch (the dialect of Mainz, Germany)</li>
  </ul>
  </blockquote>

- **Cultural variation** — how cultural context, conventions, and norms shape model behaviour  
  <blockquote>
  <ul>
    <li><a href="https://aclanthology.org/2025.naacl-long.490/"><em>Multi3Hate</em></a> — <em>NAACL 2025</em></li>
    <li><a href="https://aclanthology.org/2025.acl-long.1032/"><em>On Generalization across Measurement Systems</em></a> — <em>ACL 2025</em></li>
    <li><em>Upcoming:</em> Korean Honorific Translations</li>
  </ul>
  </blockquote>

- **Socio-demographic variation** — how social attributes such as gender, identity, and ethnicity are encoded or amplified by LLMs  
  <blockquote>
  <ul>
    <li><em>Upcoming:</em> AudioLLMs</li>
  </ul>
  </blockquote>


Together, these studies aim to advance our understanding of how LLMs internalize linguistic, cultural, and social variation, and how we can design **more equitable and inclusive AI systems** for users across diverse backgrounds.


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
