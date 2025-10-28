---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, my name is **Minh Duc Bui**, but you can call me **Duc**. I am a PhD student in **Natural Language Processing (NLP)** at **Johannes Gutenberg University Mainz (Germany)**, supervised by **Katharina von der Wense (née Kann)**. My current research focuses on **cross-cultural aspects of NLP** and the **broader implications of human interaction with LLMs**. I am particularly interested in how **linguistic** ([EMNLP 2025](https://arxiv.org/abs/2509.13835)), **cultural** ([NAACL 2025](https://aclanthology.org/2025.naacl-long.490/)), and **social** ([ACL 2025](https://aclanthology.org/2025.acl-long.1032/)) differences influence model behavior and how these differences can inadvertently disadvantage certain communities.



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
