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


Hi, my name is **Minh Duc Bui**, but you can call me **Duc**. I am a **PhD student** in Natural Language Processing (NLP) at Johannes Gutenberg University Mainz (Germany), supervised by **Katharina von der Wense (née Kann)**. My research focuses on **socially aware NLP**, investigating how LLM-based systems represent and respond to human diversity across linguistic, cultural, and socio-demographic dimensions.

- **Cultural Variation:** How cultural context, conventions, and norms shape model behaviour  
  > - [*Multi3Hate: Cross-Cultural Hate Speech Detection*](https://aclanthology.org/2025.naacl-long.490/) @ *NAACL 2025*  
  > - [*On Generalization across Measurement Systems*](https://aclanthology.org/2025.acl-long.1032/) @ *ACL 2025*  
  > - *Upcoming:* Korean Honorific Translations
- **Linguistic Variation:** How language form and variation (e.g., dialects, syntax, orthography) affect model performance  
  > - [*Large Language Models Discriminate Against Speakers of German Dialects*](https://arxiv.org/abs/2509.13835) @ *EMNLP 2025*  
  > - *Upcoming:* Meenzerisch (the dialect of Mainz, Germany)
- **Socio-Demographic Variation:** How social attributes such as gender, identity, and ethnicity are encoded or amplified by LLMs  
  > - *Upcoming:* AudioLLMs

Together, these studies advance insight into how LLMs capture diversity, guiding the development of AI systems that promote equity and inclusion across varied populations.


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
