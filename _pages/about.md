---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, My name is **Minh Duc Bui**, but you can call me Duc. I am a PhD student specializing in Natural Language Processing (NLP) at JGU Mainz (Germany), under the supervision of **Katharina von der Wense** (nèe Kann). My research centers on **cross-cultural NLP**, exploring how NLP techniques can be adapted to understand and respect cultural differences. I aim to develop models that are sensitive to cultural nuances, ultimately fostering better global communication and inclusivity in AI applications.


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

### **September 2023**  
🎓 Began my PhD at the [NALA Group](https://nala-cub.github.io), hosted by **JGU** and the **University of Colorado Boulder**, under the supervision of Prof. Dr. [Katharina von der Wense](https://scholar.google.de/citations?user=3XF5bqEAAAAJ&hl=en) (née Kann)! 

Recent Publications ([See all](/publications/))
------
{% assign reversed_publications = site.data.publications %}
{% for post in reversed_publications limit:7 %}
{% include paper.html %}
{% endfor %}
