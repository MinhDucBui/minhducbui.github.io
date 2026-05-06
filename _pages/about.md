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


Hi, I'm Duc, a PhD student at JGU Mainz (Germany), supervised by Katharina von der Wense. I work on **AI safety and NLP**: rigorously measuring and mitigating how LLM-based systems encode and amplify **social disparities** rooted in **human diversity**, spanning **cultural and linguistic variation**, across text, vision, and audio. I also study how harms can arise from integrating AI models into real-world pipelines, and how to mitigate them.


- **Cultural Variation:** Measuring and mitigating gaps in model behavior arising from cultural contexts, conventions, and norms
  > - [*Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision-Language Models*](https://aclanthology.org/2025.naacl-long.490/) @ *NAACL 2025*
  > - [*On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures*](https://aclanthology.org/2025.acl-long.1032/) @ *ACL 2025*

- **Linguistic Variation:**  Measuring and mitigating how social meaning in language variation drives discriminatory model behavior
  > - [*Large Language Models Discriminate Against Speakers of German Dialects*](https://arxiv.org/abs/2509.13835) @ *EMNLP 2025*
  > - [*Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect*](https://www.arxiv.org/abs/2602.16852) @ *LREC 2026*

- **Harm Propagation at Deployment Scale:** Measuring and mitigating how integrating models into real-world pipelines creates harm
  > - [*Greater accessibility can amplify discrimination in generative AI*](https://arxiv.org/abs/2603.22260) *(Preprint)*
  > - [*From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation*](https://arxiv.org/abs/2604.21716) @ *ACL 2026 Findings*

Together, these studies reflect my broader research agenda: building the empirical case that socially-encoded disparities in LLM behavior are a safety risk, one that scales with deployment, and developing targeted interventions that reduce them.


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
