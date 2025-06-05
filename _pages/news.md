---
layout: archive
title: "News"
permalink: /news/
author_profile: true
---

{% assign news_items = site.data.news %}
<table style="border-collapse: collapse; border:none; font-size:18px;">
  {% for item in news_items %}
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
