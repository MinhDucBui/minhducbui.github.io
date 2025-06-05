---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% include base_path %}

<!-- New style rendering if publication categories are defined -->
{% assign publication_groups = site.data.publications | sort: "type" | group_by: "type" %}
{% for publications in publication_groups %}
  <h2>{{publications.name | capitalize}}</h2>
  ---
  {% assign prefix = publications.name | slice: 0 | capitalize %}
  {% assign sorted_items = publications.items | sort: "date", "last" %}
  {% for post in sorted_items reversed %}
    {% assign prefix_index = forloop.length | minus: forloop.index0 %}
    {% include paper.html %}
  {% endfor %}
{% endfor %}


