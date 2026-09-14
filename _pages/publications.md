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
  <h2>{{ publications.name }}</h2>
  <hr>
  {% assign prefix = publications.name | split: " " | last | slice: 0 | upcase %}
  {% assign sorted_items = publications.items | sort: "date", "last" %}
  {% for post in sorted_items reversed %}
    {% assign prefix_index = forloop.length | minus: forloop.index0 %}
    <div class="paper-entry" style="margin-bottom: 1.5em;">
      <h3 style="margin-bottom: 0.2em;">
        [{{ prefix }}{{ prefix_index }}] {{ post.title }}{% if post.paperurl %}<a href="{{ post.paperurl }}" target="_blank" rel="noopener noreferrer" aria-label="Open paper: {{ post.title | escape }}" style="margin-left: 6px; white-space: nowrap;"><svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false" style="display: inline-block; vertical-align: -0.1em;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>{% endif %}
      </h3>
      <p style="margin: 0.2em 0; color: #555;">{{ post.authors }}</p>
      <p style="margin: 0.2em 0; font-style: italic;">{{ post.venue }} &middot; {{ post.date | date: "%B %Y" }}</p>
    </div>
  {% endfor %}
{% endfor %}