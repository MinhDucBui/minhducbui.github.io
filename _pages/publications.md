---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if site.author.googlescholar %}
  <div class="wordwrap">You can also find my articles on <a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</div>
{% endif %}

{% include base_path %}

---

<div class="publications-list">
  {% for post in site.publications reversed %}
  <article class="archive-item">
    <header>
      <h2 style="font-size: 14px; margin-bottom: 2px;">  <!-- Reduced margin-bottom even further -->
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <p class="publication-meta" style="font-family: Arial, sans-serif; font-size: 12px; color: #555; margin-top: 0;">
        {% if post.authors %}
          {% assign bold_name = "Minh Duc Bui" %}
          {% assign formatted_authors = "" %}
          {% assign author_count = post.authors.size %}

          {% for author in post.authors %}
            {% if author == bold_name %}
              {% assign formatted_authors = formatted_authors | append: "<strong>" | append: author | append: "</strong>" %}
            {% else %}
              {% assign formatted_authors = formatted_authors | append: author %}
            {% endif %}
            {% if forloop.index != author_count %}
              {% assign formatted_authors = formatted_authors | append: ", " %}
            {% endif %}
          {% endfor %}

          {{ formatted_authors | strip_newlines }}
        {% endif %}

        {% if post.conference %} • <strong>{{ post.conference }}</strong>{% endif %}
        {% if post.date %} • {{ post.date | date: "%B %Y" }}{% endif %}
      </p>
    </header>
    <div class="publication-excerpt">
      {% if post.excerpt %}
      {{ post.excerpt }}
      {% endif %}
    </div>
  </article>
  {% endfor %}
</div>
