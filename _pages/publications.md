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
      <h2>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <p class="publication-meta">
        {% if post.authors %}
          {% assign bold_name = "Minh Duc Bui" %}
          {% assign formatted_authors = "" %}

          {% for author in post.authors %}
            {% if author == bold_name %}
              {% assign formatted_authors = formatted_authors | append: "<strong>" | append: author | append: "</strong>, " %}
            {% else %}
              {% assign formatted_authors = formatted_authors | append: author | append: ", " %}
            {% endif %}
          {% endfor %}

          {{ formatted_authors | strip_newlines | remove: ", " }}
        {% endif %}

        {% if post.conference %} • Presented at {{ post.conference }}{% endif %}
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