---
name: recommendation-skill
description: Find current, verifiable local dining, travel, activity, or service recommendations. Use when the user asks what to eat, where to go, what is nearby, or requests a current recommendation.
---

# Recommendation Skill

1. Extract location, time, budget, distance, dietary preference, and activity constraints from the request.
2. Call `web.search` for current public sources; never substitute unrelated local-library chunks.
3. Prefer concrete places with a source URL and an address, district, or landmark.
4. Separate verifiable facts from subjective recommendation reasons.
5. Do not invent ratings, prices, opening hours, availability, or popularity.
6. State when current status could not be verified and suggest the user confirm before leaving.
7. Cite every externally verifiable claim used in the recommendation.