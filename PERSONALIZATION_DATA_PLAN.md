# Personalization Data Plan

## Short answer
To personalize search, you need more than identity (`name`, `email`).  
You need **behavioral signals** tied to a stable user ID.

## What to capture (minimum useful set)

## Glossary of key IDs and metrics

### Core identifiers
- `user_id`: Stable internal identifier for a person account. Use this to connect behavior across visits for logged-in users.
- `session_id`: Stable identifier for one visit or browsing session. Use this to connect searches, clicks, filters, and page views within the same journey, including anonymous usage.
- `search_request_id`: Unique identifier for one search request. Use this to connect the search event to the exact results shown and any later clicks or downloads.
- `event_id`: Unique identifier for one generic event row, usually used in tables such as `engagement_events`.
- `result_id`: Unique identifier for the item shown or clicked, such as a KO, contribution, project, or document.
- `result_rank`: Position of the result in the returned search list, such as `1`, `2`, `3`.

### Search metrics
- `query_text`: The exact search text entered by the user.
- `query_hash`: Protected derived value of the query for safer long-term storage when raw text should not be retained.
- `query_language`: Detected or selected language of the query.
- `applied_filters` / `filters_json`: Filters used during search, such as topic, language, or content type.
- `result_list` / `results_json`: The ranked list of result IDs returned for the search, usually storing the top N items.
- `latency_ms`: Time taken to serve the search response in milliseconds.
- `page_number`: Which search results page the user is on, such as page `1`, `2`, or `3`.
- `result_count`: Number of results returned for a search.

### Engagement metrics
- `event_type`: Type of recorded action, such as `search`, `click`, `open_detail`, `download`, `bookmark`, `share`, or `not_relevant`.
- `dwell_time_ms`: Approximate time the user spent on a clicked result or detail page before returning or moving on.
- `source_context`: Where an action came from, such as `search_results`, `chatbot_suggestion`, `support_page`, or `direct_page_view`.
- `next_query_text`: A follow-up query entered after a search, useful for identifying query reformulation and dissatisfaction.
- `feedback_label`: Explicit feedback from the user, such as `helpful`, `not_relevant`, or `hide_result`.

### Platform analytics metrics
- `page_url`: URL of the visited page.
- `page_type`: Logical type of page, such as `home`, `search`, `contribution_detail`, `project_detail`, `faq`, or `support`.
- `referrer`: Referring source for the visit, such as a search engine, campaign, or external site.
- `utm_source`, `utm_medium`, `utm_campaign`: Campaign attribution fields for acquisition analysis.
- `country_code`: Country derived from network metadata or consented analytics tooling.
- `topic`: Topic or thematic category associated with a result or project.
- `project_type`: Type/category of project for reporting and segmentation.
- `locale`: User or session language/locale preference.

### 1) User identity and account context
- `user_id` (stable internal ID; primary key for personalization)
- `email` (already available)
- `name` (already available)
- `account_created_at`
- `locale` / preferred language (if available)

Use:
- Join behavior to a user profile.
- Language-aware ranking and query handling.

### 2) Search interaction events (most important)
Capture one row per search request:
- `event_type = search`
- `event_time`
- `user_id` (nullable for anonymous)
- `session_id`
- `query_text` (or hashed + protected raw access)
- `query_language`
- `applied_filters` (topics/themes/language/etc.)
- `result_list` (top N result IDs and rank positions)
- `latency_ms`

Use:
- Learn user interests from repeated topics/queries.
- Re-rank future results by user affinity.
- Evaluate quality and speed.

### 3) Click/engagement events
Capture one row per user interaction with a result:
- `event_type` in (`click`, `open_detail`, `download`, `bookmark`, `share`)
- `event_time`
- `user_id`
- `session_id`
- `query_id` / `search_request_id` (to connect click to a search)
- `result_id`
- `result_rank`
- `dwell_time_ms` (if available)

Use:
- Strong personalization signal (clicks and dwell beat raw query text).
- Train per-user and per-segment preference weights.

### 4) Explicit preferences (high-value, low-noise)
- Preferred languages
- Preferred topics/themes/categories
- Opt-in/out personalization flag

Use:
- Cold-start personalization before enough behavioral history exists.

### 5) Negative feedback (if product supports it)
- `hide_result`, `not_relevant`, `mute_topic`

Use:
- Prevent repeated bad recommendations.

## What not to capture by default
- Raw access tokens
- Full unnecessary headers
- Sensitive personal data not needed for ranking (phone, address, etc.)
- Free-form PII from unrelated forms

## Suggested storage model

### Core tables
1. `users`
- `user_id`, `email`, `name`, `locale`, `created_at`, `personalization_opt_in`

2. `search_events`
- `search_request_id`, `event_time`, `user_id`, `session_id`, `query_text`, `query_hash`, `query_language`, `filters_json`, `results_json`

3. `engagement_events`
- `event_id`, `event_time`, `user_id`, `session_id`, `search_request_id`, `result_id`, `result_rank`, `event_type`, `dwell_time_ms`

4. `user_preferences`
- `user_id`, `preferred_languages`, `preferred_topics`, `preferred_themes`, `updated_at`

## Retention and privacy guardrails
- Keep raw query text for a short period (example: 30-90 days), then keep only aggregates/hash as possible.
- Keep event-level interaction data only as long as needed for ranking quality and analytics.
- Make personalization opt-in/opt-out explicit.
- Add delete/export workflows for user data requests.
- Mask or truncate IPs if full IP is not required.

## Personalization rollout (pragmatic)
1. Start with `search_events` + `click` events + `user_id`.
2. Build a simple user profile vector from clicked topics/languages/keywords.
3. Re-rank baseline search results with a weighted blend:
   - `final_score = base_relevance * a + user_affinity * b + freshness * c`
4. Add explicit preferences and negative feedback later.

## Note for current codebase
Current code already logs useful search telemetry (`query`, filters, result IDs, user_id when token is present, headers/IP metadata).  
To support robust personalization, add:
- stable `session_id`,
- explicit engagement events (click/dwell/bookmark),
- preference capture endpoints,
- retention controls and token-safe logging.

## Tracking matrix from presentation items

| Presentation item | Metric / question | Event needed | Required fields | Needed for personalization? |
| --- | --- | --- | --- | --- |
| General overview / visitors | Number of visitors | `session_start`, `page_view` | `event_time`, `session_id`, `user_id` (nullable), `page_url` | No |
| General overview / visitor channels | Acquisition channel breakdown | `session_start` | `event_time`, `session_id`, `referrer`, `utm_source`, `utm_medium`, `utm_campaign` | No |
| General overview / most visited pages | Top pages visited | `page_view` | `event_time`, `session_id`, `user_id` (nullable), `page_url`, `page_type` | Indirectly useful |
| General overview / visitors' country of origin | Country distribution | `session_start` or derived session metadata | `event_time`, `session_id`, `country_code` | No |
| KOs / single contribution stats | Views of a contribution / KO | `open_detail` or `page_view` | `event_time`, `session_id`, `user_id` (nullable), `result_id`, `result_type` | Yes |
| KOs / single contribution stats | Downloads of a contribution / KO | `download` | `event_time`, `session_id`, `user_id` (nullable), `result_id`, `result_type`, `source_context` | Yes |
| Search | Search terms used | `search` | `search_request_id`, `event_time`, `session_id`, `user_id` (nullable), `query_text`, `query_language` | Yes |
| Search | Contributions clicked after search | `click` | `event_time`, `session_id`, `user_id` (nullable), `search_request_id`, `result_id`, `result_rank` | Yes |
| Search satisfaction from CM slide | Do users click results shown on page 1 / 2 / 3? | `search`, `click`, `pagination` | `search_request_id`, `session_id`, `user_id` (nullable), `result_list`, `result_rank`, `page_number` | Yes |
| Search satisfaction from CM slide | Are users satisfied with search results? | `search`, `click`, `open_detail`, `download`, `query_reformulation`, optional `feedback` | `search_request_id`, `session_id`, `user_id` (nullable), `result_id`, `result_rank`, `dwell_time_ms`, `next_query_text`, `feedback_label` | Yes |
| Contributions | Total contributions | `contribution_created` or database aggregate | `contribution_id`, `created_at`, `topic`, `project_type` | No |
| Contributions | Contributions per topic | `contribution_created` or database aggregate | `contribution_id`, `topic`, `created_at` | No |
| Contributions | Contributions per project type | `contribution_created` or database aggregate | `contribution_id`, `project_type`, `created_at` | No |
| Projects | Total number of projects | `project_created` or database aggregate | `project_id`, `created_at` | No |
| Projects | Number of projects per type | `project_created` or database aggregate | `project_id`, `project_type`, `created_at` | No |
| Registrations | Number of registrations | `user_registered` | `user_id`, `created_at`, `locale` | No |
| Contact | Queries submitted via contact form | `contact_form_submit` | `event_time`, `session_id`, `user_id` (nullable), `topic` | No |
| Support page | FAQs viewed | `faq_view` or `page_view` | `event_time`, `session_id`, `user_id` (nullable), `faq_id`, `page_url` | Indirectly useful |
| Support page | Guides / manuals downloaded | `manual_download` | `event_time`, `session_id`, `user_id` (nullable), `document_id`, `source_page` | Indirectly useful |
| Chatbot from CM slide | Do users click suggested KOs? | `chatbot_recommendation_impression`, `chatbot_recommendation_click` | `event_time`, `session_id`, `user_id` (nullable), `conversation_id`, `result_id`, `suggestion_rank` | Yes, but separate from search |
| UX from CM slide | What could we track to improve UX? | `filter_apply`, `pagination`, `empty_result`, `query_reformulation`, `support_page_visit_after_search` | `event_time`, `session_id`, `user_id` (nullable), `search_request_id`, `filters_json`, `page_number`, `result_count` | Indirectly useful |
| Platform usage from CM slide | What could we track about platform usage? | `page_view`, `open_detail`, `download`, `bookmark`, `share` | `event_time`, `session_id`, `user_id` (nullable), `page_type`, `result_id`, `result_type`, `topic`, `language` | Yes |

## Minimum next events to add

If the goal is search personalization, the highest-value next additions are:
- `session_id` on every search and engagement event
- `search_request_id` for each search
- `search` event with query, filters, and ranked result list
- `click` event tied to `search_request_id` and `result_rank`
- `open_detail` and `download` events tied to the same result
- `pagination` and `query_reformulation` events
- optional explicit feedback such as `not_relevant` or `helpful`

## Session ID recommendation

Use a stable `session_id` for all anonymous and logged-in activity within a visit.

Why it is needed:
- Link multiple searches, clicks, filters, and pagination steps into one search journey.
- Measure satisfaction patterns such as zero-click searches and repeated reformulations.
- Support anonymous behavior before login and later merge it to a known `user_id` if product rules allow.

If you are forced to reduce scope for an MVP, `user_id` + `search_request_id` is enough only for logged-in-only flows.  
For platform analytics plus realistic personalization, `session_id` should be treated as required.
