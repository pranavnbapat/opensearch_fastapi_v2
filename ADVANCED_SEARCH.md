# Advanced Search

`/neural_search_relevant_advanced` is an isolated experimental endpoint for boolean-aware search.

## Goal

Keep `/neural_search_relevant` stable while allowing more explicit query control in a separate path.

## Request flag

The endpoint accepts:

- `advanced: true`
- `advanced: false`

Behavior:

- `advanced: true`
  - use the advanced parser and advanced search features described below
- `advanced: false`
  - behave like `/neural_search_relevant`
  - no boolean parser
  - no `-project` / `--project` handling

## Supported syntax

- uppercase boolean operators only: `AND`, `OR`, `NOT`
- parentheses: `(` `)`
- quoted phrases like `"crop rotation"`
- project scoping with `-project` or `--project`
- field-specific search:
  - `title:"soil health"`
  - `keywords:"crop rotation"`
  - `project:BEST4SOIL`
  - `acronym:BEST4SOIL`
  - `type:"Horizon Europe"`
- required field clauses:
  - `+title:"soil health"`
  - `+keywords:"crop rotation"`
- result mode controls:
  - `mode:strict`
  - `mode:broad`
  - `mode:semantic`
  - `mode:lexical`

Examples:

- `crop rotation AND soil health`
- `crop rotation OR soil health`
- `crop rotation AND NOT pesticide`
- `"crop rotation" AND "soil health"`
- `(crop rotation OR cover crops) AND soil health`
- `soil health --project "BEST4SOIL"`
- `(soil health OR carbon sequestration) -project BEST4SOIL`
- `title:"soil health" AND keywords:"crop rotation"`
- `+title:"soil health" crop rotation`
- `mode:lexical title:"soil health"`
- `mode:semantic crop rotation AND soil health`

## Current behavior

### Positive clauses

Positive terms or phrases use:

- semantic evidence across `title_embedding`, `subtitle_embedding`, `description_embedding`, `keywords_embedding`, `content_embedding`
- lexical evidence across `title.en`, `subtitle.en`, `description.en`, `keywords.en`, `content_chunk.en`
- extra precision boosts for:
  - `AND`-style lexical matching
  - phrase matching on title/subtitle/description
  - exact `project_acronym`

This means a positive clause can match through semantic or lexical evidence.

### Field-specific search

Supported scoped query fields:

- `title:`
- `subtitle:`
- `description:`
- `keywords:`
- `content:`

Examples:

- `title:"soil health"`
- `keywords:"crop rotation"`
- `description:compost`

### Structured query filters

Supported structured fields inside the query syntax:

- `project:`
- `acronym:`
- `type:`
- `topic:`
- `theme:`
- `location:`
- `language:`
- `category:`

Examples:

- `project:BEST4SOIL`
- `acronym:BEST4SOIL`
- `type:"Horizon Europe"`
- `soil health NOT location:France`

### Required field clauses

`+title:` and `+keywords:` act as required high-precision field clauses.

Examples:

- `+title:"soil health"`
- `crop rotation AND +keywords:"soil health"`

### Negative clauses

`NOT` is applied using lexical exclusion only.

Reason:

- semantic exclusion is too risky and opaque
- lexical exclusion is more predictable for advanced search users

### Project scoping

`-project` and `--project` add a filter constraint, not a ranking boost.

Behavior:

- exact `project_acronym` match
- `project_name` phrase match
- `project_name` lexical match with `operator=and`

Examples:

- `soil health --project "BEST4SOIL"`
- `"crop rotation" AND soil health -project AgriLink`

### Result modes

`mode:` changes how positive content clauses are interpreted.

- `mode:strict`
  - tighter lexical matching
- `mode:broad`
  - broader semantic + lexical recall
- `mode:semantic`
  - semantic-only content matching
- `mode:lexical`
  - lexical-only content matching

## Boolean model

The parser now supports boolean precedence and parentheses:

- `NOT` binds first
- `AND` binds before `OR`
- parentheses override precedence
- contiguous non-operator words are grouped into one concept term
- lowercase `and`, `or`, `not` are treated as normal text, not operators

Examples:

- `crop rotation AND soil health`
  - two concept terms: `crop rotation` and `soil health`
  - both concepts required

- `crop rotation OR soil health`
  - either concept may match

- `crop rotation AND NOT pesticide`
  - `crop rotation` is treated as one concept
  - `pesticide` excluded lexically

- `(crop rotation OR cover crops) AND soil health`
  - one of the first two concepts must match
  - `soil health` must also match

## Notes

- Parent grouping still happens on `parent_id`
- query-time meta docs are still excluded with `chunk_index = -1`
- the endpoint returns `_meta.advanced_search` with parse information for debugging
- `_meta.advanced_search.parsed_query` includes:
  - parsed AST
  - project filters
  - mode

## Current limitations

- malformed boolean queries currently return an empty result with parse metadata
- unquoted adjacent words are grouped into one concept term
- request-level filters and query-syntax filters can both be used, but that may be redundant
- per-result clause-match tracing is still limited

## Why this is separate

Your evaluated default strategy is the tuned semantic-first endpoint. Advanced boolean behavior changes retrieval semantics enough that it should remain isolated until tested properly.
