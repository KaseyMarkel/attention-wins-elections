# Edge Cases: Ambiguous Candidate Search Terms

Document all decisions about search strings BEFORE running queries.

## Bush Name Disambiguation

- **1988:** George H.W. Bush vs. Michael Dukakis
  - Search: `"Bush"` — only one Bush on ballot, acceptable
- **1992:** George H.W. Bush vs. Bill Clinton (+ Ross Perot)
  - Search: `"Bush"` — only one Bush running, acceptable
- **2000:** George W. Bush vs. Al Gore
  - Search: `"Bush"` — H.W. is out of office but still newsworthy; will use `"George W. Bush"` OR `"W. Bush"` to reduce false positives. Decision: use `"Bush"` for consistency, log potential overcounting.
- **2004:** George W. Bush vs. John Kerry
  - Search: `"Bush"` — same as above

## Clinton Name Disambiguation

- **1992, 1996:** Bill Clinton
  - Search: `"Clinton"` — Hillary not yet nationally prominent
- **2016:** Hillary Clinton vs. Donald Trump
  - Search: `"Clinton"` — Bill still active publicly; potential overcounting. Log and report sensitivity.

## Johnson Disambiguation

- **1964:** Lyndon B. Johnson vs. Barry Goldwater
  - Search: `"Johnson"` — common surname; may overcount. Will test `"Lyndon Johnson"` as sensitivity check.

## Decision Rule

For all searches: use **last name only** as the primary metric for consistency across eras.
Document any election where last name is particularly ambiguous. Run sensitivity checks
with full names for the 3 most ambiguous cases (Bush 2000/2004, Clinton 2016, Johnson 1964).
