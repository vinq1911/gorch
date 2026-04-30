# Plans

Forward-looking design notes that are too speculative to be ADRs (which
record decisions already made and accepted) but durable enough to outlive
a chat session.

Each plan starts with a status (`proposed` / `in progress` / `superseded`),
captures the goal, the trade-offs considered, and explicit non-goals. When
a plan becomes a decision, port the relevant pieces to `doc/decisions.md`
as an ADR and mark the plan superseded.

| File | Topic |
| --- | --- |
| `0001-openmythos-port.md` | Roadmap for porting OpenMythos to Go on gorch (mythos_tiny on TinyStories as v1) |
| `0002-bf16-support.md` | bf16/fp16 dtype support track in gorch (parallel to mythos work) |
| `0003-gemini-review.md` | Review of an external advisory on scaling gorch toward GPT-4-class LLMs |
