# surgical-workflow-analysis

## Subagent Model Selection
When spawning agents, pick the model based on task complexity:
- **opus** — complex multi-step reasoning, architecture decisions, deep analysis
- **sonnet** — medium tasks: implementation, code review, research with synthesis
- **haiku** — simple tasks: file searches, reads, quick lookups, pattern matching
