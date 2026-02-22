# Code Review Rules

## goal and task

- this project provides MCP/cli for code search. 
  - the existing cli `scripts/ctx.py` only implements partial features of mcp servers. 
  - in current git uncommited files, it's a new cli in a new separate folder `cli` that has implemented full features of code search and auto reindexing, using programatic function call and import useful existing code/lib instead of using mcp api call. features related to memory/context enhancement are not required to implement, focusing on good code search first. 
  - there is a new skill at `skills/context-engine-cli/SKILL.md` to provide the new cli for other agent clients.

- the goal is to review the uncommited code and provide feedback for improvement, focusing on logic correctness, code reuse, maintainability.
  - review the mcp implementation and the new cli implementation, recheck code resuse and extensibility. analyze the architecture and core data flow, make sure the new cli code indexing and search is correct and clear.  provide more concise agent skills at skills/context-engine-cli/SKILL.md, removing commands that are unrelated to code indexing and search.

## Critical Rules (ALL files)

REJECT if:

- Hardcoded secrets/credentials
- `console.log` in production code

## TypeScript

- No `any` types - use proper typing
- Use `const` over `let` when possible

## React

- Use functional components with hooks

## Styling

- Use Tailwind CSS utilities only
- No inline styles or CSS-in-JS

## References

- architecture docs: `docs/ARCHITECTURE.md`
