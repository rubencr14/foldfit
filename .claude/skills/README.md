# Agent Skills

<p align="center">
  <img src="assets/images/banner.png" alt="Agent Skills Banner" width="100%" />
</p>

<p align="center">
  <strong>A curated collection of Claude Code skills for building production-grade software.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/skills-7-brightgreen.svg" alt="Skills" />
  <img src="https://img.shields.io/badge/claude--code-compatible-blueviolet.svg" alt="Claude Code" />
</p>

<p align="center">
  <a href="#skills">Skills</a> &bull;
  <a href="#how-it-works">How it works</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#structure">Structure</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

## What is this?

A set of opinionated, production-tested **skills for Claude Code** that encode best practices for full-stack development, testing, security, and architecture. Each skill teaches Claude *how* to build things the right way — not just what to build.

These skills are designed to be **lean and agent-optimized**: short instructions that Claude loads dynamically, with deep-dive references only when needed.

## Skills

| Skill | Lines | What it covers |
|-------|-------|----------------|
| **[backend-development](skills/backend-development/)** | 387 | Python + FastAPI architecture, light DDD, SOLID, Pydantic, pytest, container security |
| **[web-development](skills/web-development/)** | 495 | Next.js/React architecture, feature-first organization, ports & adapters, mappers, TypeScript contracts |
| **[web-testing](skills/web-testing/)** | 439 | Vitest + Testing Library, boundary testing, coverage, mocking (MSW), WebGL/Canvas, accessibility |
| **[cybersecurity](skills/cybersecurity/)** | 150 | OWASP top 10, container hardening, AI/agent security, prompt injection defense, checklists |
| **[frontend-design](skills/frontend-design/)** | 42 | UI aesthetics, design thinking, typography, color, motion, spatial composition |
| **[mcp-builder](skills/mcp-builder/)** | 236 | MCP server development in Python (FastMCP) and TypeScript, evaluation framework |

> `frontend-design`, `mcp-builder`, and `skill-creator` are sourced from [Anthropic's official skills](https://github.com/anthropics/skills).

## How it works

Skills follow the [Agent Skills specification](https://agentskills.io/specification). When Claude Code activates a skill:

1. **Metadata loads** (~100 tokens) — skill name and description, always in context
2. **SKILL.md loads** — the full instructions, when the skill triggers
3. **References load on demand** — deep dives only when Claude needs them

```
User asks to "build a FastAPI service"
  → Claude matches "backend-development" skill
    → SKILL.md loads: architecture rules, golden rules, decision tree, examples
      → Claude follows the patterns: DDD layers, Pydantic validation, pytest tests
```

## Installation

### Option 1: Add to your project

Copy the `skills/` directory into your project:

```bash
cp -r skills/ /path/to/your-project/.claude/skills/
```

### Option 2: Global installation

Add skills globally so they're available in every project:

```bash
cp -r skills/ ~/.claude/skills/
```

### Option 3: Reference from settings

Add to your `.claude/settings.json`:

```json
{
  "skills": [
    "/path/to/agent-skills/skills/"
  ]
}
```

## Structure

```
agent-skills/
├── skills/
│   ├── backend-development/     # Python + FastAPI
│   │   └── SKILL.md
│   ├── web-development/         # Next.js + React
│   │   ├── SKILL.md
│   │   └── references/
│   ├── web-testing/             # Vitest + Testing Library
│   │   ├── SKILL.md
│   │   └── references/
│   ├── cybersecurity/           # Security + hardening
│   │   ├── SKILL.md
│   │   ├── references/
│   │   ├── examples/
│   │   └── checklists/
│   ├── frontend-design/         # UI design quality
│   │   └── SKILL.md
│   └── mcp-builder/             # MCP servers
│       ├── SKILL.md
│       ├── reference/
│       └── scripts/
├── claude-skills/               # Anthropic's official skills (gitignored)
└── readme.md
```

### Design principles

- **SKILL.md carries the weight.** Core rules, decision trees, and inline examples live in the main file. Claude reads this first and often needs nothing else.
- **References are optional deep dives.** Loaded only when Claude needs detail on a specific topic (e.g., WebGL testing, container hardening).
- **Under 500 lines per SKILL.md.** Respects Claude's context window. Every line must justify its token cost.
- **Examples inline, not in separate files.** Claude learns from examples — they belong next to the rules they demonstrate.
- **Every feature must include tests.** This is enforced in every skill that involves code.

## Contributing

### Adding a new skill

1. Create `skills/your-skill/SKILL.md` with YAML frontmatter:

```yaml
---
name: your-skill
description: Pushy description of when Claude should use this skill (100-1024 chars)
---

Your instructions here...
```

2. Keep SKILL.md under 500 lines
3. Add references/ only for genuinely complex topics that don't fit inline
4. Follow the pattern: principles first, decision tree, rules, inline examples

### Quality checklist

- [ ] Description is "pushy" — explicitly states when to trigger
- [ ] Instructions are imperative (do X, not "you should do X")
- [ ] Examples show the pattern, not the full implementation
- [ ] Every rule explains *why*, not just *what*
- [ ] No unnecessary files — challenge every file's token cost

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

Some skills (`frontend-design`, `mcp-builder`, `skill-creator`) are sourced from [Anthropic's official skills repository](https://github.com/anthropics/skills) and are subject to their original license terms.

---

<p align="center">
  Built by <a href="https://github.com/rubencr14">Rubén Cañadas</a> &bull; For <a href="https://claude.ai/claude-code">Claude Code</a> &bull; Following the <a href="https://agentskills.io/specification">Agent Skills Specification</a>
</p>