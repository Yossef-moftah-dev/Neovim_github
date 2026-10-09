---
trigger: always_on
---

# GitHub & Git Rules for the Antigravity Agent

These rules are mandatory. If a user request conflicts with a rule marked **NEVER** or **ALWAYS**, stop, explain the conflict, and ask for explicit confirmation before proceeding.

---

## 1. Safety First (Non-Negotiable)

- **NEVER** force push (`--force`, `-f`) to `main`, `master`, `develop`, or any `release/*` branch. On feature branches use `--force-with-lease` only, and only after telling the user.
- **NEVER** commit directly to `main`/`master`. All changes go through a branch and a pull request.
- **NEVER** commit secrets: API keys, tokens, passwords, `.env` files, private keys, certificates, or credentials. If found in staged files, unstage them immediately and warn the user.
- **NEVER** run destructive commands without explicit user approval: `git reset --hard`, `git clean -fdx`, `git push --delete`, `git branch -D`, `git rebase` on shared branches, history rewrites.
- **NEVER** bypass hooks or checks (`--no-verify`, `--no-gpg-sign`) unless the user explicitly asks.
- **NEVER** modify git config (global or local), remotes, or credentials unless asked.
- **ALWAYS** run `git status` and `git diff --staged` before committing so you know exactly what is being committed.
- **ALWAYS** confirm the current branch (`git branch --show-current`) before committing or pushing.
- If a secret was already pushed, tell the user to **rotate it immediately**; removing it from history is not enough.

---

## 2. Branching Strategy

- Branch from an up-to-date default branch: `git fetch origin && git switch -c <branch> origin/main`.
- Use this naming format: `<type>/<short-kebab-description>` or `<type>/<issue-number>-<description>`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`, `hotfix`
  - Examples: `feat/user-auth`, `fix/142-null-pointer-login`, `docs/update-readme`
- One branch = one logical change. Do not mix unrelated work.
- Keep branches short-lived. Rebase or merge the default branch regularly to avoid large conflicts.
- Delete merged branches (locally and remotely) only after the PR is merged and the user agrees.

---

## 3. Commits

Follow **Conventional Commits**:

```
<type>(<optional scope>): <imperative summary, max 72 chars>

<optional body: what and why, wrapped at 80 chars>

<optional footer: BREAKING CHANGE: ..., Closes #123>
```

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Use imperative mood: "add login validation", not "added" or "adds".
- No trailing period in the summary line.
- Explain **why**, not just what, in the body when the change is non-obvious.
- Reference issues in the footer: `Closes #12`, `Fixes #34`, `Refs #56`.
- Make small, atomic commits. Each commit should build and pass tests on its own.
- Stage files explicitly (`git add <file>` or `git add -p`). Avoid `git add .` unless you have reviewed `git status`.
- Never commit generated artifacts, build output, `node_modules`, `.DS_Store`, logs, or IDE files. Check `.gitignore` first and update it if needed.
- Do not commit commented-out code, debug prints, or stray TODOs without issue references.

---

## 4. Pull Requests

When creating a PR (prefer the `gh` CLI: `gh pr create`):

- **Title:** Conventional Commit style, e.g. `feat(auth): add OAuth login`.
- **Description must include:**
  1. **Summary:** what changed and why
  2. **Changes:** key points
  3. **Testing:** how it was verified (commands run, results)
  4. **Related issues:** `Closes #123`
  5. **Screenshots/logs** for UI or behavior changes
  6. **Breaking changes / migration notes**, if any
- Follow the repo's PR template (`.github/pull_request_template.md`) if one exists.
- Keep PRs small and focused (ideally under ~400 changed lines). Suggest splitting larger ones.
- Open as **Draft** if work is incomplete or tests are failing.
- Before opening a PR: run linters, formatters, type checks, and the full test suite locally. Fix failures first.
- Self-review the diff (`git diff origin/main...HEAD`) and remove noise before requesting review.
- Do **not** merge a PR yourself unless the user explicitly instructs it and all required checks and approvals are satisfied.
- Prefer **squash merge** for feature branches unless the repo specifies otherwise. Follow the repository's configured merge strategy.

---

## 5. Code Review

- When reviewing, check: correctness, tests, security, performance, readability, and consistency with project conventions.
- Be specific and constructive. Reference file and line, explain the reason, and suggest a fix.
- Distinguish **blocking** issues from **nits/suggestions**.
- When addressing review feedback: push follow-up commits (or fixup commits), reply to each comment, and do not resolve threads you haven't actually addressed.
- Never dismiss reviews or approve your own work.

---

## 6. Issues

- Search existing issues before creating a new one to avoid duplicates.
- Bug reports must include: steps to reproduce, expected vs actual behavior, environment/version, logs or error output.
- Feature requests must include: problem statement, proposed solution, alternatives considered, acceptance criteria.
- Apply relevant labels and link related issues/PRs.
- Use the repo's issue templates in `.github/ISSUE_TEMPLATE/` when available.

---

## 7. Syncing & Conflict Resolution

- `git fetch` before starting work and before pushing.
- Prefer `git pull --rebase` on personal feature branches; use merge on shared branches. Follow the repo's convention.
- When resolving conflicts:
  1. Read both sides and understand the intent of each change.
  2. Never blindly pick "ours" or "theirs".
  3. Re-run build and tests after resolving.
  4. Tell the user which files conflicted and how you resolved them.
- If a conflict is ambiguous or touches critical logic, stop and ask the user.

---

## 8. CI/CD & GitHub Actions

- After pushing, check CI status (`gh pr checks` / `gh run list`). If CI fails, read the logs (`gh run view --log-failed`) and fix the cause.
- **Never** disable, skip, or weaken tests/checks just to get a green build.
- Workflow files (`.github/workflows/*.yml`):
  - Pin third-party actions to a **full commit SHA** (or at least a version tag); avoid `@main`/`@master`.
  - Use the **least-privilege** `permissions:` block (default to `contents: read`).
  - Store secrets in GitHub Secrets; never hardcode them or echo them in logs.
  - Never use `pull_request_target` with untrusted code checkout.
  - Avoid injecting untrusted input (`github.event.*.title`, branch names, etc.) directly into `run:` scripts; pass it via environment variables.
  - Cache dependencies and set sensible `timeout-minutes`.
- Do not modify CI configuration unless the task requires it, and call out CI changes clearly in the PR.

---

## 9. Releases & Tags

- Follow **Semantic Versioning** (`MAJOR.MINOR.PATCH`): breaking, feature, fix.
- Tags use the `vX.Y.Z` format and are annotated: `git tag -a v1.2.0 -m "Release v1.2.0"`.
- Never move or delete a published tag.
- Maintain `CHANGELOG.md` (Keep a Changelog format) when the repo uses one.
- Create releases with `gh release create`, including generated notes. Only do this when the user asks.

---

## 10. Repository Hygiene & Security

- Respect branch protection rules, required reviews, and status checks. Never attempt to circumvent them.
- Keep `.gitignore`, `README.md`, `LICENSE`, `CONTRIBUTING.md`, and `SECURITY.md` accurate when your changes affect them.
- Update documentation and tests in the same PR as the code change.
- Do not add large binaries; use Git LFS if the repo does.
- Do not add new dependencies without justification. Check license compatibility and known vulnerabilities, and mention additions in the PR.
- Prefer signed commits if the repo requires them.
- Treat Dependabot/security alerts as high priority; surface them to the user.
- Do not make changes to repository settings, collaborators, webhooks, or secrets via `gh` or the API unless explicitly asked.

---

## 11. Using the `gh` CLI and Terminal

- Prefer `gh` for GitHub operations (PRs, issues, runs, releases) over raw API calls.
- Verify authentication with `gh auth status` before relying on it. Never print or store tokens.
- Do not auto-run commands that push, merge, delete, or publish. Present the command and wait for approval unless the user has granted autonomy for that action.
- Show the user a summary of what was done after every Git operation: branch, commit hash, and PR/issue URL.

---

## 12. Standard Workflow Checklist

For any code change, follow this sequence:

1. `git fetch` and confirm the base branch is current.
2. Create a properly named branch.
3. Make focused changes with tests and docs.
4. Run lint, format, type-check, and tests.
5. Review `git status` and `git diff`; check for secrets and junk files.
6. Commit using Conventional Commits.
7. Push the branch (no force push).
8. Open a PR with a complete description (draft if unfinished).
9. Monitor CI and fix failures.
10. Address review feedback; wait for approval before merging.

---

## 13. When Unsure

- If the requested Git action is risky, irreversible, or ambiguous, **stop and ask**.
- If repo-specific conventions (CONTRIBUTING.md, templates, CODEOWNERS) conflict with these rules, follow the repo's conventions and mention the difference.
- Be transparent: never claim a command succeeded, tests passed, or CI is green without verifying it.
