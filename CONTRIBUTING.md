# Contributing Guidelines

Thank you for your interest in contributing! We welcome contributions from research lab members, students, and the open-source community.

---

## Contribution Workflow

We follow a **Trunk-Based Development** model: all active development targets the `main` branch through short-lived branches and Pull Requests.

### 1. Issues First
Before writing code or opening a pull request, verify that an issue tracks the task:
- Navigate to the **Issues** tab and click **New issue**.
- Choose the relevant template (**Feature Task**, **Bug Report**, **Documentation**, or **Onboarding**).
- Fill in the requested details. Labels will be attached automatically.

---

### 2. Creating Your Branch

#### For Lab Members & Direct Collaborators
1. Open the assigned issue on GitHub.
2. In the right sidebar under **Development**, click **"Create a branch"**.
3. Use conventional branch prefixes:
   - `feat/<issue-id>-short-description`
   - `fix/<issue-id>-short-description`
   - `docs/<issue-id>-short-description`
   - `chore/<issue-id>-short-description`
4. Fetch and checkout the branch locally:
   ```bash
   git fetch origin
   git checkout <branch-name>
   ```

#### For External Contributors (Fork Workflow)
1. Fork the repository to your GitHub account.
2. Clone your fork locally and create a topic branch from `main`:
   ```bash
   git checkout -b feat/my-improvement
   ```

---

### 3. Pull Requests (PR)

1. Ensure all local tests and style checks pass prior to submission.
2. Push your topic branch to GitHub:
   - **Members:** `git push -u origin <branch-name>`
   - **External contributors:** `git push -u origin feat/my-improvement` (to your fork)
3. Open a Pull Request targeting the `main` branch.
4. Complete the PR template checklist and link the issue in the description (e.g., `Closes #15`).
5. A maintainer will review the code. All pull requests are merged using **Squash and merge**, and head branches are deleted automatically.

---

## Development & Code Quality

- Follow the project's formatting, linting, and style conventions.
- Ensure all public functions, classes, and APIs include descriptive documentation/docstrings.
- Add or update automated tests covering new behavior whenever applicable.

---

## Troubleshooting: Committed Directly to `main`?

If you committed directly to your local `main` branch and the push was blocked by branch protection rules, you can move your commits to a new branch without losing work:

```bash
# 1. Create a new topic branch preserving your local commits
git branch feat/<issue-id>-my-task

# 2. Reset your local main back to the clean remote state
git reset --hard origin/main

# 3. Switch to your topic branch and push normally
git checkout feat/<issue-id>-my-task
git push -u origin feat/<issue-id>-my-task
```

---

## License

By contributing, you agree that your contributions will be licensed under the project's repository license.
