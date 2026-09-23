# Contributing to gpu-fractioning

Thank you for contributing to gpu-fractioning.

Read the [Code of Conduct](CODE_OF_CONDUCT.md) and the
[Developer Certificate of Origin](CLA.md) before submitting a change.

## Developer Certificate of Origin (DCO)

This project does **not** use a Contributor License Agreement. Contributions are
accepted under the [Developer Certificate of Origin 1.1](CLA.md), the same
mechanism used by [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler).

By signing off on a commit you certify the statements in [CLA.md](CLA.md). Sign
off by adding a `Signed-off-by` trailer to every commit:

```sh
git commit -s -m "fix(fractiond): correct memory limit defaulting"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match the commit author. Configure them once with:

```sh
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

**Commits without a valid sign-off fail CI.** The
[DCO workflow](.github/workflows/dco.yaml) checks every commit in a pull request
and blocks the merge if any is unsigned. To fix an existing branch:

```sh
git rebase --signoff origin/main   # sign off every commit on the branch
git push --force-with-lease
```

## License headers

Every source and configuration file carries the project's Apache-2.0 SPDX
header. Add it to new files with:

```sh
make gen-license
```

`make validate` runs `make license-check`, which fails if any file is missing
the header. This is enforced in CI, so run it before opening a pull request.

## Commits

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

Format: `type(scope): description`

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`

Scopes: `operator`, `mpsd`, `fractiond`, `fractioning-manager`, `deps`

## Pull Requests

- One logical change per PR
- Every ready PR references at least one existing open issue
  - Use `Fixes #123` when merging the PR fully resolves and should close the issue
  - Use `Refs #123` when the PR addresses only part of the issue or the issue should
    remain open
  - Dependabot-authored dependency updates are exempt because they are generated
    automatically; maintainers must still review and approve them
- Dependency updates must pass `make third-party-check`. If it fails, a
  maintainer must run `make third-party` and commit the updated
  `THIRD-PARTY.txt` to the Dependabot PR before merging
- PR title must follow Conventional Commits
- Every commit is signed off (DCO)
- All CI checks must pass
- At least one approval required
- Use **Request changes** for blocking review feedback; maintainers may also add
  `do-not-merge/hold` when a PR must remain blocked for a non-review reason
- Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` for user-visible
  changes

## Code Style

- Run `make validate` before submitting
- Follow standard Go conventions

## Security

Do not report security vulnerabilities through public issues or pull requests.
Follow [SECURITY.md](SECURITY.md).
