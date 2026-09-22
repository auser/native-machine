# Contributing

## Local setup

Enable the repository-managed Git hooks once per clone:

```text
just install-hooks
```

The pre-commit hook runs rustfmt when Rust files are staged and re-stages the
formatted files for the same commit. If a Rust file is only partially staged,
the hook stops and asks you to stage or stash the remaining hunks first.

## Branch policy

`main` is the release branch. All changes enter through pull requests. Direct
pushes are disabled in the GitHub repository settings.

```text
git switch main
git pull --ff-only origin main
git switch -c feat/short-description
just ci
git push -u origin feat/short-description
```

Open a pull request with a focused description and evidence of local checks.
Keep commits reviewable. Do not merge locally or bypass required checks.

## Merge queue

After review and successful pull-request CI, enqueue the pull request in
GitHub. GitHub will test the merge-group result before updating `main`. The CI
workflow listens for `merge_group`, so the required check must be configured as
`CI / rust` in branch protection.

## Release procedure

Only release from an up-to-date `main` checkout:

```text
git switch main
git pull --ff-only origin main
just release-check
just release version=0.1.0
```

The release command creates and pushes a semantic-version tag. GitHub Actions
publishes the corresponding release. Release versioning follows semantic
versioning.

Alternatively, after a reviewed pull request has merged, push a semantic
version tag and let GitHub Actions publish the release:

```text
git switch main
git pull --ff-only origin main
git tag v0.1.0
git push origin v0.1.0
```
