# Roadmap

Planned work for gpu-fractioning is tracked in the open:

- **[GitHub issues](https://github.com/kai-scheduler/gpu-fractioning/issues)** — everything the
  maintainers intend to work on has an issue. Issues carry the current thinking and the discussion
  behind it, which a document like this one cannot keep up with.
- **[Milestones](https://github.com/kai-scheduler/gpu-fractioning/milestones)** — what is
  scheduled for the next release.
- **[CHANGELOG.md](CHANGELOG.md)** — what has already landed, including the unreleased section.

We would rather keep one accurate list than two, so this document deliberately does not restate
them.

## Proposing something

Open an issue describing the problem you want solved. Significant changes — a new field or version
of the `GpuFractioningConfig` API, a change to the contract between the operator and the
node-level daemons, a new component, or a change to the enforcement or security model — start as
an issue before any pull request; see [GOVERNANCE.md](GOVERNANCE.md).

gpu-fractioning enforces on the node the GPU fractions that
[KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) assigns. Changes to the pod
annotations the two projects share are agreed with the KAI Scheduler maintainers before either
side ships them, so issues that touch that interface may be sequenced against their work.
