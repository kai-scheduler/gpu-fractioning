# Governance

This document is the governance model for **gpu-fractioning**. It describes who decides what
in this repository, how those decisions are made, and how the people who make them change over
time.

gpu-fractioning is hosted in the `kai-scheduler` GitHub organization, which belongs to the
KAI Scheduler [Cloud Native Computing Foundation](https://cncf.io) sandbox project. That means
the [CNCF Charter](https://github.com/cncf/foundation/blob/main/charter.md) section 11 IP policy
binds this repository: code is licensed under the Apache License 2.0, documentation under the
Creative Commons Attribution 4.0 International License, and every inbound contribution must carry
a Developer Certificate of Origin sign-off. Those terms are not ours to change. Everything else
below is.

## Roles

### Maintainers

Maintainers are responsible for the long-term health of gpu-fractioning. They are listed in
[MAINTAINERS.md](MAINTAINERS.md), and that list drives review and merge permissions through
[OWNERS](OWNERS) and [CODEOWNERS](CODEOWNERS).

Maintainers:

- set the technical direction of this repository and decide what gets worked on;
- review and approve pull requests, and are the only people who can merge them;
- own the release process and the artifacts this repository publishes;
- keep documentation, dependencies and security posture current;
- uphold the [Code of Conduct](CODE_OF_CONDUCT.md) and mediate disputes.

### Contributors

Anyone who opens an issue or a pull request here. Contributors follow
[CONTRIBUTING.md](CONTRIBUTING.md), sign off every commit under the
[Developer Certificate of Origin](CLA.md), and are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

### Community

Everyone else with an interest in the project — adopters, integrators and users. Feedback, bug
reports and adoption stories ([ADOPTERS.md](ADOPTERS.md)) are contributions too.

## Decision making

Decisions are made by lazy consensus among this repository's maintainers, in public, on the pull
request or issue where the change is proposed.

- **Ordinary changes** — bug fixes, tests, documentation, dependency bumps: one maintainer
  approval plus green CI. Merging additionally requires code-owner review; the ruleset on `main`
  enforces both.
- **Significant changes** — a new field or version of the `GpuFractioningConfig` API, a change to
  the contract between the operator and the node-level daemons, a new component, a change to the
  enforcement or security model, or a change to the supported driver and runtime matrix: opened
  as an issue first, and merged only once no maintainer objects.
- **Disagreement** — if consensus is not reached, any maintainer may call for a vote. A simple
  majority of the maintainers listed in [MAINTAINERS.md](MAINTAINERS.md) decides.

### Changes that cross into KAI Scheduler

gpu-fractioning enforces on the node the GPU fractions that KAI Scheduler assigns. The two
projects therefore share an interface: the pod annotations KAI Scheduler writes and this project
consumes. Changes to that interface are agreed jointly with the
[KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) maintainers before either side
ships them. This is a technical dependency, not a reporting line — gpu-fractioning is
governed by the maintainers listed here.

## Changing the maintainer group

New maintainers are proposed by an existing maintainer, based on a sustained record of
contribution and review **in this repository**. The proposal is opened as a pull request against
[MAINTAINERS.md](MAINTAINERS.md), the community is invited to comment, and it is merged if no
existing maintainer objects within one week.

Maintainers who are inactive for six months, or who ask to step down, are moved out of
[MAINTAINERS.md](MAINTAINERS.md) by the same process. Stepping down is normal and carries no
stigma; former maintainers are welcome back on the same terms as anyone else.

The maintainer group is not required to remain single-vendor and is open to contributors from any
organization, consistent with CNCF's vendor-neutrality expectations.

## Code of Conduct

This repository has adopted the
[CNCF Code of Conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md); see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Violations are reported to the maintainers, or to the
CNCF Code of Conduct Committee at <conduct@cncf.io>, and are reviewed privately.

## Security

Vulnerabilities are reported privately through the channels in [SECURITY.md](SECURITY.md), never
as a public issue.

## Changing this document

By pull request, approved by a majority of this repository's maintainers.
