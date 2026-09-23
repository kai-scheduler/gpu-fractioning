# Support

How to get help with **gpu-fractioning**. It is community supported by its
[maintainers](MAINTAINERS.md); it does not come with a commercial support contract.

## Where to get help

| I want to… | Go to |
| --- | --- |
| Understand what this does and how to install it | [README.md](README.md) |
| Ask a question, or discuss an idea | The `#kai-scheduler` channel on the [CNCF Slack](https://slack.cncf.io), which the maintainers of this repository read |
| Report a bug, or request a feature | [GitHub issues](https://github.com/kai-scheduler/gpu-fractioning/issues) |
| Report a security vulnerability | **Not an issue** — the private channels in [SECURITY.md](SECURITY.md) |
| Contribute a change | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Know who maintains this | [MAINTAINERS.md](MAINTAINERS.md) |

Maintainers triage issues on a best-effort basis. There is no response-time commitment, but
issues that include the information below get answered fastest.

## What to include in a bug report

- The version of the chart or images you are running, or the commit SHA.
- Kubernetes version and distribution, and the container runtime (containerd or CRI-O) with its
  version — the NRI interface differs between runtime versions.
- NVIDIA driver version and NVIDIA GPU Operator version, and the GPU model.
- The `GpuFractioningConfig` resource, and the pod spec or annotations of the affected workload.
- `kubectl describe` output and logs from the relevant component (`operator`, `fractiond`, `mpsd`
  or `metricsd`), plus the `gpu-fractioning.nvidia.com/Ready` node condition of the affected node.

## Release cadence

There is no fixed release schedule. The project maintains the current release: bug fixes and
security fixes are merged to `main` as they are resolved and shipped in patch releases when
warranted.

Subsequent minor or major releases are driven by incoming feature requests and roadmap items.
None are currently planned, so no future release dates are committed. Feature requests are
tracked as [GitHub issues](https://github.com/kai-scheduler/kai-gpu-fractioning/issues).

## Supported versions

Fixes land on `main` and ship in the next release. Only the most recent release is supported;
there is no long-term-support branch for this repository. Security fixes are handled as described
in [SECURITY.md](SECURITY.md).

Cluster prerequisites — a supported NVIDIA driver, the NVIDIA GPU Operator and a container
runtime with NRI enabled — are documented in [README.md](README.md); reports against
unsupported prerequisite versions will be closed with a pointer there.
