# Contributor guidance

boltz-lumi is a small inference adapter for Boltz on LUMI. Preserve Boltz's
CLI, input handling, device selection, output layout and affinity prediction.
Keep numerical comparisons, timing hooks and profiling in explicit measurement
modes rather than ordinary prediction.

## Environment

Users and agents should consult LUMI's
[Infrastructure for AI agents](https://docs.lumi-supercomputer.eu/laif/software/agent-infrastructure/)
before setting up or using an agent on LUMI. It describes the agent environment,
data handling, permissions and current limitations. Users remain responsible
for commands run under their accounts. Agents should use the documented LUMI
MCP server, when available, to look up current site guidance and service status.

Follow the setup instructions in README.md. Keep local paths and Slurm accounts
in an untracked environment file selected with BOLTZ_LUMI_ENV. Do not commit
credentials, usernames, allocation numbers or personal filesystem paths.
Reuse the existing environment and caches when available. Run GPU setup and
checks inside a GPU allocation, and use device indices visible to the process:
one visible GCD is device 0. Leave unrelated jobs and environments alone.

## Changes and validation

Preserve FP32/BF16 rounding boundaries, arithmetic order and hardware, shape
and version guards. Keep fallbacks for unsupported cases. Name architecture
assumptions and explain constraints that are not clear from the code.

Run the relevant checks listed in README.md when changing kernels or adapters.
Exercise nonzero outputs, masks and tails. Numerical reports should include
absolute and relative errors, tolerances and failing counts. Check normal
prediction outputs and exceptions that Boltz reports without raising.

Separate setup and first use from warm model calls and complete input rounds.
State the cache conditions for measurements. GPU-event spans include dispatch
gaps, and overlapping spans cannot be added. Reuse existing performance
evidence when no relevant behavior changed.

Keep code and documentation concise. Comments should explain intent or
constraints. Preserve third-party license text and source notices.

## Versions

Use `<Boltz version>.<adapter revision>` as documented in README.md. When
upgrading Boltz, update SUPPORTED_BOLTZ_VERSION in src/boltz_lumi/__init__.py
and the exact dependency pin in requirements.txt together. Reset the adapter
revision for the new Boltz version and validate compatibility before release.
