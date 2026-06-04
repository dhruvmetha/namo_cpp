# scripts/sandbox/ — disposable one-offs

Everything in here is **gitignored** (except this README and `.gitignore`).

Put scripts here when they are point-in-time / machine-specific / run-once:
diagnostics, profiling, ad-hoc cleanups, experiment probes. Anything with a
hardcoded date, absolute scratch path, or a corpus version baked in.

Durable, reusable pipeline and eval tooling goes in `scripts/` (committed).

Rule of thumb: if re-running it next month would be *wrong* (stale assumptions),
it lives here. If it should still work next month, it belongs in `scripts/`.
