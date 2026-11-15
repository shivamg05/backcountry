# Run an optimization

Start with [README.md](README.md) for installation and the four commands.
Detailed model and manifest options are in [docs/usage.md](docs/usage.md).

If you are an agent operating a run for someone, read
[docs/operating.md](docs/operating.md) before starting. It defines preflight,
event-driven progress updates, failure reporting, and how to finish early
without losing final validation. Follow the requested manifest and judge;
do not inspect other runs unless asked.

```sh
backcountry start path/to/manifest.yaml --judge claude-cli   # in the background
backcountry status --wait    # repeat: each call returns at the next milestone; relay it
```

`start` keeps running if the chat session or terminal closes; `status` (add
`--json` for fields) works from any later session. `run` does the same in the
foreground.

The default output is a fresh timestamped folder under `runs/`. You can specify
`--work-dir runs/YOUR_LABEL` or another location. Never reuse a
nonempty directory or run competing GPU benchmarks alongside an optimization.

## Pause and resume

A run can stop and continue later, in a new process, where it left off:

```sh
backcountry pause      # the newest run; --work-dir names another
backcountry resume     # later, from the same folder
```

- **Pause before the Mac sleeps, restarts or needs its GPU.** The job stops at
  its next safe point, after the attempt it is on, with nothing left running
  on the GPU, and exits with status `paused`. An attempt can take a while (on
  FLUX, half take over 10 minutes and the longest took an hour). If the
  operator can't wait, press Ctrl-C instead and resume later: only the attempt
  in progress is lost, and it is tried again. A sleeping Mac can also drop the
  job's network; don't rely on the job surviving a sleep.
- **Resume** reuses the manifest and judge settings the job started with
  (`--judge`, `--model`, `--judge-effort`, `--judge-cmd` override them). It
  records the model again (seconds), restores everything setup measured,
  reinstalls the accepted kernels, checks the model's outputs still match the
  original, and continues the same region with the same history and budget. It
  refuses if the manifest changed, and warns if the tool's code changed.
- **The job pauses itself** when the judge doesn't answer three times in a
  row (no network, an expired login, a reply over 30 minutes). Those failed calls don't use attempts.
  Fix the cause, then resume.
- **Ctrl-C, a crash or a killed process** can be resumed too, from the last
  safe point: at most the one attempt in progress is repeated. A win accepted
  in that attempt is found again; its checkpoint folder stays, and the repeat
  is saved beside it.
- **A finished job deletes its `boundaries/`** (the captured region inputs
  only a search reads, often tens of GB). A paused, failed or killed job keeps
  them so it can resume.
- **An export that fails after a passed final check** (a crash, a bug, out of
  memory) resumes at the export: the search and the final check are not run
  again.
- Runs started before pause and resume existed have no saved state and can't
  be resumed.

After completion, report the confirmed whole-workload result and artifact path,
or explain why nothing shipped. See [artifact usage](docs/artifacts.md).
