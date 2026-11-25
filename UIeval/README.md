# UIeval: live training dashboard

A separate, read-only Streamlit process for `models/trained_model/`. It never
imports torch, loads checkpoints, starts training or changes a run's files.

## Start on the server

From the repository root, in a second terminal:

```bash
python -m venv UIeval/.venv
source UIeval/.venv/bin/activate
python -m pip install -r UIeval/requirements.txt
python UIeval/run.py --port 8501
```

The launcher prints local/network browser addresses. On your PC, open
`http://SERVER_IP:8501` using a reachable server IP. The port must be reachable
through your server/network configuration. `0.0.0.0` is a listening address,
not an address to type into the browser.

On Windows, activate with `UIeval\.venv\Scripts\Activate.ps1`, or run:

```powershell
UIeval\.venv\Scripts\python.exe UIeval/run.py
```

For SSH forwarding, bind only to the server's loopback interface:

```bash
# Server
python UIeval/run.py --host 127.0.0.1 --port 8501

# Your PC, in a separate terminal
ssh -N -L 8501:127.0.0.1:8501 USER@SERVER_IP
```

Then open `http://localhost:8501` on your PC. The dashboard has no built-in login;
use the SSH route when the server should not expose it on the network.

## Use

- Select a training folder from the sidebar. Use **Rescan runs** to discover new runs.
- **Live updates** defaults to on, polling every **2 seconds**; change to 1–10 seconds.
- Pause live updates or refresh manually without stopping training.
- Filter per-pass charts/table by epoch, plate or cell line. Empty selections mean all.
- Choose rolling smoothing and a backward-pass or optimizer-step horizontal axis.
- Select gradient modules and linear/log axes. Plotly zoom/legend state is retained
  during periodic refreshes when the chart identity remains the same.
- Overlay up to three other runs under **Compare runs**.

Views include loss history and saved epoch scores; module gradients; recorded
step time and peak allocated VRAM; architecture/parameter counts; and the latest
1,000 matching log rows. Comparisons use entire runs rather than the selected
run's filters. Incompatible loss targets/scales trigger a comparison warning.

Top cards summarize the entire selected run. Completed-epoch scores come from
`info.json` and update at epoch boundaries. Reconstruction probes from training
files are labeled separately from held-out validation. Per-pass gradient norms
are not the accumulated pre-clipping norm, so the dashboard does not invent a
clipping rate. VRAM is recorded training memory, not live GPU utilization.

## Data handling

Reads `log.csv`, `info.json`, and `config.json` only. The log reader incrementally
reads appended data and waits for complete records; a partly written last row is
shown on the next poll after it finishes. Malformed records or non-finite required
values are skipped with a count. Non-finite gradient values are omitted from
charts and flagged. A replaced/truncated CSV, including after checkpoint resume,
resets the reader. JSON reads retain their last good value during temporary
write/replace errors.

Large curves are smoothed before downsampling for display (at most about 1,800
points per trace). The in-memory data still includes all complete records.
Charts with zero gradients omit those values on logarithmic axes. A stale log
is not automatically called a failed run: loading/evaluation can take time.

Override the run location:

```bash
python UIeval/run.py --runs-root /scratch/project/models/trained_model
```

The sidebar also accepts a directory. Direct Streamlit launching is supported;
run from `UIeval/` to use its local theme:

```bash
cd UIeval
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

No dashboard dependency is added to `requirements-training.txt`.

## Organization

- `run.py`: launcher, library checks and browser addresses.
- `app.py`: Streamlit layout, controls and timed rendering fragment.
- `data.py`: incremental CSV/JSON readers and pure filtering helpers.
- `charts.py`: Plotly chart builders.
- `.streamlit/config.toml`: theme and headless settings.
- `requirements.txt`: dashboard-only dependencies.
- `tests/test_dashboard.py`: synthetic live-file and Streamlit interaction tests.

Test from the repository root using the dashboard environment:

```bash
python -m unittest discover -s UIeval/tests -v
```
