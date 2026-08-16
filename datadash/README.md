# SPAR Data Dashboard

This dashboard uses React for the frontend and Flask for the backend. It lets you inspect SPAR states and effects across registered environments and sweep effect configurations to find good `gen_data` presets.

It is packaged as the optional `spar-datadash` distribution and exposed as the `spar[datadash]` extra, so `pip install "spar[datadash]"` (or `uv`, or `pixi add --pypi "spar[datadash]"`) installs it with a pre-built frontend and no Node.js required at install time.

## Run

From the repository root:

```bash
pixi run -e datadashboard spar-datadash-react-install
pixi run -e datadashboard spar-datadash-react
```

This starts:

- Flask bind address: `0.0.0.0:8060`
- React development-server bind address: `0.0.0.0:5173`

From the same host, open `http://localhost:8060` and `http://localhost:5173` in the browser.

## Build the frontend

```bash
pixi run -e datadashboard spar-datadash-react-build
```

The built frontend bundle is written into the Python package so it ships as package data and is resolved at runtime via `importlib.resources`:

`datadash/spar_datadash/_frontend`

## Clean

```bash
pixi run -e datadashboard spar-datadash-clean
```

## Core modules

- `spar_datadash/react_api.py`: API routes, render pipeline, interactive session handling.
- `spar_datadash/react_cli.py`: CLI wrapper for serving the React dashboard backend.
- `spar_datadash/utils.py`: environment/effect metadata, state serialization, image rendering.
- `spar_datadash/rich_logger.py`: structured terminal logging for dashboard services.
