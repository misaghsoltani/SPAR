# SPAR Data Dashboard frontend

This directory holds the React and TypeScript frontend for the SPAR Data
Dashboard. The Flask backend lives in the sibling `spar_datadash` package.

Use the dashboard to inspect registered environments, states, rendering
effects, and data files. Command-line training and search do not depend on it.

## Run from the repository root

Install the frontend dependencies and start the backend and development server:

```bash
pixi run -e datadashboard spar-datadash-react-install
pixi run -e datadashboard spar-datadash-react
```

The current launcher binds to:

- Flask API address `0.0.0.0:8060`
- Vite development-server address `0.0.0.0:5173`

From the same host, open `http://localhost:8060` and
`http://localhost:5173` in the browser.

Binding to `0.0.0.0` makes the service reachable from other hosts allowed by
the network. Run it only on a trusted network or change the binding in the
launcher configuration.

## Build the frontend

```bash
pixi run -e datadashboard spar-datadash-react-build
```

The build output is written into `../spar_datadash/_frontend/`, where it ships
as package data and is resolved at runtime via `importlib.resources`. The build
does not start the Flask service.

## Clean generated dashboard files

```bash
pixi run -e datadashboard spar-datadash-clean
```

## Source layout

```text
src/
  components/   interface components
  lib/          shared types and helpers
  assets/       frontend assets
```

Backend routes, environment discovery, state serialization, and rendering live
under `../spar_datadash/`. Keep frontend types synchronized with the backend
payloads when either side changes.

## Validation

Before committing a frontend change:

1. build with the frozen lockfile
2. run the relevant dashboard backend tests
3. verify environment discovery and one interactive session
4. inspect browser and backend logs for warnings
5. confirm that generated `dist/` files are handled according to repository
   policy

The parent dashboard guide is [../README.md](../README.md).
