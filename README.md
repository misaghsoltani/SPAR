# SPAR: Stable Planning through Aligned Representations in Model-Based Reinforcement Learning

[![Publication](https://img.shields.io/badge/publication-RLJ%202026-%234285F4?logo=googlescholar&logoColor=%23d0d0d0)](https://rlj.cs.umass.edu/2026/papers/Paper20.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10-3.14](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&label=package%20manager)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
![Static Badge](https://img.shields.io/badge/statically%20typed-mypy-039dfc)

This repository contains the official implementation of the paper [Stable Planning through Aligned Representations in Model-Based Reinforcement Learning](https://rlj.cs.umass.edu/2026/papers/Paper20.html), accepted to **The third Reinforcement Learning Conference (RLC 2026)**.

SPAR trains a discrete world model and a goal-conditioned heuristic from clean observations. It then trains an alignment model that maps transformed observations into the clean discrete latent representation. Then, the alignment is used, alogn with the fixed world model and heuristic model, for planning

## Contents

  - [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Running the CLI](#running-the-cli)
    - [Reproducing Paper's Results](#reproducing-papers-results)
  - [Running Stages](#running-stages)
  - [Environments](#environments)
  - [Configuration](#configuration)
  - [Development](#development)
  - [Citation](#citation)
  - [License](#license)
  - [Contact](#contact)

## Quick Start

### Installation

SPAR uses [Pixi](https://pixi.sh) for package and environment management.

```bash
git clone https://github.com/misaghsoltani/SPAR_Release.git
cd SPAR_Release
pixi install -e dev
pixi run -e dev python -c "import spar"
pixi run -e dev spar --help
pixi run -e dev spar-experiments env=cube3
```

The `dev` environment includes the CLI and the repository quality tools. If the environment is active with `pixi shell -e dev`, run the same commands without the `pixi run -e dev` prefix.

### Running the CLI

SPAR accepts Hydra-style configuration overrides. A direct stage command selects an environment and a stage:

```bash
pixi run -e dev spar env=cube3 stage=gen_data data.num_cpus=4
```

An experiment preset selects the environment, stage, and related configuration together:

```bash
pixi run -e dev spar +experiment=cube3/train_alignment_disc
```

Append `--cfg job --resolve` to inspect a resolved configuration without running the stage:

```bash
pixi run -e dev spar +experiment=cube3/train_alignment_disc --cfg job --resolve
```

### Reproducing Paper's Results

The following Cube3 sequence follows the data and model dependencies used by the paper. The experiment presets are stored in [`spar/configs/experiment`](spar/configs/experiment).

1. Generate the clean offline datasets.

   ```bash
   pixi run -e dev spar +experiment=cube3/gen_offline
   ```

2. Train the discrete world model.

   ```bash
   pixi run -e dev spar +experiment=cube3/train_env_disc
   ```

3. Train the goal-conditioned heuristic using the clean world-model data.

   ```bash
   pixi run -e dev spar env=cube3 stage=train_heuristic
   ```

4. Generate the transformed offline datasets.

   ```bash
   pixi run -e dev spar +experiment=cube3/gen_offline_sim2real
   ```

5. Train the discrete alignment model.

   ```bash
   pixi run -e dev spar +experiment=cube3/train_alignment_disc
   ```

6. Test the discrete world model and alignment model.

   ```bash
   pixi run -e dev spar +experiment=cube3/test_model_disc
   ```

7. Generate start and goal pairs for search.

   ```bash
   pixi run -e dev spar env=cube3 stage=gen_search_data
   ```

8. Run one of the search stages after setting its model and pair-data paths.

   ```bash
   pixi run -e dev spar env=cube3 stage=search_qstar
   ```

The commands above start their stages. To inspect any step first, append `--cfg job --resolve`. Search stages accept dotted overrides such as `search.pairs_file=...`, `search.heuristic_model_path=...`, `search.env_model_path=...`, and `search.alignment_model_path=...`.

## Running Stages

The direct stage form is useful when changing paths, dataset sizes, or training settings:

```bash
# Generate data with a custom worker count.
pixi run -e dev spar env=cube3 stage=gen_data data.num_cpus=4

# Train the world model from the selected environment configuration.
pixi run -e dev spar env=cube3 stage=train_env_disc

# Train the heuristic.
pixi run -e dev spar env=cube3 stage=train_heuristic

# Run the three search algorithms.
pixi run -e dev spar env=cube3 stage=search_qstar
pixi run -e dev spar env=cube3 stage=search_gbfs
pixi run -e dev spar env=cube3 stage=search_ucs
```

Use `spar-experiments env=<name>` to list the experiment presets available for an environment. Use `spar --help` to list stage groups and common Hydra flags.

## Environments

Environment configurations are stored in [`spar/configs/env`](spar/configs/env). The available environment names are:

- `cube3` for Rubik's Cube 3x3x3
- `sokoban`
- `iceslider`
- `digitjump`

Select an environment with `env=<name>`. Environment implementations are in [`spar/environments`](spar/environments).

## Configuration

- [`spar/configs/experiment`](spar/configs/experiment): environment-specific experiment presets.
- [`spar/configs/stage`](spar/configs/stage): direct stage configurations.
- [`spar/configs/env`](spar/configs/env): environment definitions.
- [`spar/configs/search`](spar/configs/search): shared search settings.
- [`spar/search`](spar/search): Q*, GBFS, and UCS implementations.

The two main command forms are:

```bash
spar env=<environment> stage=<stage> [KEY=VALUE ...]
spar +experiment=<environment>/<experiment> [KEY=VALUE ...]
```

Nested values use dotted keys, for example `train.lr=3e-4` or `search.max_search_itrs=1000`.

## Development

Run the repository checks from the Pixi development environment:

```bash
pixi run -e dev fmt spar
pixi run -e dev pyrefly check spar
pixi run -e dev ty check spar
```

## Citation

If you use SPAR in your research, please cite:

```bibtex
@article{soltani2026stable,
    title={Stable Planning through Aligned Representations in Model-Based Reinforcement Learning},
    author={Misagh Soltani and Forest Agostinelli},
    journal={Reinforcement Learning Journal},
    volume={7},
    pages={},
    year={2026}
}
```

## License

SPAR is released under the [MIT License](LICENSE).

## Contact

For questions about the repository, please contact Misagh Soltani at [msoltani@email.sc.edu](mailto:msoltani@email.sc.edu).
