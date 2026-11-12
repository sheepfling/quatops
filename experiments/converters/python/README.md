# Shared Python capture environment

This Python 3.13 project locks the standalone Eigen, MuJoCo, OpenCV, OpenUSD,
PROJ and SpiceyPy capture dependencies. Install everything with:

```shell
uv sync --locked --project experiments/converters/python --all-extras --python 3.13.14
```

Use `--extra eigen`, `--extra mujoco`, `--extra opencv`, `--extra openusd`,
`--extra proj` or `--extra spiceypy` instead of `--all-extras` to install only
one target's optional dependencies. Each target's versioned README documents
its native capture and output paths. Game engines, Blender's embedded Python,
ROS, and JavaScript runtimes retain their own runtime requirements.
SciPy-based reference verifiers also have separate pinned requirements.
