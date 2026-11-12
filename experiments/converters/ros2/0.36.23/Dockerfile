FROM ros:jazzy-ros-base@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617

COPY requirements-evidence.txt /tmp/requirements-evidence.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    nlohmann-json3-dev \
    python3-colcon-common-extensions \
    python3-venv \
    ros-jazzy-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/quatops-evidence \
    && /opt/quatops-evidence/bin/python -m pip install \
       --no-cache-dir --requirement /tmp/requirements-evidence.txt

WORKDIR /workspace
ENTRYPOINT ["bash", "-lc"]
