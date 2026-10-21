# exp_roadpp

This folder contains the ROAD-Waymo pipeline used by `src/exp_roadpp/main.py`.
It expects the ROAD++ / ROAD-Waymo dataset in a local folder named `roadpp` and
runs the pipeline in four steps:

1. Step 01: object, mask, depth, and flow preparation
2. Step 02: 3D pose / trajectory estimation
3. Step 03: symbolic fact extraction
4. Step 04: causal reasoning

## Dataset download

Download the ROAD-Waymo data from the ROAD++ / Waymo Open Dataset community
contribution page:

- ROAD++ challenge page: https://sites.google.com/view/road-plus-plus/home
- Waymo Open Dataset download page: https://console.cloud.google.com/storage/browser/waymo_open_dataset_road_plus_plus?pli=1

The repository includes the upstream ROAD++ notes in
`src/exp_roadpp/external_roadpp_repo/README.md`.

## Where to put the data

Place the dataset under `src/exp_roadpp/data/roadpp/` so the layout matches the
paths used by `src/exp_roadpp/config.py`.

Expected structure:

```text
src/exp_roadpp/data/roadpp/
├── road_waymo_trainval_v1.0.json
├── videos/
│   ├── Train_00000.mp4
│   ├── Train_00001.mp4
│   └── ...
├── frames/
├── depth_maps/
└── flows/
```

If you already keep the dataset somewhere else, create a symlink named `roadpp`
that points to the real data directory.

On machines that use the `ml-pulsar` configuration, the code currently expects
the dataset at `/home/sha/mnt/remote/dgx-g/storage-01/CauVid_Data/roadpp`.
On the local `jst` configuration, it uses `src/exp_roadpp/data/roadpp/`.

## Setup

1. Use Python 3.10.
2. Create and activate your environment.
3. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

4. Make sure the ROAD-Waymo data folder exists at the path above.
5. If you plan to extract frames with OpenCV/FFmpeg, install `ffmpeg` on your
	machine as well.

## Run

Run the pipeline from the repository root:

```bash
python src/exp_roadpp/main.py --device cuda:0
```

Useful arguments:

- `--data_num`: limit how many videos are processed; default is `all`
- `--dataset`: keep this as `roadpp`
- `--exp`: experiment name used for outputs
- `--machine`: selects the dataset path and device defaults in `config.py`

The default debug configuration is stored in `src/exp_roadpp/exp_config/debug.json`.
For a quick local test, you can start with a small `data_num` value and verify
that `src/exp_roadpp/output/roadpp/` is populated.

## Run with Docker

The repository root includes a `Dockerfile` and `docker-compose.yml` that can
also be used to run this pipeline, instead of setting up a local Python
environment.

### Build the image

From the repository root:

```bash
docker compose build cauvid-dev
```

### Run the pipeline in a container

Mount your local ROAD-Waymo dataset folder to the path the container expects
(`/app/src/exp_roadpp/data/roadpp`), then run `main.py` the same way as above:

```bash
docker run --rm -it \
	--gpus all \
	-v "$(pwd)/src:/app/src" \
	-v "/path/to/your/roadpp:/app/src/exp_roadpp/data/roadpp" \
	-v "$(pwd)/configs:/app/configs" \
	-e PYTHONPATH=/app \
	-e MPLBACKEND=Agg \
	cauvid-pipeline \
	python -m src.exp_roadpp.main --device cuda:0
```

Notes:

- `--gpus all` requires the NVIDIA Container Toolkit on the host; drop this
  flag (and use `--device cpu`) to run on CPU only.
- Replace `/path/to/your/roadpp` with wherever you keep the dataset on the
  host machine; it gets mounted read/write at the path `config.py` expects.
- Any output written under `src/exp_roadpp/output/roadpp/` will appear in your
  local `src/` folder since it is bind-mounted.

### Interactive development container

Alternatively, use the `cauvid-dev` service from `docker-compose.yml` to get
an interactive shell with the source tree mounted, then run the pipeline
manually:

```bash
docker compose run --rm cauvid-dev
# inside the container:
python -m src.exp_roadpp.main --device cuda:0
```



### Discussion
1. MovTow (Moving Towards) → → IncatLft (Indicating Left)
"If a Car is moving towards the ego-vehicle/camera (MovTow), it will turn on its left blinker/turn indicator (IncatLft) in the next frame."