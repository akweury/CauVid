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



