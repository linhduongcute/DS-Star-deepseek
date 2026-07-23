# Run DS-STAR on Kaggle

Before running, enable **Internet** in the notebook settings and add a Kaggle
Secret named `OPENROUTER_API_KEY`.

## Cell 1 — Clone and install

```python
!git clone <YOUR_REPOSITORY_URL> /kaggle/working/DS-Star
%cd /kaggle/working/DS-Star
!python -m pip install -q -e .
```

If the repository is already attached or uploaded as a Kaggle Dataset, copy it
to `/kaggle/working/DS-Star` instead; `/kaggle/input` is read-only.

## Cell 2 — Load the OpenRouter key securely

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["OPENROUTER_API_KEY"] = (
    UserSecretsClient().get_secret("OPENROUTER_API_KEY")
)
assert os.environ["OPENROUTER_API_KEY"], "Missing OPENROUTER_API_KEY"
```

## Cell 3 — Select DeepSeek V4 Flash

```python
from pathlib import Path
import yaml

config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
config.update({
    "run_id": None,  # create a unique run ID
    "model_name": "deepseek/deepseek-v4-flash",
    "interactive": False,
    "max_refinement_rounds": 3,
    "preserve_artifacts": True,
    "data_dir": "/kaggle/working/data",
    "runs_dir": "/kaggle/working/runs",
})
config_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print(config_path.read_text(encoding="utf-8"))
```

## Cell 4 — Copy or upload data

For a Kaggle Dataset, replace the source path below:

```python
from pathlib import Path
import shutil

data_dir = Path("/kaggle/working/data")
data_dir.mkdir(parents=True, exist_ok=True)

source = Path("/kaggle/input/YOUR_DATASET/YOUR_FILE.csv")
target = data_dir / source.name
shutil.copy2(source, target)
print(target)
```

For a file created in the notebook, save it directly under
`/kaggle/working/data`.

## Cell 5 — Run

```python
import subprocess
import sys

data_file = "/kaggle/working/data/YOUR_FILE.csv"
query = "YOUR ANALYSIS QUESTION"

subprocess.run(
    [
        sys.executable,
        "dsstar.py",
        "--config", "config.yaml",
        "--data-files", data_file,
        "--query", query,
    ],
    check=True,
)
```

## Cell 6 — Inspect and download results

```python
from pathlib import Path
from IPython.display import display, FileLink

results = sorted(
    Path("/kaggle/working/runs").glob("*/final_output/result.json"),
    key=lambda path: path.stat().st_mtime,
)
assert results, "No result was produced"
latest_result = results[-1]
print(latest_result.read_text(encoding="utf-8"))

archive = shutil.make_archive(
    "/kaggle/working/dsstar_run",
    "zip",
    latest_result.parents[1],
)
display(FileLink(archive))
```
