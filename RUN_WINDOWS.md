# Windows PowerShell execution

If your existing folder is a Git checkout, update it with `git pull`. If it was
created from a ZIP, download this repository and copy the changed Python files
and this guide into `C:\Users\HUANG\Desktop\SCSE\SCSE_26_GEX3`. Keep your existing
`models` directory. This repository does not contain downloaded model weights.

```powershell
cd C:\Users\HUANG\Desktop\SCSE\SCSE_26_GEX3
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install torch transformers datasets accelerate peft trl modelscope liger-kernel
uv pip install --upgrade datasets
python run_exercise.py
```

Turn off the VPN for the ModelScope download, as required by the assignment.
The runner trains the domain adapter once, then trains V1/V2/V3 independently
from that adapter, evaluating the corresponding model immediately after each run.
It creates `SFT_V1_Test.txt`, `SFT_V2_Test.txt`, `SFT_V3_Test.txt` from real inference.
Existing download/domain-training steps can be skipped only if already completed:
`python run_exercise.py --skip-download --skip-domain`.

V2/V3 now preserve separate prompt/completion columns: a plain `text` column with
`completion_only_loss=True` does not provide the intended completion boundary.
Training settings and data are otherwise retained. The test prompt remains the
original exercise prompt so versions are evaluated consistently.

## Submission

Submit only the three generated text files and your own RAG GitHub repository URL.
The three checked-in text files contain genuine Qwen3-0.6B model responses to
all 15 exercise challenge questions. `TRAINING_RESULTS.json` records the completed
epochs, steps, evaluation results, output hashes, and installed package versions.
All final SFT versions use the same completed domain adapter. The domain job was
resumed from its first-epoch checkpoint after the temporary environment was lost.

These files record actual model behavior, including incorrect answers; file
completeness does not imply that every answer is correct. The RAG repository
contains `RAG_Runtime_Test.json` with genuine Qwen3-0.6B inference checks for the
missing phone number and guest Wi-Fi questions. The original policy loader and
policy data are unchanged.

Use the package versions in `TRAINING_RESULTS.json` to reproduce this run.
Model weights and adapters are distributed separately and are not committed here.

The original PDF states October 10, midnight. Confirm the course's exact timezone
and interpretation of midnight if it is not stated in the submission portal.
