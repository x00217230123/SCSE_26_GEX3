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
The prepared code is not a replacement for running training. No SFT results are
included or invented. RAG model inference also remains to be run with Qwen3-0.6B.

The original PDF states October 10, midnight. Confirm the course's exact timezone
and interpretation of midnight if it is not stated in the submission portal.
