"""Run the required training sequence and save only genuine model outputs."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(script, *args):
    subprocess.run([sys.executable, script, *args], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-domain', action='store_true')
    args = parser.parse_args()
    if not args.skip_download:
        run('download_qwen.py')
    if not args.skip_domain:
        run('domain_train.py')
    for version, script, adapter in [
        (1, 'sft_lora.py', 'models/specialized_adapter'),
        (2, 'sft_lora_v2.py', 'models/specialized_adapter_v2'),
        (3, 'sft_lora_v3.py', 'models/specialized_adapter_v3'),
    ]:
        run(script)
        run('test_specialized.py', '--model-path', adapter,
            '--output', f'SFT_V{version}_Test.txt')

if __name__ == '__main__':
    main()
