"""Evaluate one trained adapter and save the exercise's actual test outputs."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='models/specialized_adapter')
    parser.add_argument('--output')
    args = parser.parse_args()
    model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    tests = [json.loads(line) for line in Path('data/sft_challenge_test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    results = []
    for test in tests:
        prompt = 'You are a university IT support assistant.\n\nUser: ' + test['prompt'] + '\n\nAssistant:'
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        block = '\n====================\nQUESTION:\n' + test['prompt'] + '\n\nEXPECTED POINTS:\n' + str(test['expected_points']) + '\n\nMODEL:\n' + response + '\n'
        print(block)
        results.append(block)
    # Write atomically only after every test completed successfully.
    if args.output:
        destination = Path(args.output)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(''.join(results), encoding='utf-8')
        temporary.replace(destination)

if __name__ == '__main__':
    main()
