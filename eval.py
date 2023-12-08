import argparse
from evaluation.rouge_evaluator import RougeEvaluator
import json
import tqdm
import chardet

args = argparse.ArgumentParser()
args.add_argument('--pred_data', type=str, default='data/validation.json')
args.add_argument('--eval_data', type=str, default='data/validation.json')
args = args.parse_args()

evaluator = RougeEvaluator()

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)

with open(args.pred_data, 'rb') as f:
    # adjusted function to account for different encodings (as Windows often uses a UTF-16 BOM by default that causes errors in opening the file natively)
    raw_data = f.read()
    if not raw_data:
        print("Empty)")
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f'Detected encoding: {encoding} with confidence: {confidence}')
    content = raw_data.decode(encoding)
    pred_data = json.loads(content)

assert len(eval_data) == len(pred_data)

pred_sums = []
eval_sums = []
for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
    pred_sums.append(pred['summary'])
    eval_sums.append(eval['summary'])

scores = evaluator.batch_score(pred_sums[:5], eval_sums[:5])

for k, v in scores.items():
    print(k)
    print("\tPrecision:\t", v["p"])
    print("\tRecall:\t\t", v["r"])
    print("\tF1:\t\t", v["f"])