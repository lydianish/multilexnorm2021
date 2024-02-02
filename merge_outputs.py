import pandas as pd

src_file = "/home/lnishimw/scratch/datasets/rocsmt/test/aligned.raw.en.test"
pred_file = "/home/lnishimw/scratch/datasets/rocsmt/test/ufal.aligned.raw.en.test"
output_file = "/home/lnishimw/scratch/datasets/rocsmt/test/UFAL.raw.norm"
aligned_output_file = "/home/lnishimw/scratch/datasets/rocsmt/test/UFAL.aligned.norm.json"

pred_lines = [] 

with open(src_file, "r") as f1, open(pred_file, "r") as f2, open(output_file, "w") as f3:
    pred_line = []
    for line1, line2 in zip(f1, f2):
        if line1.strip() or line2.strip():
            pred_line.append(line2.strip())
        else:
            if not pred_line:
                pass
            else:
                f3.write(' '.join(pred_line) + "\n")
                pred_lines.append({"data": pred_line.copy()})
            pred_line.clear()

pd.DataFrame(pred_lines).to_json(aligned_output_file, orient='records')
