from utility.twokenize import tokenizeRawTweetText

lowercase = False

input_file = "/home/lnishimw/scratch/datasets/rocsmt/test/norm.en.test"
output_file = "/home/lnishimw/scratch/datasets/rocsmt/test/tok/test.cased.raw.ref"
aligned_output_file = "/home/lnishimw/scratch/datasets/rocsmt/test/tok/test.cased.aligned.ref"


with open(input_file, "r") as f1, open(output_file, "w") as f2, open(aligned_output_file, "w") as f3:
    for line in f1:
        tokens = tokenizeRawTweetText(line.strip()) 
        if lowercase:
            tokens = [ token.lower() for token in tokens ]
        f2.write(' '.join(tokens) + '\n')
        f3.write('\n'.join(tokens) + '\n\n')
