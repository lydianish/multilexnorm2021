import argparse, sys
import torch
from config.params import Params
from colab_model import normalize_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="path to input file to normalize", type=str, required=True)
    parser.add_argument("--basedir", help="path to base directory", type=str, default="/content/multilexnorm2021")
    script_args = parser.parse_args()

    sys.path.append(script_args.basedir)
    #sys.path.append("/home/lnishimw/scratch/multilexnorm2021")
    #sys.path.append("/gpfswork/rech/rnh/udc54vm/multilexnorm2021")

    args = Params().load(["--config", "config/inference.yaml"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #path = "/gpfswork/rech/rnh/udc54vm/multilexnorm2021/2.1.0/pretrained/ufal/byt5-small-multilexnorm2021-en"
    #args.model.pretrained_lm = args.dataset.tokenizer = path

    normalize_file(script_args.input_file, args, device)