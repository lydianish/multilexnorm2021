from utility.twokenize import tokenizeRawTweetText
import torch
import warnings
import math
warnings.filterwarnings('ignore')

import sys
#sys.path.append("/home/lnishimw/scratch/multilexnorm2021")
sys.path.append("/gpfswork/rech/rnh/udc54vm/multilexnorm2021")


from config.params import Params
from data.dataset.inference import InferenceDataset
from data.inference_data import CollateFunctor
from data.abstract_data import AbstractData
from torch.utils.data import DataLoader
from model.model import Model
from config.params import Params
from utility.output_assembler import OutputAssembler
from pytorch_lightning.utilities.apply_func import move_data_to_device


class Data(AbstractData):
    def __init__(self, inputs, args):
        super().__init__(args)
        self.dataset = InferenceDataset(inputs)
        collate_fn = CollateFunctor(self.tokenizer)

        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=collate_fn
        )


args = Params().load(["--config", "config/inference.yaml"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sentences = [
    "fyrst sntnce",
    "scond one .",
    "and yet another one of them sentencesss"
]
#sentences = ["yo hv to let ppl decide wat dey wanna do"]

tokens = [tokenizeRawTweetText(sentence) for sentence in sentences]

data = Data(tokens, args)
assembler = OutputAssembler(".", args, data.dataset)
model = Model(args, data).to(device)
model.eval()

for i, batch in enumerate(data.dataloader):
    batch = move_data_to_device(batch, device)
    output = model.generate(batch)
    assembler.step(output)

assembler.flush()

with open("outputs.txt") as f:
    for line in f.readlines():
        print(line.strip())
