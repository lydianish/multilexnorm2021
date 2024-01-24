import os
from utility.twokenize import tokenizeRawTweetText
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration

import warnings
warnings.filterwarnings('ignore')

from config.params import Params
from data.inference_data import CollateFunctor
from data.abstract_data import AbstractData
from torch.utils.data import DataLoader
from utility.output_assembler import OutputAssembler
from data.dataset.utility import Indexer
from data.dataset.abstract_dataset import AbstractDataset

class InferenceDataset(AbstractDataset):
    def __init__(self, inputs):
        self.inputs = inputs

        valid_indices = [[i for i, word in enumerate(sentence)] for sentence in inputs]
        self.indexer = Indexer(valid_indices)
        self.sorted_indices = sorted(list(range(len(self))), key=lambda x: (len(self.inputs[self.indexer.get_indices(x)[0]]), len(self.inputs[self.indexer.get_indices(x)[0]][self.indexer.get_indices(x)[1]])))

    def __getitem__(self, index):
        index = self.sorted_indices[index]
        sentence_index, word_index = self.indexer.get_indices(index)

        raw = self.inputs[sentence_index]

        # This is a hack for compatibility with new transformers
        raw = ' '.join(raw[:word_index]) + "<extra_id_124>" + raw[word_index] + "<extra_id_123>" + ' '.join(raw[word_index+1:])
        #
        # original:
        #raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
        #raw = ' '.join(raw)

        return raw, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)


class Data(AbstractData):
    def __init__(self, inputs, args):
        super().__init__(args)
        self.dataset = InferenceDataset(inputs)
        collate_fn = CollateFunctor(self.tokenizer)

        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=collate_fn
        )


class Model(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = self.dataset.tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(args.model.pretrained_lm)
        self.model.eval()

    def generate(self, batch):
        n_beams = self.args.model.n_beams
        sentence_ids, word_ids = batch["sentence_ids"], batch["word_ids"]

        outputs = self.model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            num_beams=n_beams, num_return_sequences=n_beams,
            repetition_penalty=1.0, length_penalty=1.0, max_length=512,
            output_scores=True, return_dict_in_generate=True
        )

        if n_beams > 1:
            scores = outputs.sequences_scores.cpu()
            scores = [scores[i*n_beams:(i+1)*n_beams] for i in range(len(sentence_ids))]
        else:
            scores = [[0.0] for i in range(len(sentence_ids))]

        outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        outputs = [outputs[i*n_beams:(i+1)*n_beams] for i in range(len(sentence_ids))]

        out_dict = {
            "predictions": outputs,
            "scores": scores,
            "sentence_ids": sentence_ids,
            "word_ids": word_ids,
        }
        return out_dict

def normalize_file(input_file, args, device):
  aligned_input_file = os.path.join(os.path.dirname(input_file), "aligned." + os.path.basename(input_file))
  aligned_output_file = os.path.join(os.path.dirname(input_file), "ufal-aligned." + os.path.basename(input_file))
  with open(input_file) as f:
    for line in f:
      tokens = [tokenizeRawTweetText(line.strip())]
      data = Data(tokens, args)
      assembler = OutputAssembler("outputs", args, data.dataset)
      model = Model(args, data).to(device)
      for i, batch in enumerate(data.dataloader):
          batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
          output = model.generate(batch)
          assembler.step(output)
      assembler.flush()
      with open("outputs/outputs.txt") as f1, open(aligned_input_file, 'a') as f2, open(aligned_output_file, 'a') as f3:
          for line in f1:
            if line.strip():
              split_line = line.split('\t')
              f2.write(split_line[0].strip())
              f3.write(split_line[1].strip())
            f2.write('\n')
            f3.write('\n')