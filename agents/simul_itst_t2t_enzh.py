# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import string

from fairseq import checkpoint_utils, tasks
#import sentencepiece as spm
import torch
import codecs
import apply_bpe
import sacremoses
from sacremoses import MosesTokenizer, MosesDetokenizer
from sacremoses import MosesPunctNormalizer

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import TextAgent
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


class SimulTransTextAgent(TextAgent):
    """
    Simultaneous Translation
    Text agent for chinaese
    """
    def __init__(self, args):

        # Whether use gpu
        self.gpu = getattr(args, "gpu", False)

        # Max len
        self.max_len = args.max_len

        # Load Model
        self.load_model_vocab(args)

        # build word splitter
        # self.build_word_splitter(args)

        self.eos = DEFAULT_EOS

        self.src = args.src
        self.tgt = args.tgt
        self.x = None
        self.outputs = None
        self.threshold = args.delta
        # src pre
        self.mpn = MosesPunctNormalizer(lang=self.src)
        self.src_tokenizer = MosesTokenizer(lang=self.src)
        self.src_bpe_code = args.src_bpe_code
        self.src_bpe = self.get_src_bpe_model()
        self.lower = args.lower
        self.aggressive_dash = args.aggressive_dash
        self.reset_task = args.reset_task
        self.output_file = open(args.stream, "w", encoding="utf-8")
        if self.lower:
            print("===== lowercase source")

    def get_src_bpe_model(self):
        code_file = self.src_bpe_code
        bpe_codes = codecs.open(code_file, encoding='utf-8')
        src_bpe = apply_bpe.BPE(bpe_codes)
        return src_bpe

    def rmbpe(self,line):
        return re.sub('(@@ )|(@@ ?$)', '', line)
        
    def initialize_states(self, states):
        states.incremental_states = dict()
        states.incremental_states["online"] = dict()

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin
        # print(task_args)
        # if task_args._name == 'translation_waitk':
        #     need_mask = True
        # if task_args._name != 'translation':
        #     task_args._name = "translation"
        #     task_args.task = "translation"
            # print(task_args.task)

        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None

        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()
        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary
        assert self.dict["tgt"].eos_word == DEFAULT_EOS

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--max-len", type=int, default=50,
                            help="Max length of translation")
        # parser.add_argument("--waitk", type=int, default=1024,
        #                     help="wait-k policy")
        parser.add_argument("--src", type=str, default='en',
                            help="source language")
        parser.add_argument("--tgt", type=str, default='de',
                            help="target language")
        parser.add_argument("--src_bpe_code", type=str, required=True)
        parser.add_argument("--lower", action='store_true', default=False)
        parser.add_argument("--aggressive_dash", action='store_true', default=False)
        parser.add_argument("--reset_task", action='store_true', default=False)
        parser.add_argument("--delta", type=float, required=True)
        parser.add_argument("--stream", type=str, required=True)
        
        # fmt: on
        return parser
    
    def segment_to_units(self, segment, states):
        # tok+bpe 输入经过bpe后直接返回
        # print("segment", segment)
        # return [segment]
        # src preprocess tok -> bpe
        # print("[segment]:\t", segment)
        # Split a full word (segment) into subwords (units)
        segment_norm = self.mpn.normalize(segment)
        # print("segment", segment)
        segment_norm = ' '.join(['A', segment_norm, 'B'])
        segment_tok = self.src_tokenizer.tokenize(segment_norm, return_str=True,
                                                  aggressive_dash_splits=self.aggressive_dash)
        segment_tok = segment_tok[1:-1].strip()
        print("segment", segment)
        if self.lower:
            # print("lower", self.lower)
            segment_tok = segment_tok.lower()
        segment_bpe = self.src_bpe.segment(segment_tok).strip()
        return segment_bpe.split()

    def units_to_segment(self, units_queue, states):
        # return units_queue.pop()
        tokens = units_queue.value
        if len(tokens) > self.max_len: # for special error, infinite generate sub-word
            return DEFAULT_EOS
        if "@@" in tokens[-1]: # return when token not complete
            return
        else:
            # print("[tokens]:\t", tokens)
            line = ' '.join(tokens)
            line_rmbpe = self.rmbpe(line)
            # print("tokens", line_rmbpe)
            if line_rmbpe == "</s>":
                self.output_file.write("\n")
            else:
                self.output_file.write(line_rmbpe)
            while len(units_queue.value) > 0:
                units_queue.pop()
        
        return line_rmbpe.split()


    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        # print("src: ", states.units.source.value)
        src_indices = [
            self.dict['src'].index(x)
            for x in states.units.source.value
        ] # token2ids

        if states.finish_read():
            # Append the eos index when the prediction is over
            src_indices += [self.dict["src"].eos_index]

        src_indices = self.to_device(
            torch.LongTensor(src_indices).unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([src_indices.size(1)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths)

        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION
        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [
                    self.dict['tgt'].index(x)
                    for x in states.units.target.value
                    if x is not None
                ]
            ).unsqueeze(0)
        )
        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states
        )
        self.x = x
        self.outputs = outputs
        info_transport = outputs["info_transport"]
        cum_recieved_info = info_transport[:, :, :, :-1].sum(dim=-1, keepdim=False)[
                :, :, -1
            ]
        # print(cum_recieved_info)
        if (
                (cum_recieved_info < self.threshold).sum() > 0
                and not states.finish_read()
		):
            return READ_ACTION
        else:
            return WRITE_ACTION

    def E_trans_to_C(self, string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
        return string.translate(table)

    def predict(self, states):
        # tgt_indices = self.to_device(
        #     torch.LongTensor(
        #         [self.model.decoder.dictionary.eos()]
        #         + [
        #             self.dict['tgt'].index(x)
        #             for x in states.units.target.value
        #             if x is not None
        #         ]
        #     ).unsqueeze(0)
        # )
        # x, outputs = self.model.decoder.forward(
        #     prev_output_tokens=tgt_indices,
        #     encoder_out=states.encoder_states
        # )
        #print(x.shape)
        #print(outputs)
        # Predict target token from decoder states
        decoder_states = self.x

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)[0, 0].item()

        if index != self.dict['tgt'].eos_index:
            token = self.dict['tgt'].string([index])
        else:
            if states.finish_read():
                token = self.dict['tgt'].eos_word
            else:
                return ''

        if self.max_len > 0: # max len limit
            # print(len(states.source), states.source)
            # print(len(states.target), states.target)
            if len(states.target) - len(states.source) > self.max_len:
                token = self.dict['tgt'].eos_word

        torch.cuda.empty_cache()

        # print("decoded token: ", token)

        # punctuation
        # if token in string.punctuation:
        #     token = self.E_trans_to_C(token)

        # period
        # if token == '。' and '.' not in states.units.source.value:
        #     token = ''

        # print("return decoded token: ", token)
        return token