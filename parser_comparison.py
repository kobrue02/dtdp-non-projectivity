from eval import load_conllu_file, evaluate, build_evaluation_table
from pyconll.unit.conll import Conll
from pyconll.unit.sentence import Sentence
from pyconll.util import find_nonprojective_deps
from spacy_conll import init_parser
from typing import Any

from utils import plot, eval_string_to_pandas

import argparse
import glob
import pyconll.unit
import pyconll.unit.sentence
import pyconll.util
import pyconll.exception
import pyconll
import pandas as pd
import spacy_udpipe
import seaborn as sns
import matplotlib.pyplot as plt
import os

UD_ROOT_PATH = "/Users/birdy/Downloads/Universal Dependencies 2.16/ud-treebanks-v2.16"

class Parser:

    def __init__(self, model_name = "spacy", language = "en") -> None:
        self.model_name = model_name
        self.language = language
        self.model = self._load_parser()
    
    def _load_parser(self):
        spacy_model_names = {
            "en": "en_core_web_lg",
            "de": "de_core_news_lg"
        }
        match self.model_name:
            case "spacy":
                nlp = init_parser(
                    spacy_model_names[self.language],
                    "spacy",
                    ext_names={"conll_pd": "pandas"},
                    include_headers=True
                )
            case "stanza":
                nlp = init_parser(
                    self.language,
                    "stanza",
                    parser_opts={"use_gpu": True, "verbose": False},
                    include_headers=True
                )
            case "udpipe":
                spacy_udpipe.download(self.language)
                # spacy_udpipe.download("en") # download English model
                nlp = init_parser(
                    self.language,
                    "udpipe"
                )
            case _:
                raise NotImplementedError()
        return nlp
    
    def parse(self, text: str):
        if self.model_name in ["spacy", "udpipe", "stanza"]:
            doc = self.model(text)
            return doc._.conll_str
        
    def __call__(self, *args: Any, **kwds: Any):
        if len(args) == 1:
            return self.parse(args[0])
        else: raise NotImplementedError()


class Evaluate:

    def __init__(self, fp, parser: Parser) -> None:
        self.test_file = fp
        self.test_file_nonproj = f"{self.test_file.split(".connlu")[0]}_nonprojective.conllu"
        self.test_file_proj = f"{self.test_file.split(".connlu")[0]}_projective.conllu"
        self.docs = self._load_conllu_file(fp)
        self.docs_projective = [doc for doc in self.docs if self._is_projective(doc)]
        self.docs_nonprojective = [doc for doc in self.docs if not self._is_projective(doc)]
        self.parser = parser
    
    def _is_projective(self, sentence: pyconll.unit.sentence.Sentence) -> bool:
        try:
            proj = find_nonprojective_deps(sentence)
            return len(proj) > 0
        except KeyError:
            return False
    
    def _load_conllu_file(self, fp):
        data = pyconll.load_from_file(fp)
        sents = [sent for sent in data]
        return sents

    def _parse_sentences(self, docs) -> list[Conll]:
        parser_results = []
        for ground_truth in docs:
            parsed = self.parser(ground_truth.text)
            try:
                parsed_conllu = pyconll.load.load_from_string(parsed)
            except pyconll.exception.ParseError as e:
                print(e)
                print(parsed)
                continue
            parser_results.append(parsed_conllu)
        return parser_results
    
    def _export_connlu(self, parsed, fname):
        with open(fname, "w", encoding="utf-8") as f:
            for p in parsed:
                if isinstance(p, Conll):
                    p.write(f)
                elif isinstance(p, Sentence):
                    Conll([p.conll()]).write(f)

    def _split_test_file(self):
        self._export_connlu(self.docs_nonprojective, self.test_file_nonproj)
        self._export_connlu(self.docs_projective, self.test_file_proj)
    
    def run_ud_eval(self):
        self._split_test_file()
        # nonprojective
        def nonproj():
            parsed = self._parse_sentences(self.docs_nonprojective)
            self._export_connlu(parsed, "parser_outputs_nonprojective.conllu")
            truth = load_conllu_file(self.test_file_nonproj)
            system = load_conllu_file("parser_outputs_nonprojective.conllu")
            ud_eval = evaluate(truth, system)
            table_str = build_evaluation_table(ud_eval, verbose=True)
            return eval_string_to_pandas(table_str)
        
        def proj():
            parsed = self._parse_sentences(self.docs_projective)
            self._export_connlu(parsed, "parser_outputs_projective.conllu")
            truth = load_conllu_file(self.test_file_proj)
            system = load_conllu_file("parser_outputs_projective.conllu")
            ud_eval = evaluate(truth, system)
            table_str = build_evaluation_table(ud_eval, verbose=True)
            return eval_string_to_pandas(table_str)
        
        np_df = nonproj()
        p_df = proj()
        np_df["Projective"] = False
        p_df["Projective"] = True
        return pd.concat([np_df, p_df], ignore_index=True)



def _eval_file_conllu(file, lang, model):
    try:
        parser = Parser(model, lang)
    except Exception as e:
        print("malformed conllu: ", str(e))
        return None
    try:
        test_eval = Evaluate(file, parser)
    except Exception as e:
        print("can't evaluate conllu: ", str(e))
        return None
    _result = test_eval.run_ud_eval()
    _result["Model"] = model
    return _result, len(test_eval.docs_projective), len(test_eval.docs_nonprojective)


def _eval_lang_dirs(file, lang, model):
    root_dir = UD_ROOT_PATH
    # get all sub directories that have the language name in the name
    sub_dirs = [d for d in os.listdir(root_dir) if file.lower() in d.lower()]
    result_dfs = []
    n_proj = 0
    n_nonproj = 0
    for sub_dir in sub_dirs:
        # get the test file (which ends with test.conllu)
        file = glob.glob(os.path.join(root_dir, sub_dir, '*test.conllu'))[0]
        print("got " + str(file))
        try:
            _df, n_proj_sub, n_nonproj_sub = _eval_file_conllu(file, lang, model)
        except Exception as e:
            continue
        result_dfs.append(_df)
        n_proj += n_proj_sub
        n_nonproj += n_nonproj_sub
    return pd.concat(result_dfs, ignore_index=True), n_proj, n_nonproj


def eval_file(file, lang, model):
    if file.endswith(".conllu"):
        return _eval_file_conllu(file, lang, model)
    else:
        # it is a language name
        return _eval_lang_dirs(file, lang, model)


def main(file, lang, model):
    result_dfs = []
    models = [model]
    if model == "all":
        models = ["spacy", "stanza", "udpipe"]
    for model in models:
        print("evaluating " + model)
        _df, n_proj, n_nonproj = eval_file(file, lang, model)
        result_dfs.append(_df)
    result = pd.concat(result_dfs, ignore_index=True)
    print(result)
    print(f"Projective sentences: {n_proj}")
    print(f"Nonprojective sentences: {n_nonproj}")
    plot(result)
    result.to_csv(f"result_{lang}.csv", index=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the test file")
    parser.add_argument("--lang", type=str, required=True, help="Language of the test file")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.file, args.lang, args.model)