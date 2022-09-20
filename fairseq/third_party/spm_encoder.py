import sentencepiece as spm
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input folder')
    parser.add_argument('--output', type=str, required=True, help='output folder')
    parser.add_argument('--tokenizer', type=str, required=True, help='path or name of the tokenizer')
    parser.add_argument('--max_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    files = Path(args.input).glob("*.txt")
    if args.end == -1:
        args.end = float("inf")
    for i, file in enumerate(files):
        file = str(file)
        if not (args.start <= i < args.end):
            continue
        print("Reading files:", file)
        fo = open(file, encoding="utf-8", errors="ignore")
        fw = open(args.output + "/" + file.split("/")[-1], "w", encoding="utf-8")
        line = fo.readline()
        while(line):
            toks = tokenizer.encode(line.strip(), out_type=str)
            if len(toks) > args.max_len:
                line = fo.readline()
                continue
            toks = " ".join(toks)
            fw.writelines([toks, "\n"])
            line = fo.readline()

        fo.close()
        fw.close()

if __name__ == "__main__":
  main()



