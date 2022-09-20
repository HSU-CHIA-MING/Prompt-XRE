import jieba
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,required=True)
parser.add_argument("--output", type=str,required=True)
args = parser.parse_args()

fr = open(args.input, encoding="utf-8")
fw = open(args.output, "w", encoding="utf-8")

line = fr.readline()
while line:
    seg = jieba.cut(line.strip())
    fw.writelines([" ".join(seg), "\n"])
    line = fr.readline()

fr.close()
fw.close()