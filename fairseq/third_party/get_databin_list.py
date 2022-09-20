import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    res = []
    for i in range(args.start, args.end+1):
        res.append(args.data_path + "/" + "databin." + str(i))
    res = ":".join(res)
    print(res)
if __name__ == "__main__":
    main()