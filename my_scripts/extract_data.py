import argparse


def act(input_path, output_path, sent_num): 
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path, 'w', encoding='utf-8') as tgt:
    # with open(input_path, 'r', encoding='utf-8', errors='ignore') as raw, \
    #     open(output_path, 'w', encoding='utf-8') as tgt:
        
        print("start preprocessing...")
        
        num=0
        while 1:
            try:
                raw_sent = raw.readline()
            except UnicodeDecodeError:
                print('error:', num)

            tgt.write(raw_sent)
            num += 1
            tgt.flush()
            if num >= sent_num:
                print("finish preprocessing. ^-^")
                break

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', type=str, help='raw file path')
    parser.add_argument('--output', type=str, help='modified file path')
    parser.add_argument('--sent_num', type=int, default=1000000, help='modified file path')
    args = parser.parse_args()

    act(args.input, args.output, args.sent_num)


# Moltissimo. Basterebbe controllare alla nascita il frenulo linguale, verificare la postura della lingua, controllare che il bambino respiri bene col naso, altrimenti non potr�|  mantenere la lingua sul palato. Il controllo della deglutizione è alla base di tutta l�~@~Yortodonzia funzionale moderna e il motivo è semplice: molto meglio cee rcare di addomesticare il �~@~\mostro�~@~] per convincerlo a lavorare per noi, che lottarci contro per anni con vari apparecchi.

