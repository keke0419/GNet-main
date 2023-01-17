import numpy as np
from Bio import SeqIO
import random
import os.path as osp
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="pre-process fr other characteristics.")
    parser.add_argument("-d", dest="dir", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')
    parser.add_argument("-c", dest="category", type=str, default='', help="One of the train, test and neg.")
    parser.add_argument("-s", dest="seed", type=int, default=666, help="Random seed to have reproducible results.")

    return parser.parse_args()



def processFastaFile(seq):
    phys_dic = {
        'A': [1, 1, 1],
        'T': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]}
    seqLength = len(seq)
    sequence_vector = np.zeros([1001, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 1001):
        sequence_vector[i, -1] = 1
    return sequence_vector


def nd(seq, seq_length):
    seq = seq.strip()
    nd_list = [None] * seq_length
    for j in range(seq_length):
        # print(seq[0:j])
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j + 1].count('A') / (j + 1), 3)
        elif seq[j] == 'T':
            nd_list[j] = round(seq[0:j + 1].count('T') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j + 1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j + 1].count('G') / (j + 1), 3)
    return np.array(nd_list)


def main():
    params = get_args()
    random.seed(params.seed)
    name = params.name
    category = params.category
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    seq_length = 1001

    data = []

    for index, record in enumerate(
            SeqIO.parse(open(data_dir + '/%s_%s.txt'% (name,category)), 'fasta')):
        if '>' not in record.seq:
            record.seq = record.seq.strip()
            probMatr = processFastaFile(record.seq)
            probMatr_ND = nd(record.seq, seq_length)
            probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))

            data.append(probMatr_NDCP.tolist())

    data = np.array(data)
    data = data.transpose(0, 2, 1)

    # np.savez('./K562/ETS1/data/ETS14M_neg.npz', data=dataX)
    np.savez(out_dir + '/%s4_%s.npz' % (name,category), data=data)
    # np.savez('./HeLa-S3/E2F1/data/E2F188_train.npz', data=dataX)
if __name__ == '__main__':
    main()
