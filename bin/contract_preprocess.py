import os

import configargparse
import h5py
from Bio import SeqIO
import numpy as np
import anndata
from scipy import sparse
import logging
from contract.utils import split_train_test_val, make_h5_sparse

logging.basicConfig(
    format="%(asctime)s - %(filename)s - %(name)s:   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def remove_not_genome(adata, fasta_file):
    logging.info("去除人工合成genome ...")
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
    fasta_ids = list(set(fasta_dict.keys()))

    chrom_series = adata.var['chrom']
    chr_index = [chrom_series[i] in fasta_ids for i in range(len(chrom_series))]
    chr_index = np.array(chr_index)
    adata_new = anndata.AnnData(X=adata.X[:, chr_index], obs=adata.obs, var=adata.var.loc[chr_index, :])

    return adata_new


def make_parser():
    parser = configargparse.ArgParser(
        description="Preprocess anndata to generate inputs for scBasset.")
    parser.add_argument('--ad_file', type=str, default="../datasets/Chen-2019/filetered-Chen-2019-ATAC.h5ad",
                        help='Input scATAC anndata. .var must have chrom, chromStart, chromEnd columns. anndata.X '
                             'must be in csr '
                             'format.')
    parser.add_argument('--input_fasta', type=str, default="../datasets/mm10.fa",
                        help='Genome fasta file.')
    parser.add_argument('--out_path', type=str, default='../processed/SNARE_seq/',
                        help='Output path. Default to ./processed/')
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--domain", type=str, default="ATAC")

    return parser


def main():

    parser = make_parser()
    args = parser.parse_args()
    logging.info(f"argument: {args.__dict__}")
    input_ad = args.ad_file
    input_fasta = args.input_fasta
    output_path = args.out_path

    ad = anndata.read_h5ad(input_ad)

    ad = remove_not_genome(ad, input_fasta)

    # sample cells
    os.makedirs(output_path, exist_ok=True)

    # save anndata
    ad.write('%s/ad.h5ad' % output_path)
    print('successful writing h5ad file.')

    # save peak bed file
    ad.var.loc[:, ['chrom', 'chromStart', 'chromEnd']].to_csv('%s/peaks.bed' % output_path, sep='\t', header=False,
                                                              index=False)
    print('successful writing bed file.')

    # save train, test, val splits
    train_ids, test_ids, val_ids = split_train_test_val(np.arange(ad.shape[1]), train_ratio=0.8)
    f = h5py.File('%s/splits.h5' % output_path, "w")
    f.create_dataset("train_ids", data=train_ids)
    f.create_dataset("test_ids", data=test_ids)
    f.create_dataset("val_ids", data=val_ids)
    f.close()
    print('successful writing split file.')

    # save labels (ad.X)
    m = ad.X.tocoo().transpose().tocsr()
    m_train = m[train_ids, :]
    m_val = m[val_ids, :]
    m_test = m[test_ids, :]
    sparse.save_npz('%s/m_train.npz' % output_path, m_train, compressed=False)
    sparse.save_npz('%s/m_val.npz' % output_path, m_val, compressed=False)
    sparse.save_npz('%s/m_test.npz' % output_path, m_test, compressed=False)
    print('successful writing sparse m.')

    # save sequence h5 file
    ad_train = ad[:, train_ids]
    ad_test = ad[:, test_ids]
    ad_val = ad[:, val_ids]
    make_h5_sparse(ad, '%s/all_seqs.h5' % output_path, input_fasta, domain=args.domain, batch_size=args.batch_size)
    make_h5_sparse(ad_train, '%s/train_seqs.h5' % output_path, input_fasta, domain=args.domain, batch_size=args.batch_size)
    make_h5_sparse(ad_test, '%s/test_seqs.h5' % output_path, input_fasta, domain=args.domain, batch_size=args.batch_size)
    make_h5_sparse(ad_val, '%s/val_seqs.h5' % output_path, input_fasta, domain=args.domain, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
