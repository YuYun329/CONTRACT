import anndata
import numpy as np
import pandas as pd

if __name__ == '__main__':
    ad = anndata.read_h5ad('../processed/SHARE_seq/RNA/ad.h5ad')
    X = (np.array(ad.X.todense()) != 0) * 1  # binarize
    pd.DataFrame(X).T.to_csv("./scopen_share_rna/scopen_tagcnt.txt", sep="\t")
