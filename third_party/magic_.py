import time

import anndata
import magic
import pandas as pd

if __name__ == '__main__':
    start_time = time.time()
    data = anndata.read_h5ad("../processed/10x-Multiome-Pbmc10k/ATAC/ad.h5ad")
    magic_op = magic.MAGIC()
    X_magic = magic_op.fit_transform(data).X
    X_magic = pd.DataFrame(X_magic)
    X_magic.to_csv('magic_result/tmp.csv')
    end_time = time.time()
    print(f"magic time:{end_time - start_time}")
