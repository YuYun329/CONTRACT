import anndata
import lda
import pandas as pd
import numpy as np

if __name__ == '__main__':
    adata = anndata.read_h5ad("../processed/10x-Multiome-Pbmc10k/ATAC_notsampled/ad.h5ad")
    L = lda.LDA(n_topics=adata.shape[1])
    L.fit(adata.X.toarray().astype(np.int32))
    doc_topic = L.doc_topic_
    print(doc_topic.shape)
    pd.DataFrame(doc_topic).to_csv("cisTopic/cisTopic_10x.csv", index=False)
