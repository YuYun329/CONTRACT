import time

import scale

if __name__ == '__main__':
    start_time = time.time()
    scale.SCALE_function(data_list="../processed/10x-Multiome-Pbmc10k/RNA/ad.h5ad",
                         outdir="./scale_result/scale_10x_rna/", max_iter=3000, seed=2024, batch_size=128,
                         impute=True)
    end_time = time.time()
    print(f"scale time :{end_time - start_time}")