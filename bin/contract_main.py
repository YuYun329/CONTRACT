import os.path

import anndata
import configargparse

from bin.contract_train import train, integration, impute


def make_parser():
    parser = configargparse.ArgParser(
        description="Preprocess anndata to generate inputs for contract (reference to scBasset).")
    parser.add_argument('--data_folder', '-d', type=str, default="./processed/SNARE_seq")
    parser.add_argument("--train_epoch", type=int, default=40)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bottleneck_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="contract")
    parser.add_argument("--info_nce_tau", type=float, default=0.75)
    parser.add_argument("--batch_correlation", type=bool, default=False)
    parser.add_argument("--align_epoch", type=int, default=38)
    parser.add_argument("--positive_label_sample_rate", type=float, default=1.0)
    parser.add_argument("--impute", type=bool, default=False)
    parser.add_argument("--outdir", type=str, default="./output")
    return parser


def main():
    args = make_parser().parse_args()
    model = train(args)

    ATAC_ad = anndata.read_h5ad(os.path.join(args.data_folder, "ATAC") + "/ad.h5ad")
    RNA_ad = anndata.read_h5ad(os.path.join(args.data_folder, "RNA") + "/ad.h5ad")

    adata_integrate, model = integration(args, model, ATAC_ad, RNA_ad)

    if args.impute:
        atac_impute, rna_impute = impute(args, model)
        adata_integrate.obsm['atac_impute'] = atac_impute
        adata_integrate.obsm['rna_impute'] = rna_impute
    adata_integrate.write_h5ad(os.path.join(args.our), "adata-integrate.h5ad")