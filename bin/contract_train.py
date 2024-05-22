import warnings

import anndata
import torch
import numpy as np
import random
import logging
import os
import scanpy as sc
import scanpy.external as sce
import seaborn as sns
from numba import NumbaPendingDeprecationWarning
from sklearn.metrics import roc_auc_score

from contract.loader import MyLoader
from contract.loss_util import ContrastiveLoss
from contract.my_model import PLEContract
from tqdm import tqdm
from torch import nn

torch.manual_seed(2024)
torch.random.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
np.random.seed(2024)
random.seed(2024)

logging.basicConfig(
    format="%(asctime)s - %(filename)s - %(name)s:   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


def train(args):
    atac_loader = MyLoader(mode="train", domain="ATAC", args=args)()
    rna_loader = MyLoader(mode="train", domain="RNA", args=args)()

    model = PLEContract(atac_cells=atac_loader.n_cells,
                        rna_cells=rna_loader.n_cells,
                        bottleneck_size=args.bottleneck_size).to(
        args.device
    )
    if os.path.exists("{}/{}.pt".format(args.outdir, args.model_name)):
        logging.info("the model is loaded: {}/{}.pt".format(args.outdir, args.model_name))
        model.load_state_dict(torch.load("{}/{}.pt".format(args.outdir, args.model_name),
                                         map_location=args.device))
    epochs = args.train_epoch
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                           patience=3,
                                                           verbose=True)
    bce_loss = nn.BCELoss().to(args.device)
    contrastive_loss = ContrastiveLoss(args=args).to(args.device)
    rna_iter = iter(rna_loader.learn_loader)
    for epoch in range(epochs):
        atac_bce_loss_epoch = rna_bce_loss_epoch = align_loss_epoch = 0.0
        roc_atac_epoch = roc_rna_epoch = 0.0
        model.train()
        ProgressBar = tqdm(atac_loader.learn_loader)
        for seq_atac, label_atac in ProgressBar:
            ProgressBar.set_description("Learning_Epoch %d" % epoch)
            seq_atac = seq_atac.to(torch.float32).to(args.device)
            label_atac = label_atac.to(torch.float32).to(args.device)
            try:
                seq_rna, label_rna = next(rna_iter)
            except StopIteration:
                rna_iter = iter(rna_loader.learn_loader)
                seq_rna, label_rna = next(rna_iter)

            seq_rna = seq_rna.to(torch.float32).to(args.device)
            label_rna = label_rna.to(torch.float32).to(args.device)

            pred_atac, pred_rna = model.forward(seq_atac, seq_rna)
            atac_loss_bce = bce_loss(pred_atac, label_atac)
            rna_loss_bce = bce_loss(pred_rna, label_rna)

            if epoch < args.align_epoch:
                contrast_loss = torch.tensor([0.0]).to(args.device)
            else:
                contrast_loss = contrastive_loss.forward(rna_feature=model.get_embedding()[1],
                                                         rna_label=rna_loader.cell_label,
                                                         atac_label=atac_loader.cell_label,
                                                         atac_feature=model.get_embedding()[0])
            # bce loss compute
            loss = (atac_loss_bce + rna_loss_bce + contrast_loss) / 3.0
            atac_bce_loss_epoch += atac_loss_bce.item()
            rna_bce_loss_epoch += rna_loss_bce.item()
            align_loss_epoch += contrast_loss.item()

            roc_atac_tmp = roc_auc_score(y_score=pred_atac.flatten().detach().cpu().numpy(),
                                         y_true=label_atac.flatten().detach().cpu().numpy())
            roc_atac_epoch += roc_atac_tmp
            roc_rna_tmp = roc_auc_score(y_score=pred_rna.flatten().detach().cpu().numpy(),
                                        y_true=label_rna.flatten().detach().cpu().numpy())
            roc_rna_epoch += roc_rna_tmp
            ProgressBar.set_postfix(atac_roc_auc=roc_atac_tmp,
                                    rna_roc_auc=roc_rna_tmp,
                                    loss=loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        # evaluation
        rna_iter_eval = iter(rna_loader.val_loader)
        with torch.no_grad():
            model.eval()
            roc_atac_epoch_val = roc_rna_epoch_val = loss_eval_epoch = 0
            atac_loss_epoch_val = rna_loss_epoch_val = align_loss_epoch_val = 0
            for seq_atac_val, label_atac_val in atac_loader.val_loader:
                seq_atac_val = seq_atac_val.to(torch.float32).to(args.device)
                label_atac_val = label_atac_val.to(torch.float32).to(args.device)
                try:
                    seq_rna_val, label_rna_val = next(rna_iter_eval)
                except StopIteration:
                    rna_iter_eval = iter(rna_loader.val_loader)
                    seq_rna_val, label_rna_val = next(rna_iter_eval)

                seq_rna_val = seq_rna_val.to(torch.float32).to(args.device)
                label_rna_val = label_rna_val.to(torch.float32).to(args.device)

                pred_atac_val, pred_rna_val = model.forward(seq_atac_val, seq_rna_val)

                # roc compute
                roc_atac_tmp = roc_auc_score(y_score=pred_atac_val.flatten().detach().cpu().numpy(),
                                             y_true=label_atac_val.flatten().detach().cpu().numpy())

                roc_atac_epoch_val += roc_atac_tmp
                roc_rna_tmp = roc_auc_score(y_score=pred_rna_val.flatten().detach().cpu().numpy(),
                                            y_true=label_rna_val.flatten().detach().cpu().numpy())

                roc_rna_epoch_val += roc_rna_tmp
                # loss compute
                atac_loss_bce = bce_loss(pred_atac_val, label_atac_val)
                rna_loss_bce = bce_loss(pred_rna_val, label_rna_val)
                contrast_loss_val = contrastive_loss.forward(
                    rna_feature=model.get_embedding()[1],
                    rna_label=rna_loader.cell_label,
                    atac_feature=model.get_embedding()[0],
                    atac_label=atac_loader.cell_label
                )
                atac_loss_epoch_val += atac_loss_bce.item()
                rna_loss_epoch_val += rna_loss_bce.item()
                align_loss_epoch_val += contrast_loss_val.item()

                # bce loss compute
                loss_val = (atac_loss_bce + rna_loss_bce + align_loss_epoch_val) / 3.0
                loss_eval_epoch += loss_val.item()

        scheduler.step(loss_eval_epoch)

        # train record -----------------------
        atac_bce_loss_epoch /= len(atac_loader.learn_loader)
        rna_bce_loss_epoch /= len(atac_loader.learn_loader)
        align_loss_epoch /= len(atac_loader.learn_loader)
        roc_rna_epoch /= len(atac_loader.learn_loader)
        roc_atac_epoch /= len(atac_loader.learn_loader)

        logging.info("__________________train________________________")
        logging.info("Train: Epoch [%d/%d] ATAC Loss: %.4f, RNA Loss: %.4f, ATAC ROC_AUC: %.4f, "
                     "RNA ROC_AUC: %.4f, align loss: %.4f",
                     epoch, epochs,
                     atac_bce_loss_epoch,
                     rna_bce_loss_epoch,
                     roc_atac_epoch,
                     roc_rna_epoch,
                     align_loss_epoch
                     )

        # eval record ---------------------
        roc_atac_epoch_val /= len(atac_loader.val_loader)
        roc_rna_epoch_val /= len(atac_loader.val_loader)
        rna_loss_epoch_val /= len(atac_loader.val_loader)
        atac_loss_epoch_val /= len(atac_loader.val_loader)
        align_loss_epoch_val /= len(atac_loader.val_loader)

        logging.info("__________________evaluation________________________")
        logging.info("Eval: Epoch [%d/%d] ATAC Loss: %.4f, RNA Loss: %.4f, ATAC ROC_AUC: %.4f, "
                     "RNA ROC_AUC: %.4f, align loss: %.4f",
                     epoch, epochs,
                     atac_loss_epoch_val,
                     rna_loss_epoch_val,
                     roc_atac_epoch_val,
                     roc_rna_epoch_val,
                     align_loss_epoch_val
                     )

        torch.save(model.state_dict(), "{}/{}.pt".format(args.outdir, args.model_name))

    return model


def integration(model, atac_data, rna_data):
    source_feature = model.get_embedding()[0].detach().cpu().numpy()
    target_feature = model.get_embedding()[1].detach().cpu().numpy()
    feature_vec = np.concatenate([source_feature, target_feature], axis=0)
    adata = anndata.AnnData(feature_vec)
    adata.obs["domain"] = np.concatenate(
        (atac_data.obs["domain"], rna_data.obs["domain"]), axis=0
    )
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, "domain", theta=0.0, verbose=False)
    feature_vec = adata.obsm["X_pca_harmony"]
    atac_data.obsm["Embedding"] = feature_vec[: len(atac_data.obs["domain"])]
    rna_data.obsm["Embedding"] = feature_vec[len(rna_data.obs["domain"]):]

    adata.obs["cell_type"] = np.concatenate(
        (atac_data.obs["cell_type"], rna_data.obs['cell_type']), axis=0
    )

    logging.info("Running UMAP visualization...")
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=["cell_type"],
        palette=sns.color_palette(
            "husl", np.unique(adata.obs["cell_type"].values).size
        ),
        save="_cellType.png",
        show=False,
    )
    sc.pl.umap(
        adata,
        color=["domain"],
        palette=sns.color_palette("hls", 2),
        save="_domain.png",
        show=False,
    )
    return adata, model


def impute(args, model):
    atac_loader = MyLoader(mode="all", domain="ATAC", args=args)()
    rna_loader = MyLoader(mode="all", domain="ATAC", args=args)()

    ProgressBar = tqdm(atac_loader.all_loader)
    rna_iter = iter(rna_loader.all_loader)

    atac_impute = None
    rna_impute = None
    # ground_label = []
    with torch.no_grad():
        model.eval()
        for seq_atac, _ in ProgressBar:
            ProgressBar.set_description("Inferring")
            seq_atac = seq_atac.to(torch.float32).to(args.device)
            try:
                seq_rna, _ = next(rna_iter)
            except StopIteration:
                rna_iter = iter(rna_loader.all_loader)
                seq_rna, _ = next(rna_iter)
            seq_rna = seq_rna.to(torch.float32).to(args.device)
            output = model(seq_atac, seq_rna)
            pred_atac, pred_rna = output
            if atac_impute is None:
                atac_impute = pred_atac.detach().cpu().numpy()
            else:
                atac_impute = np.concatenate([atac_impute, pred_atac.detach().cpu().numpy()], axis=0)

            if rna_impute is None:
                rna_impute = pred_rna.detach().cpu().numpy()
            else:
                rna_impute = np.concatenate([rna_impute, pred_rna.detach().cpu().numpy()], axis=0)

    return atac_impute, rna_impute
