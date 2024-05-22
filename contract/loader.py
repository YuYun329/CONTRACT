import gc
import os.path
import random

import anndata
import h5py
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset, DataLoader, Subset

from contract.loader_packge import LoaderPkg


class MyDataset(Dataset):
    def __init__(self, file, m):
        self.file = file  # h5 file for sequence
        self.m = m  # csr matrix, rows as seqs, cols are cells
        self.n_cells = m.shape[1]
        self.ones = np.ones(1344)
        self.rows = np.arange(1344)

        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            self.length = X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        m = self.m
        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            x = X[idx]
            x_tf = sparse.coo_matrix((self.ones, (self.rows, x)),
                                     shape=(1344, 5),
                                     dtype='int8').toarray()
            # y_tf = m.getrow(idx).toarray()
            y = m.indices[m.indptr[idx]:m.indptr[idx + 1]]
            y_tf = np.zeros(self.n_cells, dtype='float32')
            y_tf[y] = 1

            x_tf = torch.tensor(x_tf, dtype=torch.int8)
            y_tf = torch.tensor(y_tf, dtype=torch.float32)
            return x_tf, y_tf


class MyLoader:
    def __init__(self,
                 args,
                 mode="train",
                 domain="ATAC",
                 ):

        if domain not in ['ATAC', 'RNA']:
            raise ValueError("The domain must be in '[ATAC, RNA]'")
        if mode not in ['train', 'test', 'all']:
            raise ValueError("The mode must be in ['train', 'test']")
        self.args = args
        self.domain = domain
        self.debug = args.debug
        self.mode = mode
        self.bs = args.batch_size
        self.data_folder = args.data_folder
        self.preprocess_folder = os.path.join(self.data_folder, domain)
        self.ad = anndata.read_h5ad("%s/ad.h5ad" % self.preprocess_folder)
        self.n_cells = self.ad.n_obs
        self.type_num = len(np.unique(self.ad.obs['cell_type']))
        self.cell_label = self._cell_label()
        self.batch_ids = self.ad.obs['batch'] if args.batch_correlation else None

    def _cell_label(self):
        cell_label = self.ad.obs["cell_type"]
        cell_label_int = cell_label.rank(method="dense", ascending=True).astype(int) - 1
        return cell_label_int

    def _train_load(self):
        # define the datasets
        preprocess_folder = self.preprocess_folder
        train_data = '%s/train_seqs.h5' % preprocess_folder
        val_data = '%s/val_seqs.h5' % preprocess_folder
        ad = self.ad
        split_file = '%s/splits.h5' % preprocess_folder

        # convert to csr matrix
        with h5py.File(split_file, 'r') as hf:
            train_ids = hf['train_ids'][:]
            val_ids = hf['val_ids'][:]
        if self.args.positive_label_sample_rate != 1:
            m = ad.X.tocoo().transpose()
            nonzero_cnt = len(m.data)
            num_zero_elements = int(nonzero_cnt * (1 - self.args.positive_label_sample_rate))
            random_indices = np.random.choice(nonzero_cnt, num_zero_elements, replace=False)
            m.data[random_indices] = 0
            m = m.tocsr()
        else:
            m = ad.X.tocoo().transpose().tocsr()

        del ad
        gc.collect()
        m_train = m[train_ids, :]
        m_val = m[val_ids, :]
        del m
        gc.collect()
        if self.debug:
            subset_indices = range(random.randint(self.bs, 100))
            dataset_class = lambda data, mask: Subset(MyDataset(data, mask), subset_indices)
        else:
            dataset_class = MyDataset

        learn_loader = DataLoader(dataset=dataset_class(train_data, m_train),
                                  batch_size=self.bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=dataset_class(val_data, m_val),
                                batch_size=self.bs, shuffle=True, drop_last=True)
        self.learn_loader = learn_loader
        self.val_loader = val_loader

    def _test_load(self):
        # define the datasets
        preprocess_folder = self.preprocess_folder
        test_data = '%s/test_seqs.h5' % preprocess_folder
        ad = self.ad
        split_file = '%s/splits.h5' % preprocess_folder

        # convert to csr matrix
        with h5py.File(split_file, 'r') as hf:
            test_ids = hf['test_ids'][:]

        # 归一化
        m = ad.X.tocoo().transpose().tocsr()
        del ad
        gc.collect()
        m_test = m[test_ids, :]
        del m
        gc.collect()

        test_loader = DataLoader(dataset=MyDataset(test_data, m_test),
                                 batch_size=self.bs,
                                 shuffle=True,
                                 drop_last=True)
        self.test_loader = test_loader

    def _all_load(self):
        # define the datasets
        preprocess_folder = self.preprocess_folder
        all_data = '%s/all_seqs.h5' % preprocess_folder
        ad = self.ad
        m = ad.X.tocoo().transpose().tocsr()
        all_loader = DataLoader(dataset=MyDataset(all_data, m),
                                batch_size=self.bs,
                                shuffle=True,
                                drop_last=False)
        self.all_loader = all_loader

    def __call__(self) -> LoaderPkg:
        """
        :return: n_cells, cell_label, type_num, dataloader
        """
        if self.mode == 'train':
            self._train_load()
            return LoaderPkg(n_cells=self.n_cells,
                             cell_label=self.cell_label,
                             type_num=self.type_num,
                             learn_loader=self.learn_loader,
                             val_loader=self.val_loader,
                             batch_ids=self.batch_ids,
                             )
        elif self.mode == 'test':
            self._test_load()
            return LoaderPkg(
                n_cells=self.n_cells,
                cell_label=self.cell_label,
                type_num=self.type_num,
                test_loader=self.test_loader,
                batch_ids=self.batch_ids,
            )
        else:
            self._all_load()
            return LoaderPkg(
                n_cells=self.n_cells,
                cell_label=self.cell_label,
                type_num=self.type_num,
                all_loader=self.all_loader,
                batch_ids=self.batch_ids,
            )
