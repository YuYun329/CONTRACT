from pandas import Series
from torch.utils.data import DataLoader


class LoaderPkg:
    def __init__(self, n_cells: int = 0,
                 cell_label: Series = None,
                 type_num: int = 0,
                 learn_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 all_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 batch_ids: Series = None):
        self.n_cells = n_cells
        self.cell_label = cell_label
        self.type_num = type_num
        self.learn_loader = learn_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.batch_ids = batch_ids
        self.all_loader = all_loader
