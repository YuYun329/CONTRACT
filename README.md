[![Stars](https://img.shields.io/github/stars/YuYun329/CONTRACT?logo=GitHub&color=yellow)](https://github.com/YuYun329/CONTRACT/stargazers)
# [Research on deep learning-based integration of single-cell multi-omics data and inference of regulation networks]

![主图](/architecture/main.png)

## News
#### [2024-05-23] CONTRACT第一代版本构建完成

## 安装	

#### install from GitHub
install the latest develop version

    pip install git+https://github.com/YuYun329/contract.git

or git clone and install

    pip install git+https://github.com/YuYun329/contract.git
    cd scContract
    python setup.py install
    

## 快速开始

contract可以根据一下步骤完成执行


### 1. 数据预处理

     contract_preprocess.py [-h] [--ad_file AD_FILE]
                              [--input_fasta INPUT_FASTA]
                              [--out_path OUT_PATH][--batch_size BATCH_SIZE]
    

其中`--ad_file`表示传入的数据文件，`--input_fasta`表示基因坐标参考文件，`--out_path`表示输出的保存路径
### 2. 训练模型

    contract_main.py --data_folder "./processed/" --outdir "output"
    
    
`--data_folder`: 数据预处理步骤的输出路径，其下应有一个“ATAC”与一个“RNA”目录

`--outdir`: 最终输出，包括数据的整合，降噪以及训练的模型。保存在最终的integration.h5ad文件中。

    
    
    


