
python train_src.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/

python train_adv.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter020000.pth

python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume results/adv_test/model_iter040000.pth OUTPUT_DIR datasets/b_datasets/soft_labels DATASETS.TEST b_datasets_train