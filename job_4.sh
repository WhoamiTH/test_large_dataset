#!/bin/bash
set -e


mkdir -p ./test_yeast5/result_MLP_both2_200_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_500_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1000_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1500_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_200_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1000_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_200_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_200_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_500_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both2_1500_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_200_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_200 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_200 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_500_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_500 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1000_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1000 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1000 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1500_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1500 test_method=normal_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1500 test_method=normal_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_200_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_200 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_200 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_500 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1000_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1000 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1000 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1500 test_method=bm_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1500 test_method=bm_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_200_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_200 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_200 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_500 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1000 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1000 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1500 test_method=im_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1500 test_method=im_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_200_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_200 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_200 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_500_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_500 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1000 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1000 test_method=both_num_10_half .device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_1500_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_1500 test_method=both_num_10_half .device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_1500 test_method=both_num_10_half .device_id=1



