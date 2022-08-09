# 选择资源


#PBS -N test_lg
#PBS -l ngpus=1
#PBS -l mem=46gb
#PBS -l ncpus=8
#PBS -l walltime=12:00:00
#PBS -M han.tai@student.unsw.edu.au
#PBS -m ae
#PBS -j oe

#PBS -o /srv/scratch/z5102138/test_large_dataset/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py36


cd /srv/scratch/z5102138/test_large_dataset
which python



mkdir -p ./test_yeast5/result_MLP_both2_1000_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=normal_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_1500_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=normal_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_200_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=bm_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=bm_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_1000_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=bm_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_1500_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=bm_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_200_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=im_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_500 test_method=im_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_1000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1000 test_method=im_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_1500_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_1500 test_method=im_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_both2_200_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both2_200 test_method=both_num_10_half device_id=0



