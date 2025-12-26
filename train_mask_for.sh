#!/bin/bash

# 创建日志目录

# 参数列表（在这里修改取值）
BATCH_SIZES=(8 4)                  # 你想尝试的 batch_size
LEARNING_RATES=("1e-3" "5e-3" "1e-4")  # 学习率，支持科学计数法（字符串形式）
FC1_OUTS=(256 128)                    # 分类头第一层维度
FC2_OUTS=(128 64)                    # 分类头第二层维度
DROPOUT1S=(0.7 0.5 0.3)                   # dropout1
DROPOUT2S=(0.7 0.5 0.3)                   # dropout2

# 固定参数（可根据需要修改）
PYTHON="/home/jinfang/anaconda3/envs/my_project/bin/python"
SCRIPT="./train_mask_for.py"
EPOCHS=50
NUM_CLASSES=2
FREEZE_BACKBONE="--freeze-backbone"   # 如果不想冻结，改成空字符串 ""

# 网格搜索主循环
for BS in "${BATCH_SIZES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for FC1 in "${FC1_OUTS[@]}"; do
            for FC2 in "${FC2_OUTS[@]}"; do
                for D1 in "${DROPOUT1S[@]}"; do
                    for D2 in "${DROPOUT2S[@]}"; do

                        # 生成实验名称（包含关键参数，便于区分）
                        EXP_NAME="mask_bs${BS}_lr${LR}_fc1${FC1}_fc2${FC2}_d${D1}-${D2}_$(date +%m%d_%H%M%S)"

                        LOG_FILE="logs/log_${EXP_NAME}.txt"

                        echo "=================================================="
                        echo "启动实验: $EXP_NAME"
                        echo "日志文件: $LOG_FILE"
                        echo "命令: batch_size=$BS, lr=$LR, fc1_out=$FC1, fc2_out=$FC2, dropout1=$D1, dropout2=$D2"
                        echo "=================================================="

                        # 启动训练（这里用 nohup 后台运行，每个组合独立进程）
                        $PYTHON $SCRIPT \
                            --epochs $EPOCHS \
                            --batch-size $BS \
                            --lr $LR \
                            --freeze-backbone \
                            --num-classes $NUM_CLASSES \
                            --fc1_out $FC1 \
                            --fc2_out $FC2 \
                            --dropout1 $D1 \
                            --dropout2 $D2 


                        echo ""

                        # 可选：加个短暂暂停，避免同时启动太多进程占用资源
                        sleep 2

                    done
                done
            done
        done
    done
done

echo "所有实验组合已全部提交！"
echo "日志保存在 logs/ 目录下"
echo "查看所有运行中的训练：ps aux | grep train_mask_for.py"