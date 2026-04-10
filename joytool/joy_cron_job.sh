#!/bin/bash
# chmod +x /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/joy_cron_job.sh
# crontab -e
#  0 */3 * * * /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/joy_cron_job.sh >> /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/joy_cron.log 2>&1
# crontab -l
# tail -f /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/cron_execution.log
# grep CRON /var/log/syslog
#手动触发 # 手动运行脚本，并将输出重定向到你想要的 log 位置
# /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/joy_cron_job.sh >> /home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405/joytool/joy_cron.log 2>&1
# 1. 定义环境和路径
PYTHON_BIN="/home/ubuntu/.pyenv/versions/Alpha/bin/python3"
PROJECT_DIR="/home/ubuntu/SERVER_select-coin-pro_v2.0.0_20260405"
LOG_FILE="$PROJECT_DIR/joytool/cron_execution.log"

# 2. 进入目录并开始顺序执行
cd $PROJECT_DIR

{
    echo "=========================================="
    echo "任务启动时间: $(date)"
    
    echo "[$(date '+%F %T')] Step 1: 正在执行 merge_preprocess_batches.py..."
    $PYTHON_BIN joytool/merge_preprocess_batches.py
    
    echo "[$(date '+%F %T')] Step 2: 正在执行 01单策略.py..."
    $PYTHON_BIN 01单策略.py
    
    echo "[$(date '+%F %T')] Step 3: 正在执行 forward_eval_symbol_joy.py..."
    $PYTHON_BIN joytool/forward_eval_symbol_joy.py
    
    echo "任务结束时间: $(date)"
} >> "$LOG_FILE" 2>&1