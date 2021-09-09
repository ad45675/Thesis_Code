#!/usr/bin/python
# coding=utf-8
# 該文件用於提取訓練log，去除不可解析的log後使log文件格式化，生成新的log文件供可視化工具繪圖
import inspect
import os
import random
import sys


def extract_log(log_file, new_log_file, key_word):
    with open(log_file, 'r') as f:
        with open(new_log_file, 'w') as train_log:
            for line in f:
                # 去除多GPU的同步log；去除除零錯誤的log
                if ('Syncing' in line) or ('nan' in line):
                    continue
                if key_word in line:
                    train_log.write(line)
    f.close()
    train_log.close()


extract_log('trainRecord.log', 'train_log_loss.txt', 'images')
extract_log('trainRecord.log', 'train_log_iou.txt', 'IOU')