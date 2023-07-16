#!/bin/bash

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "至少需要一个参数：push 或 pull"
    exit 1
fi

# 根据第一个参数执行不同的操作
if [ $1 = "push" ]; then
    # 如果执行 push 操作，需要第二个参数作为 commit 信息
    if [ $# -lt 2 ]; then
        echo "push 操作需要第二个参数作为 commit 信息"
        exit 1
    fi

    # 执行 git 操作
    git add .
    git commit -m "$2"
    git push gitee main && git push github main

elif [ $1 = "pull" ]; then
    # 如果执行 pull 操作，从两个仓库拉取最新的代码
    git pull gitee main && git pull github main

else
    echo "无效的参数：$1. 参数应为 push 或 pull"
    exit 1
fi
