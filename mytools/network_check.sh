#!/bin/bash
echo "=== 网络状态诊断 ==="
echo "1. 基础连接:"
ping -c 2 baidu.com >/dev/null 2>&1 && echo "✓ 国内网络正常" || echo "✗ 国内网络异常"

echo "2. 国际连接:"
curl -I --max-time 5 https://google.com >/dev/null 2>&1 && echo "✓ 国际网络正常" || echo "✗ 国际网络异常"

echo "3. 代理状态:"
curl -I --proxy http://127.0.0.1:7890 --max-time 5 https://google.com >/dev/null 2>&1 && echo "✓ 代理工作正常" || echo "✗ 代理异常"

echo "4. 环境变量:"
echo "   http_proxy: ${http_proxy:-未设置}"
echo "   HF_ENDPOINT: ${HF_ENDPOINT:-未设置}"
