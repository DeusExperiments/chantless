#!/bin/bash
cd /root/os
exec llama.cpp/build/bin/llama-server -m models/mimo9b-bf16.gguf --host 0.0.0.0 --port 8080 \
  -c 1572864 -np 6 -fa on --jinja -ngl 999 --alias mimo9b-bf16 --metrics > logs/server_bf16.log 2>&1
