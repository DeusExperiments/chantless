#!/bin/bash
cd /root/os/llama.cpp
rm -f ../models/mimo9b-bf16.gguf
.venv/bin/python convert_hf_to_gguf.py ../models/mimo9b --outtype bf16 --no-mtp --outfile ../models/mimo9b-bf16.gguf > ../logs/convert2.log 2>&1
echo CONVERT_EXIT=$? >> ../logs/convert2.log
.venv/bin/python /root/os/meta.py | grep -E "block_count|nextn|recurrent" >> ../logs/convert2.log
/root/os/serve_bf16.sh
