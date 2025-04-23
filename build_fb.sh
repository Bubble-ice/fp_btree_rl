#!/bin/bash

# 进入 fp_btree 目录
cd fp_btree || { echo "Error: Failed to enter fp_btree directory"; exit 1; }

# 编译 C++ 扩展模块
uv run setup.py build_ext --inplace || { echo "Error: Build failed"; exit 1; }

uv pip install -e . --no-build-isolation

# 返回上级目录
cd ..

echo "Build completed successfully!"