from setuptools import setup, Extension
import pybind11
import os

# 获取当前文件的绝对路径
cur_dir = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    "fp_btree",
    sources=[
        os.path.join(cur_dir, "cpp/src/api2py.cpp"),
        os.path.join(cur_dir, "cpp/src/btree.cpp"),
        os.path.join(cur_dir, "cpp/src/fplan.cpp"),
        os.path.join(cur_dir, "cpp/src/sa.cpp"),
        os.path.join(cur_dir, "cpp/bind_main.cpp"),  # 绑定文件放最后
    ],
    include_dirs=[os.path.join(cur_dir, "cpp/include"), pybind11.get_include()],
    language="c++",
)


setup(name="fp_btree", version="1.0.0", ext_modules=[ext])
