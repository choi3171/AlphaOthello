# setup.py
from setuptools import setup, Extension
import pybind11

# 확장 모듈 정의
ext_modules = [
    Extension(
        "quoridor_engine",       # 파이썬에서 import할 이름
        ["binding.cpp"],        # 소스 파일 (quoridor_core.cpp는 include했으니 생략)
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name="quoridor_engine",
    version="0.0.1",
    ext_modules=ext_modules,
)