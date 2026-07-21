#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from juliacall import Main as jl

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)

jl.seval(f'import Pkg; Pkg.activate("{parent_directory}")')
jl.seval('Pkg.instantiate()')

julia_file_path = os.path.join(script_directory, "test.jl")
jl.include(julia_file_path)

python_variable = jl.final_result
print(f"Python picked up the result: {python_variable}")