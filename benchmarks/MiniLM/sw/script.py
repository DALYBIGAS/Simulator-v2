import subprocess

# 打印提示信息
print("------------> 将对MLIR文件进行通用优化......")

# 定义MLIR优化命令
mlir_opt_command = [
    "mlir-opt",
    "-linalg-fuse-elementwise-ops",
    "-canonicalize",
    "-convert-tensor-to-linalg",
    "-eliminate-empty-tensors",
    "-empty-tensor-to-alloc-tensor",
    "-linalg-bufferize",
    "-arith-bufferize",
    "-tensor-bufferize",
    "-func-bufferize",
    "-finalizing-bufferize",
    "-buffer-deallocation",
    "-buffer-results-to-out-params",
    "-canonicalize",
    "-cse",
    "02-linalg-on-tensors.mlir",
    "-o",
    "02-linalg-bufferized.mlir"
]

# 执行命令
try:
    subprocess.run(mlir_opt_command, check=True)
    print("------------> MLIR文件优化完成。\n")
except subprocess.CalledProcessError as e:
    print(f"------------> 执行MLIR优化命令时出错: {e}")

#----------------------- The Operator Counter -----------------------#

import re
from collections import Counter

# 读取MLIR文件
def read_mlir_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {filename} 未找到。")
        return None

# 统计linalg算子出现次数
def count_linalg_ops(mlir_content):
    # 正则表达式匹配linalg算子
    pattern = r'\blinalg\.(\w+)'
    matches = re.findall(pattern, mlir_content)
    # 使用Counter统计出现次数
    return Counter(matches)

# 输出排名前10的算子名称及出现次数
def print_top_linalg_ops(counter, top_n=10):
    print("------------> 排名前10的linalg算子及出现次数：")
    for op, count in counter.most_common(top_n):
        print(f"{op}: {count}")

mlir_filename = "02-linalg-bufferized.mlir"
mlir_content = read_mlir_file(mlir_filename)
if mlir_content:
    linalg_counter = count_linalg_ops(mlir_content)
    print_top_linalg_ops(linalg_counter)

#----------------------- Selection & Tiling -----------------------#

# 提示用户输入
anchor_op_input = input("请输入需要抽象为加速器IP的算子名称：")
tiling_size_input = input("请输入加速器IP算子tiling尺寸：")
# for test
# anchor_op_input = "linalg.conv_2d_nchw_fchw"
# tiling_size_input = "1,1,1,1,1,1"


# 解析用户输入
tiling_size = tiling_size_input
anchor_op = anchor_op_input

# 定义SODA优化命令
soda_opt_command = [
    "soda-opt",
    "02-linalg-bufferized.mlir",
    "-soda-linalg-tile=tile-sizes={} anchor-op={}".format(tiling_size, anchor_op),
    "-cse"
]

# 构建输出文件名
output_filename = f"02-linalg-tiled-{anchor_op}-{tiling_size}.mlir"

# 执行SODA优化命令
try:
    subprocess.run(soda_opt_command, check=True, stdout=open(output_filename, 'w'))
    print(f"------------> SODA优化完成，输出文件：\n{output_filename}\n")
except subprocess.CalledProcessError as e:
    print(f"------------> 执行SODA优化命令时出错: {e}")

#----------------------- Convert & Host Code Generation -----------------------#

print("------------> 将继续进行算子转换及生成主机代码文件......\n")

# 定义SODA优化命令
soda_cvt_command = [
    "soda-opt",
    f"02-linalg-tiled-{anchor_op}-{tiling_size}.mlir",
    f"-convert-operation-to-soda=anchor-op={anchor_op}",
    "--soda-outline-bambu-code",
    "--soda-generate-bambu-hostcode"
]

linalg_host_filename = f"03-linalg-host-{anchor_op}-{tiling_size}.mlir"

# 执行命令
try:
    subprocess.run(soda_cvt_command, check=True, stdout=open(linalg_host_filename, 'w'))
    print(f"------------> 算子转换及主机代码(linalg)生成已完成完成，输出文件：\n{linalg_host_filename}。\n")
except subprocess.CalledProcessError as e:
    print(f"执行命令时出错: {e}")

# 定义SODA优化命令
soda_llvm_command = [
    "soda-opt",
    f"{linalg_host_filename}",
    "-lower-all-to-llvm"
]

llvm_host_filename = f"04-llvm-host-{anchor_op}-{tiling_size}.mlir"

try:
    subprocess.run(soda_llvm_command, check=True, stdout=open(llvm_host_filename, 'w'))
    print(f"------------> 主机代码(llvm dialect)生成已完成完成，输出文件：\n{llvm_host_filename}。\n")
except subprocess.CalledProcessError as e:
    print(f"执行命令时出错: {e}")


#----------------------- Redundancy removement -----------------------#
# 去除算子方言的名称，仅保留算子的名称
anchor_suffix = anchor_op.split('.')[-1]

# 定义需要移除和替换的模式
pattern_remove = r'llvm\.func\s@' + re.escape(anchor_suffix) + r'_kernel_[0-9]*_' + re.escape(anchor_suffix) + r'_kernel'
pattern_replace = re.escape(anchor_suffix) + r'_kernel(_[0-9]*)?_' + re.escape(anchor_suffix) + r'_kernel'

print("------------> 将继续进行冗余函数声明消除和调用替换（此处采用默认配置：单加速器配置）\n")



# 读取原始文件内容
def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

# 写入修改后的文件内容
def write_file(filename, lines):
    with open(filename, 'w') as file:
        file.writelines(lines)

# 删除含有特定pattern的行
def remove_pattern(lines, pattern):
    return [line for line in lines if not re.search(pattern, line)]

# 替换冗余的函数调用
def replace_pattern(lines, pattern, replacement):
    return [re.sub(pattern, replacement, line) for line in lines]


# 主函数

tailored_filename = f"04-llvm-host-tailored-{anchor_op}-{tiling_size}.mlir"
replaced_filename = f"04-llvm-host-{anchor_op}-{tiling_size}-replaced.mlir"

# 读取原始文件
file_content = read_file(llvm_host_filename)

# 移除指定模式的行
file_content = remove_pattern(file_content, pattern_remove)
file_content = replace_pattern(file_content, pattern_replace, anchor_suffix + "_kernel")
write_file(replaced_filename, file_content)

print(f"------------> 处理完成，输出文件：\n{replaced_filename}\n")

# 定义mlir命令
mlir_translate_command = [
    "mlir-translate",
    "--mlir-to-llvmir",
    f"{replaced_filename}"
]

tailored_filename = f"05-llvm-host-{anchor_op}-{tiling_size}.ll"

try:
    subprocess.run(mlir_translate_command, check=True, stdout=open(tailored_filename, 'w'))
    print(f"------------> 主机代码(llvm ir)生成已完成完成，输出文件：\n{tailored_filename}。\n")
except subprocess.CalledProcessError as e:
    print(f"执行命令时出错: {e}")

#----------------------- Declaration Extraction & Files Generation -----------------------#

print("------------> 正在提取加速器算子声明\n")
file_content = read_file(linalg_host_filename)

pattern_decl = re.escape(anchor_suffix) + r'_kernel_' + re.escape(anchor_suffix)
matching_lines = [line for line in file_content if re.search(pattern_decl, line)]

decl_filename = f"03-linalg-host-{anchor_op}-{tiling_size}-decl.mlir"

with open(decl_filename, 'w') as file:
    file.writelines(matching_lines)

print("------------> 正在生成运行时环境\n")

# 构建命令
output_filename = "main.cpp"

# 定义要执行的命令
command = [
    "python", "../../common/generateCustomizedRuntime.py",
    decl_filename,
    anchor_suffix
]

# 执行命令并重定向输出
with open(output_filename, 'w') as outfile:
    subprocess.run(command, stdout=outfile)

print(f"命令执行完成，输出已写入 {output_filename}\n")

print("------------> 正在生成系统配置文件\n")
# 定义要执行的命令
command = [
    "python", "../../common/generateConfigYAML.py",
    decl_filename,
    anchor_suffix
]

subprocess.run(command, check=True)
print(f"命令执行完成，输出已写入../config.yaml\n")

print("------------> 正在生成主控制程序代码\n")
# 定义要执行的命令
command = [
    "python", "../../common/generateTopAccel.py",
    decl_filename,
    anchor_suffix
]

# 构建命令
output_filename = "../hw/top.cpp"

# 执行命令并重定向输出
with open(output_filename, 'w') as outfile:
    subprocess.run(command, stdout=outfile)

print(f"命令执行完成，输出已写入{output_filename}\n")



#---------------- Remove All Intermediate Files ------------#
import glob
import os

# 获取当前目录下所有.mlir文件的路径
mlir_files = glob.glob('*.mlir')

# 遍历文件列表并删除每个文件
for file_path in mlir_files:
    try:
        if file_path != '02-linalg-on-tensors.mlir':
            os.remove(file_path)
            print(f"已删除文件：{file_path}\n")
    except OSError as e:
        print(f"删除文件时出错：{e}")

#---------------- Generate Empty HW File ------------#

# 定义文件名和路径
file_path = f"../hw/{anchor_suffix}_kernel.cpp"

# 定义要写入文件的注释
content = f"""// 请在该文件中实现加速器算法代码，并用#pragma unroll n原语实现并行

void {anchor_suffix}"""+"""_kernel(){



}"""

# 创建并写入文件
try:
    with open(file_path, 'w') as file:
        file.write(content + '\n')  # 在注释后添加换行符
    print(f"文件 {file_path} 已成功创建并写入内容。\n")
except IOError as e:
    print(f"创建文件时出错：{e}")

#---------------- Rename Forward File ------------#
forward_file_name = 'forward.ll'

try:
    os.rename(tailored_filename, forward_file_name)
    print(f"文件已从 {tailored_filename} 重命名为 {forward_file_name}\n")
except OSError as e:
    print(f"重命名文件时出错：{e}")

#------------------ Update the Makefile in HW Folder -----------#

# 定义文件路径
file_path = "../hw/Makefile"

# 新添加的行内容
new_line = f"KERN={anchor_suffix}_kernel.cpp top.cpp"

# 读取原始文件内容
try:
    with open(file_path, 'r') as file:
        original_content = file.read()
        
    # 将新行添加到文件内容的开头
    updated_content = new_line + '\n\n' + original_content
    
    # 将更新后的内容写回文件
    with open(file_path, 'w') as file:
        file.write(updated_content)
        
    print(f"已成功向 {file_path} 的开头添加KERN定义。\n")
except IOError as e:
    print(f"处理文件时出错：{e}")

#------------------ Adding Instructions to Config -----------#

# 定义配置文件路径
config_file_path = "../config.yml"


# # 读取原始文件内容
# try:
#     with open(config_file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 使用正则表达式找到'hw_config:'行的索引
#     pattern = re.compile(r"^hw_config:")
#     last_index = 0
#     for index, line in enumerate(lines):
#         if pattern.search(line):
#             last_index = index
#             break
    
#     # 删除'hw_config:'行及其后面的所有行
#     new_lines = lines[:last_index]
    
#     # 将更新后的内容写回文件
#     with open(config_file_path, 'w') as file:
#         file.writelines(new_lines)
    
#     print(f"已成功更新 {config_file_path}，删除了 'hw_config:' 行及其后面的所有行。")
# except IOError as e:
#     print(f"处理文件时出错：{e}")


# 打开文件并写入内容
with open(config_file_path, 'a') as config_file:  # 'a' 模式表示追加模式
    config_file.write("hw_config:\n")
    config_file.write("  top:\n")
    config_file.write(f"  {anchor_suffix}_kernel:\n")
    config_file.write("####### Modify the hardware configuration here. ######\n")

# 读取另一个文件并追加到配置文件
hw_config_instructions_path = "hw_config_instructions.yml"
with open(hw_config_instructions_path, 'r') as hw_config_file:
    instructions = hw_config_file.read()

with open(config_file_path, 'a') as config_file:
    config_file.write(instructions)

print(f"配置文件 {config_file_path} 已更新。\n")

