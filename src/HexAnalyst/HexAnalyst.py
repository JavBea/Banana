#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：HexAnalyst.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/8/27 20:33 
"""
import os
import struct
import textwrap
import cinrad


def read_file_with_encoding(file_path, encoding_list=None, max_preview_lines=20, save_to_txt=False):
    """
    读取文件，并尝试使用不同编码方式解析。

    :param file_path: 文件路径
    :param encoding_list: 要尝试的编码列表，可包含 "utf-8", "gbk", "latin-1", "ascii", "int", "float", "hexdump"
    :param max_preview_lines: 预览的最大行数（仅控制终端打印，不影响保存）
    :param save_to_txt: 是否将完整内容保存为 txt 文件
    """
    if encoding_list is None:
        encoding_list = ["utf-8", "utf-16", "gbk", "latin-1", "ascii", "int", "float", "hexdump"]

    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"开始分析文件: {file_path}")
    print("=" * 60)

    for enc in encoding_list:
        try:
            print(f"\n尝试使用编码/解析方式: {enc}")

            if enc in ["utf-8", "utf-16", "gbk", "latin-1", "ascii"]:
                # 普通文本编码
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    lines = f.readlines()

                    # 打印预览
                    for i, line in enumerate(lines):
                        if i >= max_preview_lines:
                            print("... (只显示部分内容)")
                            break
                        print(line.rstrip("\n"))

                    print(f"✅ 成功读取，编码方式: {enc}")

                    if save_to_txt:
                        base_name = os.path.basename(file_path)
                        output_file = f"{base_name}-解析结果-{enc}.txt"
                        with open(output_file, "w", encoding="utf-8") as out:
                            out.writelines(lines)
                        print(f"📂 内容已保存到: {output_file}")

            elif enc == "int":
                # 按 int32 解析
                with open(file_path, "rb") as f:
                    data = f.read()
                    ints = []
                    for i in range(0, len(data), 4):
                        if i + 4 <= len(data):
                            val = struct.unpack("i", data[i:i + 4])[0]
                            ints.append(val)

                    print("预览整数数据:", ints[:max_preview_lines])
                    if save_to_txt:
                        base_name = os.path.basename(file_path)
                        output_file = f"{base_name}-解析结果-int.txt"
                        with open(output_file, "w", encoding="utf-8") as out:
                            out.write("\n".join(map(str, ints)))
                        print(f"📂 整数数据已保存到: {output_file}")

            elif enc == "float":
                # 按 float32 解析
                with open(file_path, "rb") as f:
                    data = f.read()
                    floats = []
                    for i in range(0, len(data), 4):
                        if i + 4 <= len(data):
                            val = struct.unpack("f", data[i:i + 4])[0]
                            floats.append(val)

                    print("预览浮点数据:", floats[:max_preview_lines])
                    if save_to_txt:
                        base_name = os.path.basename(file_path)
                        output_file = f"{base_name}-解析结果-float.txt"
                        with open(output_file, "w", encoding="utf-8") as out:
                            out.write("\n".join(map(str, floats)))
                        print(f"📂 浮点数据已保存到: {output_file}")

            elif enc == "hexdump":
                # 十六进制输出
                with open(file_path, "rb") as f:
                    data = f.read()
                    hex_str = data.hex()  # 转为纯16进制字符串
                    # 每16字节一行显示
                    lines = textwrap.wrap(hex_str, 32)  # 16字节 = 32个十六进制字符

                    print("预览十六进制数据:")
                    for i, line in enumerate(lines):
                        if i >= max_preview_lines:
                            print("... (只显示部分内容)")
                            break
                        # 格式化成 16字节分隔 + ascii显示
                        hex_pairs = " ".join([line[j:j + 2] for j in range(0, len(line), 2)])
                        ascii_repr = "".join(
                            chr(int(line[j:j + 2], 16)) if 32 <= int(line[j:j + 2], 16) < 127 else "."
                            for j in range(0, len(line), 2)
                        )
                        print(f"{i * 16:08x}  {hex_pairs:<47}  {ascii_repr}")

                    if save_to_txt:
                        base_name = os.path.basename(file_path)
                        output_file = f"{base_name}-解析结果-hexdump.txt"
                        with open(output_file, "w", encoding="utf-8") as out:
                            for i, line in enumerate(lines):
                                hex_pairs = " ".join([line[j:j + 2] for j in range(0, len(line), 2)])
                                ascii_repr = "".join(
                                    chr(int(line[j:j + 2], 16)) if 32 <= int(line[j:j + 2], 16) < 127 else "."
                                    for j in range(0, len(line), 2)
                                )
                                out.write(f"{i * 16:08x}  {hex_pairs:<47}  {ascii_repr}\n")
                        print(f"📂 十六进制数据已保存到: {output_file}")

        except Exception as e:
            print(f"❌ 使用 {enc} 打开失败，错误信息: {e}")



def compare_file_edges(file1, file2, num_bytes=64, save_to_txt=False):
    """
    对比两个文件的开头和结尾二进制内容，并以hexdump风格排版。

    :param file1: 第一个文件路径
    :param file2: 第二个文件路径
    :param num_bytes: 比较的字节数（默认64字节）
    :param save_to_txt: 是否保存对比结果到txt文件
    """

    def read_edges(path, n):
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(n)
            if size > n:
                f.seek(-n, os.SEEK_END)
                tail = f.read(n)
            else:
                tail = b""
        return head, tail, size

    h1, t1, s1 = read_edges(file1, num_bytes)
    h2, t2, s2 = read_edges(file2, num_bytes)

    header_info = [
        f"文件1: {file1} (大小 {s1} 字节)",
        f"文件2: {file2} (大小 {s2} 字节)",
        "=" * 70
    ]

    def format_hexdump_block(data1, data2, base_offset=0):
        """生成hexdump对比字符串，每行16字节"""
        lines = []
        max_len = max(len(data1), len(data2))
        for i in range(0, max_len, 16):
            chunk1 = data1[i:i + 16]
            chunk2 = data2[i:i + 16]

            hex1 = " ".join(f"{b:02X}" for b in chunk1).ljust(47)
            hex2 = " ".join(f"{b:02X}" for b in chunk2).ljust(47)

            ascii1 = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk1).ljust(16)
            ascii2 = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk2).ljust(16)

            mark = " " if chunk1 == chunk2 else "≠"
            offset = f"{base_offset + i:08X}"

            lines.append(f"{offset}  {hex1}  {ascii1} {mark} {hex2}  {ascii2}")
        return "\n".join(lines)

    # 生成对比结果
    result = []
    result.extend(header_info)

    result.append(f"\n--- 文件开头 {num_bytes} 字节对比 ---")
    result.append(format_hexdump_block(h1, h2, 0))

    result.append(f"\n--- 文件结尾 {num_bytes} 字节对比 ---")
    off1 = max(s1 - num_bytes, 0)
    off2 = max(s2 - num_bytes, 0)
    result.append(format_hexdump_block(t1, t2, min(off1, off2)))

    final_output = "\n".join(result)

    # 打印到屏幕
    print(final_output)

    # 是否保存
    if save_to_txt:
        base1 = os.path.basename(file1)
        base2 = os.path.basename(file2)
        out_name = f"对比结果_{base1}_vs_{base2}.txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"\n📂 对比结果已保存到: {out_name}")


def read_plot(file):


    path = file
    f = cinrad.io.CinradReader(path)  # 老版本数据
    print(f)
    print(f.name, "雷达角度：", f.el)

    rl = list(f.iter_tilt(230, 'REF'))
    # 组合反射率
    cr = cinrad.easycalc.quick_cr(rl, resolution=(230, 366))
    # ppi出图
    fig = cinrad.visualize.PPI(cr, dpi=999, add_city_names=True)
    fig("test.png")

def hexdump(filepath, length=16, n_bytes=None, start_offset=0, save=False, save_path=None, do_print=False):
    """
    用类似 hexdump -C 的风格打印二进制文件内容
    :param filepath: 文件路径
    :param length: 每行显示字节数
    :param n_bytes: 最多读取的字节数 (默认 None 表示读到文件末尾)
    :param start_offset: 起始偏移量 (默认 0)
    :param save: 是否保存到 txt 文件 (默认 False)
    :param save_path: 保存路径 (默认 None，即不保存)
    :param do_print: 是否打印到控制台（默认False,即不保存）
    """
    output_lines = []

    with open(filepath, "rb") as f:
        f.seek(start_offset)  # 跳到指定偏移量
        offset = start_offset
        read_total = 0

        while True:
            # 控制最多读取字节数
            if n_bytes is not None:
                remain = n_bytes - read_total
                if remain <= 0:
                    break
                chunk = f.read(min(length, remain))
            else:
                chunk = f.read(length)

            if not chunk:
                break

            read_total += len(chunk)

            # 十六进制部分
            hex_bytes = " ".join(f"{b:02x}" for b in chunk)
            hex_bytes = hex_bytes.ljust(length * 3)

            # ASCII 部分
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)

            line = f"{offset:08x}  {hex_bytes} {ascii_str}"
            print(line)
            output_lines.append(line)

            offset += len(chunk)

    # 保存到文件
    if save:
        if not save_path:
            save_path = filepath+"-解析结果-hexdump.txt"
        else:
            save_path += os.path.basename(filepath) + "-解析结果-hexdump.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(output_lines))
        print(f"\n✅ 已保存到: {save_path}")


if __name__ == "__main__":

    f1 = r"E:\MyFiles\Projects\Banana\output\CAPPI0408.npy"

    # read_plot(f1)

    # 示例：比较两个 ARF 文件开头和结尾的64字节，并保存结果
    compare_file_edges(f1, f1, num_bytes=1280, save_to_txt=True)

    # # 用不同的编码或解析方式读取arf文件
    # hexdump(f12, length=16, save=True, save_path=r"./", do_print=True)
