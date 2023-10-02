"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    # TODO: Implement this function
    final = {}
    for item in text:
        final.setdefault(item, 0)
        final[item] += 1

    return final


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        for item in freq_dict:
            random_byte = (item + 1) % 256
            dummy_tree = HuffmanTree(random_byte)
            return HuffmanTree(None, HuffmanTree(item), dummy_tree)

    single_trees = []
    for item in freq_dict:
        single_trees.append(HuffmanTree(item))

    freq = []
    for item in freq_dict:
        freq.append(freq_dict[item])

    while len(single_trees) > 1:
        first_smallest = None
        first_smallest_freq = float('inf')
        second_smallest = None
        second_smallest_freq = float('inf')
        for i in range(len(single_trees)):
            if freq[i] < first_smallest_freq:
                second_smallest = first_smallest
                second_smallest_freq = first_smallest_freq
                first_smallest = single_trees[i]
                first_smallest_freq = freq[i]
            elif freq[i] < second_smallest_freq:
                second_smallest = single_trees[i]
                second_smallest_freq = freq[i]
        new_tree = HuffmanTree(None, first_smallest, second_smallest)
        freq.remove(first_smallest_freq)
        freq.remove(second_smallest_freq)
        single_trees.remove(first_smallest)
        single_trees.remove(second_smallest)
        freq.append(first_smallest_freq + second_smallest_freq)
        single_trees.append(new_tree)

    return single_trees[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    # TODO: Implement this function
    if tree.left is None or tree.right is None:
        return {tree.symbol: ""}
    left = get_codes(tree.left)
    right = get_codes(tree.right)
    for item in left:
        left[item] = "0" + left[item]
    for item in right:
        left[item] = "1" + right[item]
    return left


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    acc = helper_number_nodes(tree)
    i = 0
    for tree in acc:
        tree.number = i
        i += 1


def helper_number_nodes(tree: HuffmanTree) -> list[HuffmanTree]:
    if tree.left.symbol is not None and tree.right.symbol is not None:
        return [tree]
    elif tree.right.symbol is not None:
        left = helper_number_nodes(tree.left)
        left.append(tree)
        return left
    elif tree.left.symbol is not None:
        right = helper_number_nodes(tree.right)
        right.append(tree)
        return right
    else:
        left = helper_number_nodes(tree.left)
        left.extend(helper_number_nodes(tree.right))
        left.append(tree)
        return left


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # TODO: Implement this function
    temp = helper_avg_length(tree, 0, {})
    total_freq = 0
    total = 0
    for item in freq_dict:
        total_freq += freq_dict[item]
    for item in freq_dict:
        total += freq_dict[item] * temp[item]
    return total/total_freq


def helper_avg_length(tree: HuffmanTree, n: int, temp_dict: dict[int, int]):
    if tree.left is None and tree.right is None:
        temp_dict[tree.symbol] = n
    else:
        helper_avg_length(tree.left, n + 1, temp_dict)
        helper_avg_length(tree.right, n + 1, temp_dict)
    return temp_dict


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # TODO: Implement this function
    acc = ""
    for item in text:
        acc += codes[item]
    final = []
    for i in range(0, len(acc), 8):
        final.append(acc[i: i + 8])
    return bytes([bits_to_byte(item) for item in final])

    # acc = ""
    # for item in text:
    #     acc += codes[item]
    # temp = helper_compress_bytes(acc)
    # final = []
    # for item in temp:
    #     final.append(bits_to_byte(item))
    # return bytes(final)


# def helper_compress_bytes(total_bits: str):
#     acc = []
#     while total_bits != "":
#         temp = total_bits[:8]
#         acc.append(temp)
#         total_bits = total_bits[8:]
#     return acc


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    # TODO: Implement this function
    temp = helper_number_nodes(tree)
    acc = []
    for tree in temp:
        if tree.left.symbol is not None:
            acc.append(0)
            acc.append(tree.left.symbol)
        else:
            acc.append(1)
            acc.append(tree.left.number)
        if tree.right.symbol is not None:
            acc.append(0)
            acc.append(tree.right.symbol)
        else:
            acc.append(1)
            acc.append(tree.right.number)

    return bytes(acc)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # TODO: Implement this function
    root = node_lst[root_index]
    lst = node_lst[:root_index + 1]
    return helper_generate_tree_general(lst, root)


def helper_generate_tree_general(node_list, root_node):
    temp = HuffmanTree()
    if root_node.l_type == 0:
        temp.left = HuffmanTree(root_node.l_data)
    if root_node.r_type == 0:
        temp.right = HuffmanTree(root_node.r_data)
    if root_node.l_type == 1:
        temp.left = helper_generate_tree_general(node_list, node_list[root_node.l_data])
    if root_node.r_type == 1:
        temp.right = helper_generate_tree_general(node_list, node_list[root_node.r_data])
    return temp


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # TODO: Implement this function
    node_list = node_lst[:root_index + 1]
    return helper_generate_tree_postorder(node_list, node_list.pop())


def helper_generate_tree_postorder(node_list, curr_node):
    temp = HuffmanTree()
    if curr_node.r_type == 0:
        temp.right = HuffmanTree(curr_node.r_data)
    if curr_node.r_type == 1:
        var = node_list.pop()
        temp.right = helper_generate_tree_postorder(node_list, var)
    if curr_node.l_type == 0:
        temp.left = HuffmanTree(curr_node.l_data)
    if curr_node.l_type == 1:
        var = node_list.pop()
        temp.left = helper_generate_tree_postorder(node_list, var)
    return temp


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # TODO: Implement this function
    acc = ""
    final = []
    for item in text:
        acc += byte_to_bits(item)
    i = 0
    curr = 0
    while i < size:
        symbol, curr = helper_decompress_bytes(tree, acc, curr)
        final.append(symbol)
        i += 1
    return bytes(final)


def helper_decompress_bytes(tree, bits, i):
    while tree.symbol is None:
        if bits[i] == "0":
            tree = tree.left
        if bits[i] == "1":
            tree = tree.right
        i += 1
    return tree.symbol, i


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    # import doctest
    #
    # doctest.testmod()
    #
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'allowed-io': ['compress_file', 'decompress_file'],
    #     'allowed-import-modules': [
    #         'python_ta', 'doctest', 'typing', '__future__',
    #         'time', 'utils', 'huffman', 'random'
    #     ],
    #     'disable': ['W0401']
    # })cd

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
