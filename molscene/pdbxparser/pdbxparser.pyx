# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Declare the C function isspace from <ctype.h>
cdef extern from "ctype.h":
    int isspace(int)

from cpython.unicode cimport PyUnicode_DecodeUTF8

cdef list split_line(bytes line_bytes):
    """
    A fast C-level splitter for a bytes line.
    Splits on any whitespace (using isspace) and returns a Python list of Unicode strings.
    """
    cdef Py_ssize_t length = len(line_bytes)
    cdef const char* s = line_bytes  # pointer to the bytes data
    cdef Py_ssize_t i = 0, start = 0
    cdef list tokens = []
    cdef object token
    cdef int in_token = 0

    while i < length:
        # Cast to unsigned char then to int for isspace
        if isspace(<int>(<unsigned char>s[i])):
            if in_token:
                token = PyUnicode_DecodeUTF8(s + start, i - start, "strict")
                tokens.append(token)
                in_token = 0
            i += 1
        else:
            if not in_token:
                start = i
                in_token = 1
            i += 1
    if in_token:
        token = PyUnicode_DecodeUTF8(s + start, i - start, "strict")
        tokens.append(token)
    return tokens


def extract_atom_section(str cif_file):
    """
    Extracts only the _atom_site section from an mmCIF file as fast as possible.
    
    This Cython routine:
      • Reads the file in binary mode
      • Iterates over the bytes with minimal overhead
      • Uses a custom C function for splitting tokens
      • Stops as soon as the _atom section ends.
    
    Args:
        cif_file (str): Path to the CIF file.
    
    Returns:
        tuple: (List of column headers, List of parsed atom data rows)
    """
    cdef list atom_header = []
    cdef list atom_data = []
    cdef int in_atom_section = 0
    cdef bytes content
    # Read entire file as bytes
    with open(cif_file, "rb") as f:
         content = f.read()
    cdef Py_ssize_t n = len(content)
    cdef const char* buf = content  # pointer into content
    cdef Py_ssize_t i = 0, line_start = 0, line_len
    cdef bytes line_bytes
    cdef Py_ssize_t dot_index

    while i < n:
        if buf[i] == ord('\n'):
            line_len = i - line_start
            line_bytes = content[line_start:i].strip()
            if line_bytes:
                if line_bytes[0] == ord('#'):
                    pass  # skip comments
                elif line_bytes.startswith(b"loop_"):
                    in_atom_section = 0  # reset section flag
                elif line_bytes.startswith(b"_atom_site."):
                    # Header line: extract header name after the dot.
                    dot_index = line_bytes.find(b'.')
                    if dot_index != -1:
                        atom_header.append(line_bytes[dot_index+1:].decode('utf-8'))
                    in_atom_section = 1
                elif in_atom_section:
                    # If a new section starts, break out.
                    if line_bytes[0] == ord('_') and not line_bytes.startswith(b"_atom_site."):
                        break
                    # Otherwise, split tokens in the data line.
                    atom_data.append(split_line(line_bytes))
            i += 1
            line_start = i
        else:
            i += 1

    # Process the last line if there's no trailing newline
    if line_start < n:
        line_bytes = content[line_start:n].strip()
        if line_bytes:
            if line_bytes[0] != ord('#'):
                if line_bytes.startswith(b"loop_"):
                    in_atom_section = 0
                elif line_bytes.startswith(b"_atom_site."):
                    dot_index = line_bytes.find(b'.')
                    if dot_index != -1:
                        atom_header.append(line_bytes[dot_index+1:].decode('utf-8'))
                    in_atom_section = 1
                elif in_atom_section:
                    if not (line_bytes[0] == ord('_') and not line_bytes.startswith(b"_atom_site.")):
                        atom_data.append(split_line(line_bytes))
    return atom_header, atom_data
