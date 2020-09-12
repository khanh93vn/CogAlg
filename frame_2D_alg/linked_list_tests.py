import subprocess
from ctypes import *
from pprint import pprint
from struct import unpack

class SLinkedListElement(Structure):
    pass

SLinkedListElement._fields_ = [
    ('val', c_longlong),
    ('next', POINTER(SLinkedListElement)),
]

class SLinkedList(Structure):
    _fields_ = [
        ('first', POINTER(SLinkedListElement)),
    ]

    def __iter__(self):
        current = self.first
        while current:
            yield current[0].val
            current = current[0].next

# Compile C code
subprocess.run(['gcc', '-std=c11', '-fPIC', '-shared', '-o',
                'linked_list_tests.so', 'linked_list_tests.c'])

# Load C shared object
test_module = CDLL('linked_list_tests.so')
test_module.test_linked_list.restype = SLinkedList
print("Running...")
ll = test_module.test_linked_list()

print([*ll])
