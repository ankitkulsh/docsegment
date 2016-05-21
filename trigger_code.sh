#!/bin/bash
g++ -ggdb 'pkg-config --cflags opencv' -o 'basename decode_object.cpp .cpp' decode_object.cpp 'pkg-config --libs opencv' -llept -ltesseract
g++ -ggdb 'pkg-config --cflags opencv' -o 'basename table.cpp .cpp' table.cpp 'pkg-config --libs opencv' -llept -ltesseract
