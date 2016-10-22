# docsegment
Segments text, table, image regions

Open the image (PNG/TIFF/JPEG) in the imported folder and run the following script. Segments the regions as table (table.cpp), and text with bounding box information (decode.cpp)
#!/bin/bash
g++ -ggdb 'pkg-config --cflags opencv' -o 'basename decode_object.cpp .cpp' decode_object.cpp 'pkg-config --libs opencv' -llept -ltesseract
g++ -ggdb 'pkg-config --cflags opencv' -o 'basename table.cpp .cpp' table.cpp 'pkg-config --libs opencv' -llept -ltesseract
