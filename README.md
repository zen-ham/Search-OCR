Digests all your image files into a pickle cluster with a fast searchengine GUI:
=

[![ ](https://img.shields.io/badge/GarrysMod_Workshop_Scripts-blue?logo=opensourceinitiative)](https://github.com/zen-ham/garrysmod_workshop_scripts) day 5 release!

<p align="center">
    <img src="https://github.com/zen-ham/Search-OCR/blob/master/repo_assests/Screenshot%202025-03-29%20151547.png" width="45%" />
    <img src="https://github.com/zen-ham/Search-OCR/blob/master/repo_assests/Screenshot%202025-03-29%20135909.png" width="45%" />
</p>

<p align="center">
    <img src="https://github.com/zen-ham/Search-OCR/blob/master/repo_assests/Screenshot%202025-03-29%20151512.png" style="width:100%;" />
</p>

---
Technical details:
-
The software is able to read a wide range of file types, many more then PIL supports for example. File types such as SVG and EPS are supported, along with almost every single other image format.

For quick indexing I implemented a recursive subdirectory file list function in [my library](https://github.com/zen-ham/zhmiscellany) that uses multithreading, multiprocessing (ray), and cache based optimizations, in order to index millions of files in ~2.0 seconds, to keep the boot time of the software small.

For optimizations in the search engine and in the actual processing pipeline of the files, the ray wrapper in [my library](https://github.com/zen-ham/zhmiscellany) was used extensively, this brought search times and rendering from 20+ seconds to ~0.5 seconds. 
