## 9. Bond sphere prediction in the new coder (coder blob version 5)

The new coder gains a prediction from bonded neighbours: for every atom with a bonded neighbour B coded earlier in the same frame, the coordinate axis with the largest offset from B is coded last and predicted from the sphere of the mean bond length around B (mode sphere); mode sphere plus circle also predicts the middle axis from the circle given by the bond length and the mean angle at B. Bonds are chosen by the encoder from the original coordinates (the earlier atom whose distance varies least) and stored in the file with the bond lengths and angles, so no topology is needed. The encoder codes each file with the sphere off, sphere and sphere plus circle (same predictor) and keeps the smallest; the mode is carried by the coder blob version, so the rest of the file is the same in every mode and no file is larger than with the sphere off. All sizes are whole .mdz files including the container and the metadata trailer; ratios are raw float32 bytes over file bytes. Model weights are not part of any ratio.

### 9.1 mdCATH, 16 test domains, T = 1,000 (mean ratio)

| max error | new-coder, sphere off (best of raw and mean) | new-coder with sphere | raw: off, sphere | mean: off, sphere | MDCompress best setting (-l 4 --preset archive, through the codec) | files larger than sphere off |
|---|---|---|---|---|---|---|
| 0.5 Å | 22.12 | **22.17** | 19.12, 19.31 | 22.04, 22.04 | 14.50 | 0 of 32 |
| 0.3 Å | 17.22 | **17.52** | 15.41, 16.15 | 17.18, 17.43 | 13.30 | 0 of 32 |
| 0.2 Å | 14.66 | **15.39** | 13.77, 14.67 | 14.51, 15.18 | 12.42 | 0 of 32 |
| 0.1 Å | 11.72 | **12.88** | 11.56, 12.74 | 11.43, 12.54 | 10.71 | 0 of 32 |
| lossless | 4.81 | **5.35** | 4.80, 5.35 | 4.74, 5.26 | 4.76 | 0 of 32 |

### 9.2 mdCATH, 16 test domains, T = 250 (mean ratio)

| max error | new-coder, sphere off (best of raw and mean) | new-coder with sphere | raw: off, sphere | mean: off, sphere | MDCompress best setting (-l 4 --preset archive, through the codec) | files larger than sphere off |
|---|---|---|---|---|---|---|
| 0.5 Å | 21.66 | **21.68** | 18.91, 18.94 | 21.52, 21.52 | 14.01 | 0 of 32 |
| 0.3 Å | 16.87 | **17.05** | 15.27, 15.82 | 16.79, 16.86 | 12.89 | 0 of 32 |
| 0.2 Å | 14.41 | **14.89** | 13.63, 14.36 | 14.16, 14.57 | 12.06 | 0 of 32 |
| 0.1 Å | 11.58 | **12.55** | 11.47, 12.48 | 11.15, 12.01 | 10.45 | 0 of 32 |
| lossless | 4.79 | **5.29** | 4.79, 5.29 | 4.68, 5.13 | 4.70 | 0 of 32 |

Files larger than sphere off counts coder blobs; whole files where the encoder kept the sphere off differ from the production file only by the run time fields of the metadata trailer (a few bytes).

MDCompress column: the best single setting over the 16 domains is `-l 4 --preset archive` at every bound once the archive preset is also run at 0.3 and 0.2 Å (resolution 60,000 and 40,000 fm), written through the codec (about 1.5 KB of container and trailer per file). The summary of section 4.6 used native MDCompress files and the archive preset only at 0.5 Å, 0.1 Å and lossless, so its 0.3 and 0.2 Å values (12.55 and 11.70) are plain `-l 4`.

### 9.3 MDDB protein systems, T = 1,000 (ratio)

| system | atoms | max error | new-coder, sphere off | new-coder with sphere | mode (raw, mean) | MDCompress best setting |
|---|---|---|---|---|---|---|
| small_1APQ | 415 | 0.5 Å | 28.81 | **28.81** | off, off | 17.41 |
| small_1APQ | 415 | 0.3 Å | 23.11 | **23.11** | off, off | 15.37 |
| small_1APQ | 415 | 0.2 Å | 19.14 | **19.33** | sphere, sphere | 13.63 |
| small_1APQ | 415 | 0.1 Å | 14.22 | **15.27** | sphere, sphere | 11.81 |
| small_1APQ | 415 | lossless | 5.23 | **5.80** | circle, circle | 5.04 |
| medium_12CA | 2028 | 0.5 Å | 40.09 | **40.09** | off, off | 18.57 |
| medium_12CA | 2028 | 0.3 Å | 27.69 | **27.69** | off, off | 16.45 |
| medium_12CA | 2028 | 0.2 Å | 22.00 | **22.01** | off, sphere | 14.68 |
| medium_12CA | 2028 | 0.1 Å | 15.66 | **16.62** | sphere, sphere | 12.28 |
| medium_12CA | 2028 | lossless | 5.43 | **5.93** | circle, circle | 5.15 |
| large_1AYX | 3870 | 0.5 Å | 47.98 | **47.98** | off, off | 18.99 |
| large_1AYX | 3870 | 0.3 Å | 31.07 | **31.07** | off, off | 16.84 |
| large_1AYX | 3870 | 0.2 Å | 24.12 | **24.12** | off, off | 15.13 |
| large_1AYX | 3870 | 0.1 Å | 16.69 | **17.53** | sphere, sphere | 12.52 |
| large_1AYX | 3870 | lossless | 5.56 | **6.04** | circle, sphere | 5.23 |

MDCompress here is the better of `-l 2` and `-l 4 --preset archive` per system. The DNA, RNA, protein and DNA complex and membrane systems are not part of this section.

### 9.4 Hybrid against new-coder with sphere (mdCATH, 16 domains, T = 1,000, mean ratio)

The hybrid streams are still coded with the sphere off. 'with sphere' is the measured size if both hybrid coder streams used the sphere (each stream coded again with auto, bonds from the reconstructed positions); those files were not written.

| max error | hybrid, packer only | packer only, with sphere | hybrid, full predictor | full predictor, with sphere | new-coder with sphere | MDCompress best setting |
|---|---|---|---|---|---|---|
| 0.5 Å | 21.19 | 21.19 | running, 1 of 16 domains done | running, 1 of 16 domains done | 22.17 | 14.50 |
| 0.3 Å | 17.18 | 17.18 | running, 1 of 16 domains done | running, 1 of 16 domains done | 17.52 | 13.30 |
| 0.2 Å | 14.82 | 14.82 | running, 1 of 16 domains done | running, 1 of 16 domains done | 15.39 | 12.42 |
| 0.1 Å | 11.65 | 12.69 | running | running | 12.88 | 10.71 |
| lossless | 4.77 | 5.23 | running | running | 5.35 | 4.76 |

### 9.5 Timings (median per file, seconds, 2 core machine, encode includes the codec's verification decode)

| T | max error | encode, sphere off | encode, with sphere | decode, sphere off | decode, with sphere |
|---|---|---|---|---|---|
| 1,000 | 0.2 Å | 2.38 | 3.44 | 0.39 | 0.41 |
| 1,000 | 0.1 Å | 2.77 | 3.89 | 0.41 | 0.50 |
| 1,000 | lossless | 4.16 | 6.55 | 0.75 | 0.89 |
| 250 | 0.2 Å | 0.75 | 1.06 | 0.09 | 0.11 |
| 250 | 0.1 Å | 0.81 | 1.12 | 0.10 | 0.11 |
| 250 | lossless | 1.22 | 1.75 | 0.18 | 0.25 |

The encode with sphere is measured in an analysis run that also coded a 128 frame subsample of every mode; that subsample time is subtracted. Decoding needs no extra pass; its extra cost is the sphere and circle arithmetic.

### 9.6 Portability

A bundle of 24 coded blobs (8 in the experimental format, 16 in coder blob version 5, lossless and 0.1 Å, sphere and sphere plus circle, zero and nonzero base) was decoded in a fresh process from the files alone with gcc and clang at -O0, -O2, -O3 and -O3 -march=native and with aarch64 gcc at -O0, -O2 and -O3 under qemu: every decode matched the stored SHA-256 of the integers. The coder must be compiled with -ffp-contract=off -fno-fast-math; an aarch64 build with FMA contraction allowed decoded 5 of the 8 experimental blobs wrongly. A second physical machine has not run the bundle yet.

