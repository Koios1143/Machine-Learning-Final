# 2023 Machine Learning Final Project

<center>
<img src="https://hackmd.io/_uploads/SJ7U1GRma.png" width=500>
</center>
</br>
> Image from paper 「MindEye: fMRI-to-Image reconstruction & retrieval」

## Project 目標

- fMRI to image
- 使用 MindEye 架構，加上 Bagging 策略

## Dataset

From TA, possibly subset of NSD dataset.

### image encoder
> https://github.com/huggingface/diffusers/tree/main

### NN

首先我們確定每個 subject 的 training fmri 的資料大小分別是

- **lh**: $(5000, 19004)$
- **rh**: $(5000, 20544)$

兩個維度分別表示訓練的圖片編號以及 voxels。

目前對於上面是如何把輸入從 $(N \times 15000)$ 變成 $(N \times 4 \times 64 \times 64)$ 的，從 trace code 的結果來看應該是從 $(N \times 15724)$ 變成 $(N \times 16384)$。
這邊可以選擇繼續 trace code，甚至下載下來做各種測試，不過目前在環境上有點小卡關，可能需要再研究一下。

其他的話這裡也許會選擇把每個圖片的 lh 跟 rh 合在一起，變成 $(5000, 39548)$ 的資料輸入，目標變成 $(5000, 16384)$，然後再 reshape 成 $(5000, 4, 64, 64)$。


### image decoder
> https://github.com/huggingface/diffusers/tree/main
