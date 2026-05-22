# B.5 + B.6 image-as-modality SFT 2026-05-22T01:26:29+00:00

Branch HEAD: 6da30576a7928ef31f36d82f20eb7caaf79fe9d4

## B.5 image-as-modality (cam + HD map + bbox + ego) — L2_avg=0.6646  collision_avg=3.54%
  vs R1' baseline 0.6423 / 3.73%

## B.6 = B.5 + modality dropout p=0.1 — L2_avg=0.7998  collision_avg=3.52%
  vs R1' baseline 0.6423 / 3.73%


## Final summary
overlay         309G  207G  103G  67% /

total 12
drwxr-xr-x 2 root root   78 May 22 07:19 .
drwxr-xr-x 5 root root 4096 May 22 01:26 ..
-rw-r--r-- 1 root root 1373 May 22 04:28 B5_multimodal.json
-rw-r--r-- 1 root root 1377 May 22 07:19 B6_multimodal_dropout.json
