#!/bin/bash

mkdir -p 'data'
mkdir -p 'void_release'
mkdir -p 'void_release/tmp'
mkdir -p 'void_release/void_150'
mkdir -p 'void_release/void_500'
mkdir -p 'void_release/void_1500'
mkdir -p 'void_release/void_150/data'
mkdir -p 'void_release/void_500/data'
mkdir -p 'void_release/void_1500/data'

cd 'void_release'

# VOID 150 data urls
if [ $# -eq 0 ]; then
  gdown https://drive.google.com/uc?id=1sOImN1ipcbCV1MA-rvK_26BQf9fWtIvN
fi

unzip -o 'void_150-0.zip' -d 'void_150/'
mv 'void_150-0.zip' 'tmp/'

void_150_urls=(
https://drive.google.com/uc?id=19N3NcSeYsk21NHIWupsnWKVkUdd6keN0
https://drive.google.com/uc?id=1b_9uOV5ntdHEAvmCq5KfGtB26Y4rg3ES
https://drive.google.com/uc?id=1eZPod2dv09-uB8kOXzvhgCz_v2DSmxgU
https://drive.google.com/uc?id=1AK8u4gVw2zqyM_UjEyMDSe4HrlwHWW35
https://drive.google.com/uc?id=10pkez7VOpJzkpUooSJ1BLXhcOf96YQfV
https://drive.google.com/uc?id=1Zxj-iUx2v1xJ91NAJGlWKeefNRh1b7hX
https://drive.google.com/uc?id=1CAI_r0YR3NFX1H_DKq0PPaUxqDT3VoLK
https://drive.google.com/uc?id=1vx3ShmEM54HGK3AgA7P4gEmqeWXFdzHn
https://drive.google.com/uc?id=1z2_Kf8TeDNBF-9TVisOtwH-KhRetff08
https://drive.google.com/uc?id=1j4A12KBP6zMApwV3q1Pc5bzhJlFyWyQi
https://drive.google.com/uc?id=1Y_glEz5zbO6_AucwMkyzYyCDKMVm3hvW
https://drive.google.com/uc?id=1mXOSXpanT8a05DTAZYvHQ17bPbeSWpiF
https://drive.google.com/uc?id=16EKLKWZnJEym5U-XqPgZ66O3SG-iRvwE
https://drive.google.com/uc?id=1KpeaDbKubsr8AgHkJ5nZkAm1yc6DD2T4
https://drive.google.com/uc?id=10t-pwGSNkEeCeoHaWFAbbLowhUs6_qbX
https://drive.google.com/uc?id=1B9kTQAIhSzerymUYCK2_Ah2ACAz1w25K
https://drive.google.com/uc?id=1ioK18uvMC2cwIUFfvpzBICfe8slracEG
https://drive.google.com/uc?id=1pMy5ZEAa5m7U3O-WYi5PNyfcBUQeOcCA
https://drive.google.com/uc?id=1JQP1AiyGvqmAun-X5gQVbX9pyAT4l_VI
https://drive.google.com/uc?id=1Ct-F1hPs-NNlR18hEo6SmKz0GIE73S_Y
https://drive.google.com/uc?id=1LsV3_lYtLjHjuGM-fHWstt6tCzNN5faZ
https://drive.google.com/uc?id=19ewJzyMhL9r6KbQ-341YnMA5RIR9sz81
https://drive.google.com/uc?id=1htSjsX6pB-FvajwLqmmpaVdSGTP3RLtI
https://drive.google.com/uc?id=1J6jCF_HH4KhTj-Loxx0v_WBAN7B3jSRp
https://drive.google.com/uc?id=1NK252f6oyPLI7-lrM_38t_NJlngKT249
https://drive.google.com/uc?id=17__wyqkWd1pelC4OxlYGIkYY4tIoO73i
https://drive.google.com/uc?id=10rDH7BkIDo94ZSE6TMFNhBuhPEwHitfQ
https://drive.google.com/uc?id=1igMplf-QHYuyJkLTNjNvYXh6CYkoFUUD
https://drive.google.com/uc?id=1G6MKLrN2dDoR1hI_G6xmEUZXcGWGqGaC
https://drive.google.com/uc?id=18cx_5E4vcqUlHVkytV68y3ntaCRvtlwJ
https://drive.google.com/uc?id=1dHHPY0IPcfJkRQNGkMTQVxUJfXihOht1
https://drive.google.com/uc?id=1AF2ghTsfGtYX3R5G0ZC34dWQZ2iJ305d
https://drive.google.com/uc?id=1pIh5tHDwJY_cuaUVq6hlZJyHiqmH1lGz
https://drive.google.com/uc?id=1nMP5S26kCVH-gKhJkmnEnTs7qvUh6izW
https://drive.google.com/uc?id=12hOHsR_GiTjKzehXHcnqClyhbn9dNrQa
https://drive.google.com/uc?id=1_bkEVDtixXAHkYmfgdag8pNkXvFHEHSs
https://drive.google.com/uc?id=1PJ0UgARb1enZwQuiYxjcZcH1-PzScGwF
https://drive.google.com/uc?id=1riXpESKwSY_kM7658xxaYuV9u__Tq33A
https://drive.google.com/uc?id=1yjVdTykDWcg8YrLCW_aIKMcYTkNGrMG3
https://drive.google.com/uc?id=1IJJoBUe4e3foQvBKM22jPx2l2y1lBL8-
https://drive.google.com/uc?id=1larB_l5SsQ6J0V9e2SlyawP0yffWRJ_-
https://drive.google.com/uc?id=1iazSXb6N06x0xuVMKPBlYchIDHbY6vfE
https://drive.google.com/uc?id=1VUDF9Hneqswj6qjjP4RT_pFXMmeNoZDI
https://drive.google.com/uc?id=1Ywi5QU5fEJM-Sai9FWTs2IF9HsOy99lg
https://drive.google.com/uc?id=1pQpFUZJedBlu9AVPKfud8pu0FBBQ7g-B
https://drive.google.com/uc?id=1x-WAuz2B7c1pWdfSLhlqUrMgs6bHgFkS
https://drive.google.com/uc?id=16LFIKTRp1-NVgtd9QCIEoJ4STaO2wf22
https://drive.google.com/uc?id=17YHYcFTrm6Agv6cmGm_1GeBvTKW3YjtM
https://drive.google.com/uc?id=1MUcj3JpyNGvM_e9XHPzan4YrdMI4ZWc8
https://drive.google.com/uc?id=13oS8RuD5qfLKpJC7bkQqVEXVmMJcF0-R
https://drive.google.com/uc?id=1fIkBja41bOS8jQAyZ8UlHDiA8GFiSq3l
https://drive.google.com/uc?id=1z6YJgCEOi2P4L6kCxgxorsMie5tYKBHr
https://drive.google.com/uc?id=1Nph7ay5LvdIpoC2CBJm98E6ge5wny1Oy
https://drive.google.com/uc?id=1dQGSXvBNn-fFHS_40Kxb9C60WevBNfPO
https://drive.google.com/uc?id=1oDEt6KAJVJerwQImgvlouzOt03YBVnPn
https://drive.google.com/uc?id=1WkUAo823A2lBhYW04KfuFeKXNctiurrA
)

if [ $# -eq 0 ]; then
  for file_url in ${void_150_urls[@]}; do
    gdown $file_url
  done
fi

file_id=1
while [ $file_id -le ${#void_150_urls[@]} ]; do
  unzip -o 'void_150-'${file_id}'.zip' -d 'void_150/data/'
  mv 'void_150-'${file_id}'.zip' 'tmp/'
  file_id=$(( ${file_id} + 1 ))
done

# VOID 500 data urls
if [$# -eq 0]; then
  gdown https://drive.google.com/uc?id=1Q0T8UUZjVDrfIDEMigMyu_7nb-gcQdeW
fi
unzip -o 'void_500-0.zip' -d 'void_500/'
mv 'void_500-0.zip' 'tmp/'

void_500_urls=(
https://drive.google.com/uc?id=1VjRJs97FICTq2liWh_mEJUZQyTXXB0F-
https://drive.google.com/uc?id=1fy6zAoJoaFpsA4uGrPYs06D3fsRj6vZv
https://drive.google.com/uc?id=1sEY3pYhr2DRXzrxaMVqq6F7Jh5lqcDx6
https://drive.google.com/uc?id=1tNvswogQZ-gP9rk3QMzGdcurse5SFudQ
https://drive.google.com/uc?id=1oBvtWXYJQiyMXDBmb3BrihoSS_ST--9z
https://drive.google.com/uc?id=15mpEFx_czKCvQRchrMst60KfH-ahpCoz
https://drive.google.com/uc?id=1tRvjTHTeeFdlKfTrrEqkRi1I6afeEzlP
https://drive.google.com/uc?id=1_7FDEaKWmT6oUvWla49JeRZFuTti_Rvi
https://drive.google.com/uc?id=1uauvKEE2_SSbKPC-6K59_pqu1gN6j01z
https://drive.google.com/uc?id=1CiS9oswBegNzqCnBC-IImtFMNaeDjp6S
https://drive.google.com/uc?id=1yFVKTgCBvzwoy6NwMQkuTYx1nhXQ1xTK
https://drive.google.com/uc?id=1JEZ_wNCfqyavy5LVPppflEqJR4vmmzqT
https://drive.google.com/uc?id=1sIywdmcHG9dUJ9meP3TWaOzOmmaYRssc
https://drive.google.com/uc?id=1sSLPoPOzpT8h9L8I4ByPcRuNbiB7aqox
https://drive.google.com/uc?id=1BamNvjeryshyKZ0ZIgefzhQ2SJktjGyY
https://drive.google.com/uc?id=1GUP0lVRayjuvm5ooOTbSvRliwQWJ9sAQ
https://drive.google.com/uc?id=1FZkFeEsWkN-zAwmhvV-ppfBORdKIuEAc
https://drive.google.com/uc?id=1lN7eXtBi81x2BhaT5Hf-1LeJWjgKtqmq
https://drive.google.com/uc?id=1OOAdFSZ65rstbPhIhHRXMk6KapeRv9uC
https://drive.google.com/uc?id=1L8qruHwxKDaMvcUIOlItK5EolVKW9GTz
https://drive.google.com/uc?id=1DjlOL2_8vn01wc5oNRT9hU_a_XldbBC6
https://drive.google.com/uc?id=1GnwmPYZZfnx9N_kSriAIIumh8nHvMiO4
https://drive.google.com/uc?id=1U5I-NKZPWx_oLQFZ-wAkzy4_vPnBAQML
https://drive.google.com/uc?id=1XukaiJ3nnTm5ZIkRRkvkC3eGHQ60claf
https://drive.google.com/uc?id=1ra_ApRd8Ytmb2Zpv6YjsZJLidvh6RCjR
https://drive.google.com/uc?id=1deLHpF7Zs6zzhThRM8unhHSyp7XRhc83
https://drive.google.com/uc?id=1WeynVDGHFQ7C_YuDEOfFlaBujvyBZXIJ
https://drive.google.com/uc?id=1XcgdyGkK9OILggdyCHdk-pNxU83kCaVV
https://drive.google.com/uc?id=1_AARgJrPp4B8jMBsm6YHAQkvD-o5PIk2
https://drive.google.com/uc?id=1BlA-Ra5vTqu4v4DTtsJOwGY1oDO63j-H
https://drive.google.com/uc?id=1mx06TjVkNWBCOWsHiiJjr6f1H1iVP38K
https://drive.google.com/uc?id=1bVVp6LIjLeHaWrMse7F76yuAxVcy8lQe
https://drive.google.com/uc?id=1BHNHoUNT7gWnffcut5qcyyCZfs9jCsWY
https://drive.google.com/uc?id=1MpEqQKsTXk6HsnjCuSYnbQIS59plB_AO
https://drive.google.com/uc?id=1b-AVVAwYHPdgnVaFgcsbwhAjX-VXZK_j
https://drive.google.com/uc?id=1dROxzTRSp_4DcZgBqGXHNBjv-aE3S8Vj
https://drive.google.com/uc?id=1zB5WQNmy1-XnJqLPxMqA3J3l6RSW48t_
https://drive.google.com/uc?id=1y-cMiq-TelsQmdWH81yGJbXxbzAEMz5E
https://drive.google.com/uc?id=1ksorWD-BcZdfSDKS2nTv7TW6v8-33itM
https://drive.google.com/uc?id=1wPHYaOU40vWS8yuAqSxduDvbvCamNVob
https://drive.google.com/uc?id=1ZA5fPJYCcXfG-4nnI51jtrIQV5Au7rcC
https://drive.google.com/uc?id=1T2VfLj1c6wgub-jOmUznHvcZRx8tehiV
https://drive.google.com/uc?id=1ZqHjOaRjPDc7-lZlNCcTWCP63dVcKE6J
https://drive.google.com/uc?id=187zEmKDfKKmS2JRSoUPjZnPBgmyrnRrj
https://drive.google.com/uc?id=1yrWnDdYGkJxVDcWZhJnDVCUHfx5NuPov
https://drive.google.com/uc?id=1tflJncQrYVs0QVTWLvFna_5nDNzTypMi
https://drive.google.com/uc?id=14lrFljkS5cxBpabxXn36hrZyNbTOquB3
https://drive.google.com/uc?id=1Zq1kJGDtBCO2Y_R7B-JnINHZvMuqxvtD
https://drive.google.com/uc?id=1-1sma6-zZp2n_3rkjncTwyjxq2FblXvr
https://drive.google.com/uc?id=15KcOyA-0mGtgtstp_FlRNo6FNInIvkL2
https://drive.google.com/uc?id=1icVzYfQal00uN0dg8pbhgGXGHAWDVDpX
https://drive.google.com/uc?id=1YxYcFrVpv42QUtTW2GNsmIh7rzr6DGgw
https://drive.google.com/uc?id=1iATSoPCeOzzFDechAApc_BtXlVRf2S5w
https://drive.google.com/uc?id=1Hak97kml1-gYXaNvcUoWksc435RPS7uL
https://drive.google.com/uc?id=1JZEfFrhMKS47kRcFnVZAcnh-_ccD__pD
https://drive.google.com/uc?id=1ZX4PZA2Q-1J2bXcGc9l9xFVv_G2P969C
)

if [ $# -eq 0 ]; then
  for file_url in ${void_500_urls[@]}; do
    gdown $file_url
  done
fi

file_id=1
while [ $file_id -le ${#void_500_urls[@]} ]; do
  unzip -o 'void_500-'${file_id}'.zip' -d 'void_500/data/'
  mv 'void_500-'${file_id}'.zip' 'tmp/'
  file_id=$(( ${file_id} + 1 ))
done

# VOID 1500 data urls
if [$# -eq 0]; then
  gdown https://drive.google.com/uc?id=10p61P0rUmK_-GnZ_WcfYlWJGwLlEd7KW
fi
unzip -o 'void_1500-0.zip' -d 'void_1500/'
mv 'void_1500-0.zip' 'tmp/'

void_1500_urls=(
https://drive.google.com/uc?id=1bbN46kR_hcH3GG8-jGRqAI433uddYrnc
https://drive.google.com/uc?id=1Z7xBmGDUa-wjWoB9BSMkyrs1GOEsyDSh
https://drive.google.com/uc?id=1ddHJBS3hL64hWam3qLeuM6OBma84fZ5f
https://drive.google.com/uc?id=1fKXx6C8osFvKwWtwYlXlCfgEQISYS23l
https://drive.google.com/uc?id=1H2nY4QFDl0kvYN5QygK1d0D1LtDpzx7P
https://drive.google.com/uc?id=1r6I9mbNEF7WfQSGVv0p0RMoSCtTlQlDv
https://drive.google.com/uc?id=1PQxpM63li73hop0mjkZncBBN7_K6uFJb
https://drive.google.com/uc?id=1ss3H6hUjacj6v1iwD6rAaC8tLclhWnmP
https://drive.google.com/uc?id=1QWWfeFmXusyU-oGLi6bqHY6uxmnUP9CG
https://drive.google.com/uc?id=13BwiC1vwSWlMAQfJJjiVEL2QLQcKZ3oL
https://drive.google.com/uc?id=1z5Hr8HOD8qpqm4Tpm2UeaAlTlM9Sfven
https://drive.google.com/uc?id=1O0vMD6HrANmF57c9eXJ1cF29ZpgEGyzZ
https://drive.google.com/uc?id=13c50Dr065sVBu7qlXSZ3FNeXKRT1Gzyt
https://drive.google.com/uc?id=1Hqs_7bT_ha7LdHkDiHNiryL_xny1Ke23
https://drive.google.com/uc?id=1KPq6vl69uPnYBgKungfePt-V6u3B0TFb
https://drive.google.com/uc?id=1rQgkQTfRDeIOBmE1nMpx9b3_PysR2Yv_
https://drive.google.com/uc?id=1uSRvolDsgBqnr2sa__UFORcd9UPAg1_F
https://drive.google.com/uc?id=19RBWO-vVoXhL7oUjhnninL5n3z8jzyKu
https://drive.google.com/uc?id=1FHOj1iNIQ_J_PBbAbEp7zaIlyBuiO3vM
https://drive.google.com/uc?id=12GyvNz62GMPu4iUVzlO6DtResOXX-IV-
https://drive.google.com/uc?id=1A3wa4xOnIddnqh7FLlEQOZ8m4BV8jLS0
https://drive.google.com/uc?id=1ddIFcYd2zsFCgq4hWG8RsKROLyyBJai3
https://drive.google.com/uc?id=1XCSySjZATM2tdr0xjp4_-aGZFDlk2bnz
https://drive.google.com/uc?id=1JFzdTpt5j1-XNx34Xfh0YN_DAg7-LziY
https://drive.google.com/uc?id=1tGURNPniIN0y6AUxGnrA3smXQuLDlwUT
https://drive.google.com/uc?id=1Tl-SSBs992_k6PXWEPmcNQbvuOj2gvZW
https://drive.google.com/uc?id=1_fk5ABoPmm6cmNEdm_pY-WXlKDtUgxwD
https://drive.google.com/uc?id=1V66_5byl_Py-4kRJx1jOWR8RwFT0Yale
https://drive.google.com/uc?id=18iTILK5HSLnOUuIr0IP135RKL33uX1Kf
https://drive.google.com/uc?id=1S_9hd9VGLu2JKjXAyq7VEYWYcI9aSquQ
https://drive.google.com/uc?id=1KmVwLdHjujgxhOjxEXoXIp5i2nwleyJZ
https://drive.google.com/uc?id=189rLtntGUabTkMXa_p0gPDKAFy6_d_CU
https://drive.google.com/uc?id=17N0qkANF2-tY0qvHLVJpeYZxNVGcb-pd
https://drive.google.com/uc?id=1M7iVofcoLpDQtQbWeUtdldk20ZGMUagS
https://drive.google.com/uc?id=1SEXeuxWiinQPVjCIc-eu6t1HNk4HAJ1d
https://drive.google.com/uc?id=1ACtikMf8unoiSA7L2tWUWIv18mf-mKe5
https://drive.google.com/uc?id=1gknuOlortoDqF-vR2GQfVhTrfip4F1W5
https://drive.google.com/uc?id=1tA5MfhEXglO41tOzplbZAhsj7ehBh4Wh
https://drive.google.com/uc?id=1g9iGwEbBiUAvyTbIrXxPVOrj7snLpRqR
https://drive.google.com/uc?id=11HyFW0tdBkt9tRTsbxIGciUJH3mq53-J
https://drive.google.com/uc?id=1WvUvnfaPJMPjXj46SkWNfKKXuvfdj12v
https://drive.google.com/uc?id=1S6sC3zM9SSQmjAEtZqL8PrYkvIx6kB2U
https://drive.google.com/uc?id=1Fukx8655O2oQNeQvr36R825XSH-1NWaD
https://drive.google.com/uc?id=1W2NnLL2WeapDTXztrxuZNJA-OJhR4BOs
https://drive.google.com/uc?id=1yuqceufmRD2--dzykVBnDDWLwP8G6hhV
https://drive.google.com/uc?id=1Z8rjbizz23MtnMntzH2VR0OOr_kVU4a7
https://drive.google.com/uc?id=13iEfKI2jxb8dMEXsLvntNC9p9UjWne_a
https://drive.google.com/uc?id=1RwIKstYEP6zLTrIg17xXAi7emniw6nBx
https://drive.google.com/uc?id=1VOx8Ka0Mi7-I6ll8jcOSOxYQpxJBiPxU
https://drive.google.com/uc?id=1mQF2sYY_V4Y0x_AN0KOCktu2Xt7-TQm0
https://drive.google.com/uc?id=1fyyWsgYll2ekxhZoZm0oRW5BgneVT-s2
https://drive.google.com/uc?id=1OSTX95IL_4uYl0qwjBkQ3BDnlYJgXdoY
https://drive.google.com/uc?id=1n8zugXuD8IuJfVjmEN-_eGtXb3A7OV43
https://drive.google.com/uc?id=1E5SDMVsV5Qsl-BmvnIzCSxrDwzFxKgx4
https://drive.google.com/uc?id=1lcvsOfJwT04VR4TDJqW7E-r0KeyVlSA0
https://drive.google.com/uc?id=1PfIfDhkw0hFDdC2PBgrFpvEQWUkmXtzX
)

if [ $# -eq 0 ]; then
  for file_url in ${void_1500_urls[@]}; do
    gdown $file_url
  done
fi

file_id=1
while [ $file_id -le ${#void_1500_urls[@]} ]; do
  unzip -o 'void_1500-'${file_id}'.zip' -d 'void_1500/data/'
  mv 'void_1500-'${file_id}'.zip' 'tmp/'
  file_id=$(( ${file_id} + 1 ))
done

cd ..
mv void_release data/void_release
python setup/setup_dataset_void.py