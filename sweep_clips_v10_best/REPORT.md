# Sampling sweep — /ckpts/v10_dac_2b/best.pt

Phase 1: 25s clips, 2 seed(s)/cell, CLAP+librosa auto-score. Phase 2 finalists: 60s.

**The auto-ranking is a coarse filter (collapse/rhythm + genre-match). Pick the final default BY EAR from `phase2_finalists/`.** `lyric_cfg` was NOT auto-scored — judge it by ear in the lyric finalists.

## lyric

| rank | profile | cfg | cross-genre | per-genre (rank_score) | avg CLAP |
|---|---|---|---|---|---|
| 1 ⭐ | user_ladder | 7 | 0.088 | pop=0.07 hiph=0.29 elec=0.10 indi=0.07 | 0.135 |
| 2 ⭐ | open_flat | 7 | 0.078 | pop=0.07 hiph=0.28 elec=0.14 indi=0.02 | 0.245 |
| 3 | tight_ladder | 2 | 0.066 | pop=0.11 hiph=0.04 elec=0.06 indi=0.10 | 0.214 |
| 4 | low_topp | 2 | 0.059 | pop=0.05 hiph=0.10 elec=0.06 indi=0.06 | 0.178 |
| 5 | user_ladder | 5 | 0.037 | pop=0.07 hiph=0.00 elec=0.06 indi=0.10 | 0.261 |
| 6 | mid_topp | 1 | 0.036 | pop=0.12 hiph=0.00 elec=0.00 indi=0.18 | 0.149 |
| 7 | mid_topp | 2 | 0.034 | pop=0.00 hiph=0.00 elec=0.22 indi=0.09 | 0.158 |
| 8 | default_ladder | 2 | 0.033 | pop=0.04 hiph=0.00 elec=0.35 indi=0.03 | 0.197 |
| 9 | mid_topp | 7 | 0.032 | pop=0.00 hiph=0.03 elec=0.23 indi=0.05 | 0.236 |
| 10 | default_ladder | 1 | 0.031 | pop=0.04 hiph=0.05 elec=0.00 indi=0.16 | 0.179 |
| 11 | default_ladder | 7 | 0.030 | pop=0.00 hiph=0.10 elec=0.06 indi=0.04 | 0.152 |
| 12 | open_flat | 3 | 0.026 | pop=0.03 hiph=0.05 elec=0.00 indi=0.10 | 0.256 |
| 13 | open_flat | 2 | 0.024 | pop=0.03 hiph=0.09 elec=0.04 indi=0.00 | 0.184 |
| 14 | user_ladder | 3 | 0.023 | pop=0.07 hiph=0.00 elec=0.09 indi=0.01 | 0.186 |
| 15 | tight_ladder | 3 | 0.019 | pop=0.04 hiph=0.01 elec=0.12 indi=0.00 | 0.167 |
| 16 | user_ladder | 1 | 0.019 | pop=0.00 hiph=0.07 elec=0.00 indi=0.08 | 0.289 |
| 17 | tight_ladder | 1 | 0.012 | pop=0.01 hiph=0.00 elec=0.24 indi=0.00 | 0.223 |
| 18 | open_flat | 1 | 0.012 | pop=0.00 hiph=0.00 elec=0.03 indi=0.08 | 0.201 |
| 19 | user_ladder | 2 | 0.011 | pop=0.03 hiph=0.00 elec=0.00 indi=0.08 | 0.176 |
| 20 | low_topp | 1 | 0.011 | pop=0.00 hiph=0.00 elec=0.03 indi=0.05 | 0.133 |
| 21 | default_ladder | 3 | 0.010 | pop=0.01 hiph=0.04 elec=0.02 indi=0.00 | 0.257 |
| 22 | low_topp | 3 | 0.009 | pop=0.00 hiph=0.00 elec=0.03 indi=0.05 | 0.200 |
| 23 | mid_topp | 5 | 0.008 | pop=0.01 hiph=0.00 elec=0.01 indi=0.08 | 0.235 |
| 24 | open_flat | 5 | 0.008 | pop=0.00 hiph=0.00 elec=0.03 indi=0.03 | 0.179 |
| 25 | default_ladder | 5 | 0.007 | pop=0.00 hiph=0.03 elec=0.03 indi=0.00 | 0.336 |
| 26 | low_topp | 7 | 0.004 | pop=0.00 hiph=0.00 elec=0.01 indi=0.08 | 0.211 |
| 27 | tight_ladder | 7 | 0.003 | pop=0.00 hiph=0.00 elec=0.09 indi=0.00 | 0.211 |
| 28 | low_topp | 5 | 0.002 | pop=0.00 hiph=0.00 elec=0.00 indi=0.04 | 0.155 |
| 29 | mid_topp | 3 | 0.001 | pop=0.00 hiph=0.00 elec=0.00 indi=0.03 | 0.229 |
| 30 | tight_ladder | 5 | 0.000 | pop=0.00 hiph=0.01 elec=0.00 indi=0.00 | 0.245 |

**Auto-recommended lyric default:** `user_ladder` @ cfg 7 (⭐ = rendered at 60s in `phase2_finalists/`).

## instrumental

| rank | profile | cfg | cross-genre | per-genre (rank_score) | avg CLAP |
|---|---|---|---|---|---|
| 1 ⭐ | user_ladder | 7 | 0.104 | pop=0.00 hiph=0.33 elec=0.38 indi=0.04 | 0.192 |
| 2 ⭐ | default_ladder | 1 | 0.097 | pop=0.22 hiph=0.00 elec=0.37 indi=0.08 | 0.241 |
| 3 | user_ladder | 2 | 0.079 | pop=0.03 hiph=0.03 elec=0.21 indi=0.23 | 0.133 |
| 4 | default_ladder | 5 | 0.058 | pop=0.00 hiph=0.12 elec=0.31 indi=0.04 | 0.239 |
| 5 | tight_ladder | 2 | 0.047 | pop=0.04 hiph=0.15 elec=0.03 indi=0.06 | 0.188 |
| 6 | tight_ladder | 5 | 0.042 | pop=0.02 hiph=0.13 elec=0.02 indi=0.09 | 0.281 |
| 7 | default_ladder | 2 | 0.041 | pop=0.00 hiph=0.07 elec=0.25 indi=0.04 | 0.301 |
| 8 | low_topp | 3 | 0.039 | pop=0.04 hiph=0.07 elec=0.07 indi=0.02 | 0.161 |
| 9 | mid_topp | 5 | 0.036 | pop=0.00 hiph=0.12 elec=0.18 indi=0.00 | 0.240 |
| 10 | mid_topp | 1 | 0.029 | pop=0.17 hiph=0.02 elec=0.05 indi=0.00 | 0.163 |
| 11 | default_ladder | 7 | 0.028 | pop=0.00 hiph=0.11 elec=0.12 indi=0.00 | 0.212 |
| 12 | open_flat | 5 | 0.026 | pop=0.27 hiph=0.00 elec=0.02 indi=0.03 | 0.228 |
| 13 | mid_topp | 7 | 0.026 | pop=0.06 hiph=0.07 elec=0.03 indi=0.00 | 0.178 |
| 14 | mid_topp | 2 | 0.025 | pop=0.00 hiph=0.04 elec=0.14 indi=0.03 | 0.188 |
| 15 | open_flat | 3 | 0.022 | pop=0.00 hiph=0.00 elec=0.10 indi=0.08 | 0.279 |
| 16 | user_ladder | 3 | 0.022 | pop=0.09 hiph=0.05 elec=0.02 indi=0.00 | 0.207 |
| 17 | tight_ladder | 3 | 0.021 | pop=0.02 hiph=0.00 elec=0.36 indi=0.01 | 0.191 |
| 18 | user_ladder | 1 | 0.018 | pop=0.00 hiph=0.07 elec=0.08 indi=0.00 | 0.131 |
| 19 | low_topp | 7 | 0.018 | pop=0.00 hiph=0.00 elec=0.11 indi=0.05 | 0.245 |
| 20 | default_ladder | 3 | 0.017 | pop=0.20 hiph=0.00 elec=0.00 indi=0.03 | 0.321 |
| 21 | tight_ladder | 1 | 0.016 | pop=0.04 hiph=0.09 elec=0.00 indi=0.00 | 0.156 |
| 22 | open_flat | 1 | 0.014 | pop=0.00 hiph=0.01 elec=0.04 indi=0.05 | 0.239 |
| 23 | low_topp | 2 | 0.010 | pop=0.02 hiph=0.00 elec=0.14 indi=0.00 | 0.147 |
| 24 | low_topp | 1 | 0.009 | pop=0.00 hiph=0.02 elec=0.12 indi=0.00 | 0.167 |
| 25 | open_flat | 2 | 0.005 | pop=0.00 hiph=0.13 elec=0.00 indi=0.00 | 0.201 |
| 26 | open_flat | 7 | 0.005 | pop=0.00 hiph=0.00 elec=0.14 indi=0.00 | 0.260 |
| 27 | low_topp | 5 | 0.005 | pop=0.00 hiph=0.00 elec=0.14 indi=0.00 | 0.159 |
| 28 | user_ladder | 5 | 0.001 | pop=0.00 hiph=0.00 elec=0.03 indi=0.00 | 0.278 |
| 29 | tight_ladder | 7 | 0.001 | pop=0.00 hiph=0.00 elec=0.01 indi=0.00 | 0.242 |
| 30 | mid_topp | 3 | 0.000 | pop=0.00 hiph=0.00 elec=0.00 indi=0.00 | 0.167 |

**Auto-recommended instrumental default:** `user_ladder` @ cfg 7 (⭐ = rendered at 60s in `phase2_finalists/`).

## Listening guide

1. Open `phase2_finalists/<genre>/<mode>/` and A/B the ⭐ profiles per genre.
2. For lyric cells, compare `__lcfg0/3/6` — pick the lyric_cfg where words are clearest without the music degrading.
3. Pick the profile+cfg that holds up across ALL four genres (a robust default).

## Apply the winner

Set the chosen values as defaults in:
- `server/main.py` `/generate` endpoint — `per_cb_temperature`, `per_cb_top_k`, `per_cb_top_p`/`top_p`, `cfg_scale`, `lyric_cfg_scale`.
- the webapp 'advanced' panel defaults.

Profile definitions (temperature, top_k, top_p):

- `default_ladder`: temp=[1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42], top_k=[120, 90, 70, 50, 36, 26, 18, 12, 8], top_p=0.95
- `user_ladder`: temp=[0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3], top_k=[120, 90, 70, 50, 36, 26, 18, 12, 8], top_p=0.95
- `tight_ladder`: temp=[0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25], top_k=[100, 70, 50, 36, 26, 18, 12, 8, 6], top_p=0.95
- `low_topp`: temp=[1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42], top_k=[120, 90, 70, 50, 36, 26, 18, 12, 8], top_p=0.44
- `mid_topp`: temp=[1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42], top_k=[120, 90, 70, 50, 36, 26, 18, 12, 8], top_p=0.7
- `open_flat`: temp=0.95, top_k=80, top_p=0.95
