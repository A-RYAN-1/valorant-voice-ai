# Dataset audit

Measured 2026-10-24 on Kaggle (kernel `vv-data-audit`, version 2) from the public Hugging Face dataset
`NgThVinh/ValorantAgentVoiceLines` (26 agents). Settings: minimum clip length 1.5 s, maximum 12 s,
Whisper `small.en` check at word error rate 0.3, split by hash of the normalised text (seed 13).
Counts only; no audio or text is stored in this repository. Rerun: `scripts/kaggle/data_audit`.

| agent | fetched | kept | minutes | train | val | test | unreadable | too_short | too_long | no_text | duplicate | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| astra | 502 | 112 | 6.0 | 104 | 4 | 4 | 0 | 339 | 0 | 0 | 3 | 48 |
| breach | 484 | 113 | 5.8 | 106 | 4 | 3 | 0 | 362 | 0 | 0 | 2 | 7 |
| brimstone | 461 | 92 | 4.7 | 82 | 7 | 3 | 0 | 364 | 0 | 0 | 1 | 4 |
| chamber | 447 | 106 | 6.4 | 97 | 3 | 6 | 0 | 319 | 0 | 0 | 3 | 19 |
| clove | 461 | 109 | 7.3 | 93 | 8 | 8 | 0 | 314 | 1 | 0 | 9 | 28 |
| cypher | 480 | 118 | 7.4 | 103 | 7 | 8 | 0 | 342 | 0 | 0 | 5 | 15 |
| deadlock | 436 | 84 | 4.2 | 74 | 4 | 6 | 0 | 323 | 0 | 0 | 7 | 22 |
| fade | 440 | 95 | 5.1 | 84 | 6 | 5 | 0 | 329 | 0 | 0 | 2 | 14 |
| gekko | 563 | 74 | 3.4 | 63 | 7 | 4 | 0 | 458 | 0 | 0 | 7 | 24 |
| harbor | 432 | 104 | 5.5 | 95 | 4 | 5 | 0 | 317 | 0 | 0 | 3 | 8 |
| iso | 423 | 67 | 3.1 | 60 | 4 | 3 | 0 | 330 | 0 | 0 | 4 | 22 |
| jett | 472 | 102 | 5.3 | 90 | 7 | 5 | 0 | 358 | 0 | 0 | 2 | 10 |
| kayo | 469 | 98 | 4.5 | 84 | 6 | 8 | 0 | 368 | 0 | 0 | 1 | 2 |
| killjoy | 488 | 102 | 7.9 | 94 | 2 | 6 | 0 | 371 | 0 | 0 | 3 | 12 |
| neon | 458 | 81 | 3.8 | 73 | 3 | 5 | 0 | 360 | 0 | 0 | 2 | 15 |
| omen | 496 | 141 | 8.5 | 129 | 5 | 7 | 0 | 328 | 0 | 0 | 11 | 16 |
| phoenix | 460 | 93 | 5.1 | 82 | 5 | 6 | 0 | 354 | 0 | 0 | 2 | 11 |
| raze | 493 | 97 | 6.0 | 86 | 6 | 5 | 0 | 358 | 0 | 0 | 7 | 31 |
| reyna | 481 | 105 | 6.1 | 98 | 3 | 4 | 0 | 347 | 0 | 0 | 6 | 23 |
| sage | 474 | 113 | 6.8 | 103 | 5 | 5 | 0 | 351 | 0 | 0 | 6 | 4 |
| skye | 463 | 103 | 6.4 | 94 | 4 | 5 | 0 | 350 | 0 | 0 | 4 | 6 |
| sova | 479 | 96 | 5.2 | 84 | 5 | 7 | 0 | 376 | 1 | 0 | 3 | 3 |
| tejo | 389 | 71 | 4.1 | 63 | 4 | 4 | 0 | 298 | 0 | 0 | 5 | 15 |
| viper | 484 | 116 | 6.7 | 111 | 3 | 2 | 0 | 358 | 0 | 0 | 4 | 6 |
| vyse | 457 | 122 | 7.7 | 107 | 6 | 9 | 0 | 326 | 0 | 0 | 8 | 1 |
| yoru | 472 | 97 | 5.1 | 87 | 3 | 7 | 0 | 364 | 0 | 0 | 2 | 9 |
| **total** | 12164 | 2611 | 148.1 | 2346 | 125 | 140 | 0 | 9064 | 2 | 0 | 112 | 375 |


Split totals: train 2346, val 125, test 140.

## Where the lines go

Length of each fetched clip after trimming silence (all agents, 12,164 clips):

| length | clips |
|---|---|
| under 0.5 s | 980 |
| 0.5 to 1 s | 5623 |
| 1 to 1.5 s | 2461 |
| 1.5 to 3 s | 1644 |
| 3 to 6 s | 1216 |
| 6 to 12 s | 238 |
| over 12 s | 2 |

The game lines are mostly short barks, so the 1.5 s minimum removes 9,064 of 12,164 clips (75 percent).
Lowering the minimum to 1.0 s would add 2,461 clips; whether that helps the fine-tune is decided by the
evaluation, not assumed. About 5.7 minutes of audio remain per agent on average, which is too little
for 26 separate fine-tunes and favours one pooled multi-speaker model (VV-F4 compares both on Sage).
