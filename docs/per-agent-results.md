# Per-agent results, zero-shot

Measured 2026-11-12 on Kaggle (T4), `results/2026-11-12-xtts-v2-zero-shot-all-test-lines.json`: base XTTS-v2, no fine-tune, each agent conditioned on three of its own training clips, scored on every held-out line of that agent (140 lines, 26 agents). Speaker similarity is ECAPA-TDNN cosine (higher is better), word error rate is what Whisper hears in the output (lower is better).

This is the strategy the service uses for every agent: eight fine-tunes (see the README) did not beat it, so no agent is served from a trained checkpoint. Agents marked thin have fewer than 5 held-out lines, so their row is mostly noise. Miks, Veto and Waylay are in the agent list but have no lines in the public dataset, so they are not served.

Regenerate with `python -m valorant_voice agent-table`.

| agent | held-out lines | speaker similarity | word error rate | real-time factor |
|---|---|---|---|---|
| Gekko | 4 (thin) | 0.258 | 0.104 | 0.45 |
| Phoenix | 6 | 0.262 | 0.083 | 0.43 |
| Fade | 5 | 0.267 | 0.000 | 0.47 |
| Astra | 4 (thin) | 0.289 | 0.341 | 0.44 |
| Sova | 7 | 0.304 | 0.040 | 0.44 |
| Viper | 2 (thin) | 0.315 | 0.045 | 0.46 |
| Harbor | 5 | 0.323 | 0.057 | 0.46 |
| Breach | 3 (thin) | 0.341 | 0.111 | 0.46 |
| Jett | 5 | 0.344 | 0.133 | 0.44 |
| Deadlock | 6 | 0.385 | 0.037 | 0.45 |
| Sage | 5 | 0.385 | 0.033 | 0.44 |
| Skye | 5 | 0.390 | 0.000 | 0.44 |
| Raze | 5 | 0.391 | 0.098 | 0.44 |
| Clove | 8 | 0.404 | 0.044 | 0.44 |
| Neon | 5 | 0.422 | 0.182 | 0.45 |
| Omen | 7 | 0.422 | 0.013 | 0.45 |
| Tejo | 4 (thin) | 0.433 | 0.233 | 0.45 |
| Killjoy | 6 | 0.445 | 0.044 | 0.45 |
| Cypher | 8 | 0.458 | 0.030 | 0.45 |
| Vyse | 9 | 0.473 | 0.022 | 0.44 |
| Yoru | 7 | 0.475 | 0.000 | 0.44 |
| Chamber | 6 | 0.480 | 0.017 | 0.44 |
| KAY/O | 8 | 0.489 | 0.101 | 0.44 |
| Brimstone | 3 (thin) | 0.525 | 0.171 | 0.45 |
| Reyna | 4 (thin) | 0.527 | 0.042 | 0.45 |
| Iso | 3 (thin) | 0.554 | 0.000 | 0.44 |
| **all 26** | 140 | 0.399 | 0.076 | 0.45 |
