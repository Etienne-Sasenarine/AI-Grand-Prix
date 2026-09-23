# The two racing checkpoints the paper reports

Pulled from the RunPod chains on 21 Sep 2026 (skrl `best_agent.pt`, skrl's own
pick by mean total reward). Both are v2 contract (55 x 32 = 1760 inputs),
trained at **40 Hz**, so run them with `AIGP_POLICY_HZ=40`. They predate the
`contract.json` stamping in `train.py`, so `play.py` cannot verify their hash;
the contract they trained under is the v2 defined in `contract/` at commit
`338c689` or later, which has not changed since.

| file | chain | training metric | scored under |
|---|---|---|---|
| `race40_best_agent.pt` | race40, leg of 21 Sep 01:19 | 15.27 gates/episode | perfect corners |
| `race40drop_best_agent.pt` | race40drop, leg of 21 Sep 07:51 | 10.23 gates/episode | corners dropping (`AIGP_KP_DROP=0.13 AIGP_KP_STICKY=0.81`) |

Under the same measured dropout from the competition pad, race40drop scored
9.31 gates to race40's 4.62 (the one head-to-head recorded). Full provenance,
including the ratchet-cycling failure that ended the race40 chain, is in
`docs/TRAINING_RESULTS.md`.

Recording either one, chase view beside the onboard camera, is
`scripts/diag/record_race_pov.py`; on the Windows laptop,
`scripts/windows/record_race_pov.ps1` does the whole job.
