# Data plan (298A Cycles 1–2)

Owner: Bavishna  
Related Linear issue: personal-data sandbox + PinchBench 23-task sources.

## Sources we will use

1. **PinchBench Table 1 set (23 tasks)**  
   Public agent tasks (calendar, email, files, spreadsheet, etc.).  
   List: `baseline-reproduction/configs/pinchbench_table1_23.txt`  
   Licence: follow PinchBench / OpenJarvis repo terms. Used only to reproduce Saad-Falcon et al. (2026) Table 1.

2. **OpenJarvis evaluation assets**  
   Code: https://github.com/open-jarvis/OpenJarvis  
   Paper: https://arxiv.org/abs/2605.17172  
   Role: published baseline, not one of our four models.

3. **30-day simulated personal traces (team-built)**  
   Not collected yet. Consented sandbox of fake users (calendar, messages, files).  
   No real Gmail/iCloud of team members.  
   Used for Topic 17 metrics: task success, intervention rate, preference regret, deletion compliance, privacy leakage, p95 latency, cost per user-day.

## Splits (to lock at Team Meeting 1)

Train / validation / test on the 30-day traces will be versioned. PinchBench 23-task eval is a fixed public set, not mixed into our train split.

## Status 2026-09-24

PinchBench source obtained (cloned via OpenJarvis cache).  
30-day sandbox: planning only. No real personal data stored in this repo.
