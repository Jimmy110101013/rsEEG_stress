# Table 1 — Excluded benchmarks

**Purpose**: documentation of benchmarks deliberately NOT included in Table 1, with exclusion reason. Referenced from paper §2 as support for the "split protocols are inconsistent" critique.

---

## Excluded because split type ambiguous or trial-level in source

| Benchmark | Source | Reported split | Why excluded |
|---|---|---|---|
| ADFTD (AD/FTD/CN, 3-class) | EEG-FM-Bench | Stratified 70/15/15 (no subject-disjoint guarantee) | Stratified → subjects can appear in all splits → trial-like leakage |
| TUEV (event types, 6-class) | EEG-FM-Bench | Stratified 80/10/10 | Same as ADFTD-3cls |
| Mimul-11 (motor imagery, 3-class) | EEG-FM-Bench | Stratified 76/12/12 | Same |
| MentalArithmetic / Workload | EEG-FM-Bench | Stratified 72/14/14 | Same |
| TUSL (slowing, 3-class) | EEG-FM-Bench | Stratified | Same |
| SEED (emotion, 3-class) | EEG-FM-Bench | Subject-dependent (15 trials split 9:3:3, all sessions merged) | Subject-dependent means trials from one subject appear in train AND test — trial-level leakage |
| SEED-V (emotion, 5-class) | EEG-FM-Bench | Subject-dependent 1:1:1 | Same |
| SEED-V (emotion, 5-class) | CBraMod own paper §D.1 | Trial 5:5:5 subject-dependent | Same |
| Mumtaz2016 (MDD) | CBraMod own paper | Split not stated | Ambiguous |
| ISRUC, CHB-MIT, BCIC2020-3, MentalArith, TUEV, TUAB (in CBraMod paper) | CBraMod own paper D.1 | Datasets documented but split protocol not | Ambiguous — use EEG-FM-Bench's numbers instead |
| REVE Table 2 all rows (TUAB, TUEV, PhysioNet-MI, BCIC-IV-2a, FACED, Mumtaz, MAT, BCIC2020-3, ISRUC, HMC) | REVE paper | "Adhering to the same train/val/test splits used in earlier studies" — not quoted per dataset | Ambiguous provenance; use EEG-FM-Bench's numbers instead where available |
| REVE Table 4 (Linear Probing) | REVE paper | Same as REVE T2 | Ambiguous |

---

## Key point for §2.2 methods discussion

Of the **14 benchmark datasets** documented by the three major FM papers collectively:
- **8 are reported under subject-level split** (explicit subject-disjoint train/val/test): TUAB, Siena, PhysioMI, BCIC-IV-2a, HMC, SEED-VII, Things-EEG-2, FACED
- **6 are reported under stratified or subject-dependent split** (potential trial-level leakage): ADFTD-3cls, TUEV, Mimul-11, MentalArithmetic, TUSL, SEED / SEED-V

Most EEG-FM benchmark papers report these 14 datasets in a single summary table without flagging the mixed split protocols. A reader seeing "LaBraM gets 61.6% on SEED" next to "LaBraM gets 80.0% on TUAB" may not realize the former is subject-dependent (trial-like) and the latter is subject-level.

**This is directly the critique we make in §1.**
