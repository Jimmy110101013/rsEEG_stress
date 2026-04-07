# Results Index

Auto-generated from `scripts/build_results_index.py`. Source of truth: each run's `config.json` + `summary.json`.

**Total runs:** 16  |  **Rows:** 24

Flat CSV companion: `results/results_index.csv`.

## Leaderboard (best subject_bal_acc per paradigm × dataset × extractor)

| paradigm | dataset | extractor | mode | best bal_acc | run_id | model |
|---|---|---|---|---|---|---|
| classical | stress | — | classical | **0.6660** | `20260405_1713_classical_subject-dass` | RF |
| cross-dataset | adftd | labram | ft | **0.7521** | `20260406_0935_ft_dass_aug75_labram_adftd_feat` | — |
| cross-dataset | tdbrain | labram | ft | **0.6812** | `20260407_1533_ft_dass_aug75_labram_tdbrain_feat` | — |
| subject-level | stress | cbramod | ft | **0.4880** | `20260404_1633_ft_subjectdass_aug75_cbramod` | — |
| subject-level | stress | labram | ft | **0.6559** | `20260405_1426_ft_subjectdass_aug75_labram` | — |
| subject-level | stress | reve | ft | **0.5529** | `20260404_1633_ft_subjectdass_aug75_reve` | — |
| trial-level | stress | cbramod | ft | **0.7117** | `20260404_2108_trial_ft_subjectdass_aug75_cbramod` | — |
| trial-level | stress | labram | ft | **0.8619** | `20260404_1714_trial_ft_subjectdass_aug75_labram` | — |
| trial-level | stress | reve | ft | **0.7697** | `20260405_0114_trial_ft_subjectdass_aug75_reve` | — |

## classical · stress

| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260405_1713_classical_subject-dass` | — | classical | LogReg_L1 | — | 0.5153 | 0.5429 | 0.5536 | 0.0286 | 70 | complete |
| `20260405_1713_classical_subject-dass` | — | classical | LogReg_L2 | — | 0.6235 | 0.6286 | 0.6388 | 0.2268 | 70 | complete |
| `20260405_1713_classical_subject-dass` | — | classical | RF | — | 0.6660 | 0.6857 | 0.6913 | 0.3180 | 70 | complete |
| `20260405_1713_classical_subject-dass` | — | classical | SVM_RBF | — | 0.5800 | 0.6000 | 0.6094 | 0.1500 | 70 | complete |
| `20260405_1713_classical_subject-dass` | — | classical | XGBoost | — | 0.6115 | 0.6571 | 0.6571 | 0.2229 | 70 | complete |
| `20260405_1714_classical_subject-dass` | — | classical | LogReg_L1 | — | 0.4790 | 0.4756 | 0.4762 | -0.0413 | 82 | complete |
| `20260405_1714_classical_subject-dass` | — | classical | LogReg_L2 | — | 0.4742 | 0.4756 | 0.4768 | -0.0513 | 82 | complete |
| `20260405_1714_classical_subject-dass` | — | classical | RF | — | 0.4225 | 0.4268 | 0.4274 | -0.1546 | 82 | complete |
| `20260405_1714_classical_subject-dass` | — | classical | SVM_RBF | — | 0.5162 | 0.5244 | 0.5220 | 0.0327 | 82 | complete |
| `20260405_1714_classical_subject-dass` | — | classical | XGBoost | — | 0.4670 | 0.4756 | 0.4730 | -0.0665 | 82 | complete |

## cross-dataset · adftd

| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260406_0935_ft_dass_aug75_labram_adftd_feat` | labram | ft | — | 5 | 0.7521 | 0.7538 | 0.7541 | 0.5030 | 195 | complete |

## cross-dataset · tdbrain

| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260407_1533_ft_dass_aug75_labram_tdbrain_feat` | labram | ft | — | 5 | 0.6812 | 0.8556 | 0.8562 | 0.3592 | 734 | complete |

## subject-level · stress

| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260404_1633_ft_subjectdass_aug75_cbramod` | cbramod | ft | — | 5 | 0.4880 | 0.4512 | 0.3647 | -0.0222 | 82 | complete |
| `20260404_1633_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.5369 | 0.5366 | 0.5377 | 0.0732 | 82 | complete |
| `20260404_1633_ft_subjectdass_aug75_reve` | reve | ft | — | 5 | 0.5529 | 0.5488 | 0.5493 | 0.1040 | 82 | complete |
| `20260405_0123_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.4393 | 0.4268 | 0.4178 | -0.1171 | 82 | complete |
| `20260405_0450_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.5640 | 0.5610 | 0.5618 | 0.1261 | 82 | complete |
| `20260405_1333_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.5108 | 0.5000 | 0.4951 | 0.0210 | 82 | complete |
| `20260405_1426_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.6559 | 0.6571 | 0.6666 | 0.2863 | 70 | complete |
| `20260405_1427_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.5370 | 0.5571 | 0.5686 | 0.0687 | 70 | complete |
| `20260406_0419_ft_subjectdass_aug75_labram_feat` | labram | ft | — | 5 | 0.6559 | 0.6571 | 0.6666 | 0.2863 | 70 | complete |

## trial-level · stress

| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260404_1714_trial_ft_subjectdass_aug75_labram` | labram | ft | — | 5 | 0.8619 | 0.8537 | 0.8537 | 0.7101 | 82 | complete |
| `20260404_2108_trial_ft_subjectdass_aug75_cbramod` | cbramod | ft | — | 5 | 0.7117 | 0.7073 | 0.7078 | 0.4174 | 82 | complete |
| `20260405_0114_trial_ft_subjectdass_aug75_reve` | reve | ft | — | 5 | 0.7697 | 0.7683 | 0.7688 | 0.5355 | 82 | complete |
