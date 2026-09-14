# DeiT Interaction Router vs Oracle Diagnostic

- Checkpoint: `local checkpoint (not included in this repository)`
- Fixed evaluation subset: first `128` rows of the local test split
- Objective: CE + `1.0` * logit-KD, evaluated over every legal route per budget.
- `loss oracle` chooses the minimum objective route per image. `any-correct oracle` chooses whether any legal route predicts the correct label.

| Budget | Router acc | Loss-oracle acc | Any-correct oracle acc | Best fixed acc | Router/oracle match | Oracle mean router rank | Router mean loss rank | Score/-loss corr | Router max share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4/12 | 0.2266 | 0.5859 | 0.6562 | 0.3359 | 0.1172 | 12.34 | 14.86 | 0.5142 | 1.0000 |

4/12 best fixed by objective: `[0, 4, 7, 11]`; best fixed by accuracy: `[0, 3, 7, 11]`.
| 6/12 | 0.5781 | 0.8516 | 0.8906 | 0.6328 | 0.0156 | 54.81 | 58.93 | 0.4985 | 0.5391 |

6/12 best fixed by objective: `[0, 3, 4, 6, 8, 11]`; best fixed by accuracy: `[0, 4, 5, 7, 8, 11]`.
| 7/12 | 0.6562 | 0.8906 | 0.9297 | 0.6875 | 0.0234 | 65.74 | 78.30 | 0.4133 | 1.0000 |

7/12 best fixed by objective: `[0, 3, 4, 6, 8, 9, 11]`; best fixed by accuracy: `[0, 1, 4, 5, 7, 8, 11]`.
