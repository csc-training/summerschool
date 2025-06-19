# Scalability test

| GCDs | Runtime (s) | Resource cost (GCD-s) | Time spent in communication (s) | Time spent in computing (s) | Speedup | Parallel efficiency |
| ---: | ----------: | --------------------: | ------------------------------: | --------------------------: | ------: | ------------------: |
|   1  |       9.59  |                 9.59  |                           0.00  |                       9.41  |      -  |                  -  |
|   2  |       5.07  |                10.14  |                           0.18  |                       4.79  |   1.89  |                95%  |
|   4  |       2.76  |                11.04  |                           0.21  |                       2.47  |   3.47  |                87%  |
|   8  |       1.65  |                13.20  |                           0.29  |                       1.29  |   5.81  |                73%  |
|  16  |       1.10  |                17.60  |                           0.31  |                       0.74  |   8.72  |                54%  |
|  32  |       0.86  |                27.52  |                           0.37  |                       0.42  |  11.15  |                35%  |

Note that the numbers vary a bit due to random factors (e.g., other load on the node etc).
Statistics would need to be collected if accurate values would be needed.

