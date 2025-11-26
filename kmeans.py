import numpy as np


def kmeans(points: np.ndarray, k: int, iters: int, seed: int = 42):
    """Classic single-run K-Means returning cluster counts."""
    centers = points[:k].astype(float).copy()

    for _ in range(iters):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array(
            [
                points[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                for c in range(k)
            ]
        )
        shift = np.linalg.norm(new_centers - centers)
        if shift < 1e-8 or np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    counts = [int(np.sum(labels == c)) for c in range(k)]
    return sorted(counts)


def main():
    first_line = input().strip()
    if not first_line:
        return
    k, m, n = map(int, first_line.split())
    features = []
    for _ in range(m):
        line = input().strip()
        while not line:
            line = input().strip()
        features.append(list(map(float, line.split())))
    points = np.array(features)

    counts = kmeans(points, k, n)
    print(" ".join(map(str, counts)))


if __name__ == "__main__":
    main()
"""
另一个题 
输入是：

第一行：k m n 
k代表聚类个数 m终端数量 n迭代次数
第2行~第m+1行：m个终端的特征向量

第2行~第m+1行：每行四列 代表四个特征 




给出输入
3 20 1000
0.11 0.79 0.68 0.97
1.0 0.8 0.13 0.33
0.27 0.02 0.5 0.46
0.83 0.29 0.23 0.75
0.97 0.08 0.84 0.55
0.29 0.71 0.17 0.83
0.03 0.6 0.88 0.28
0.24 0.26 0.82 0.03
0.96 0.12 0.82 0.36
0.13 0.12 0.86 0.44
0.23 0.7 0.35 0.06
0.42 0.49 0.67 0.84
0.8 0.49 0.47 0.7
0.68 0.03 0.11 0.07
0.77 0.19 0.95 0.44
0.25 0.12 0.98 0.04
0.7 0.11 0.53 0.3
0.73 0.67 0.46 0.96
0.11 0.31 0.91 0.57
0.43 0.61 0.13 0.1

输出是
4 6 10

输出k款终端数量 从小到大排序


输入
4 32 800
0.73 0.96 0.2 0.53
0.01 0.19 0.42 0.46
0.27 0.24 0.87 0.8
0.97 0.77 0.42 0.04
0.41 0.69 0.96 0.56
0.27 0.4 0.56 0.56
0.28 0.04 0.74 0.82
0.17 0.2 0.95 0.1
0.2 0.1 0.14 0.93
0.86 0.59 0.42 0.52
0.35 0.77 0.37 0.08
0.52 0.48 0.16 0.56
0.59 0.97 0.21 0.05
0.67 0.94 0.28 0.08
0.09 0.65 0.55 1.
0.77 0.14 0.35 0.01
0.02 0.18 0.72 0.26
0.71 0.78 0.86 0.11
0.54 0.02 0.75 0.2
0.15 0.76 0.59 0.23
0.71 0.66 0.43 0.32
0.17 0.57 0.53 0.42
0.04 0.34 0.66 0.28
0.79 0.14 0.11 0.6
0.04 0.48 0.05 0.04
0.62 0.43 0.28 0.6
0.47 0.13 0.35 0.17
0.9 0.82 0.97 0.71
0.99 0.53 0.24 0.56
0.83 0.44 0.7 0.4
0.71 0.45 0.64 0.53
0.6 0.54 0.86 0.11

输出：
6 8 9 9


"""
