#!/usr/bin/env python
# coding: utf-8

# # Latin Hypercube Sample
# 
# 実装は難しい物ではありませんが、コンセプトがとても面白いと思いました。
# 
# アルゴリズムの詳細は[こちら](https://goobitgitkome.github.io/my-portfolio/Optimization/Latin%20Hypercube%20Sampling%20%28LHS%29%20%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E6%9C%89%E7%94%A8%E6%80%A7/)

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# In[ ]:


# Parameter Setup
size = 20
grid_range = (0, 1)
cmap = ListedColormap(['#f8f9fa', '#e63946']) 


# In[ ]:


def generate_lhs(n):
    rng = np.random.default_rng()
    idx_x = rng.permutation(n)
    idx_y = rng.permutation(n)

    # Normalized coordinates + jitter
    points_x = (idx_x + rng.uniform(size=n)) / n
    points_y = (idx_y + rng.uniform(size=n)) / n

    # print(idx_x)
    # print(rng.uniform(size=n))

    grid = np.zeros((n, n))
    grid[idx_y, idx_x] = 1 

    return grid, (points_x, points_y)


# In[ ]:


# Generate lhs
region_grid, (data_x, data_y) = generate_lhs(size)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", facecolor='white')

# Shared styling parameters
grid_style = dict(color='black', linestyle='-', linewidth=0.8, alpha=0.15)
ticks = np.linspace(*grid_range, size + 1)

# Plot heatmap
ax1.imshow(region_grid, cmap=cmap, origin='lower', extent=[*grid_range, *grid_range])
ax1.set_title("Latin Hypercube Sampling Grid", pad=15, weight='bold')

# Plot coordinates
ax2.scatter(data_x, data_y, s=10, c='#e63946', edgecolors='#a8222d', linewidth=1.5, zorder=3)
ax2.set_title("Latin Hypercube Smpling Plots", pad=15, weight='bold')

# Unified Axis Formatting
for ax in (ax1, ax2):
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)
    ax.set_aspect('equal')
    ax.grid(True, **grid_style)

plt.show()


# ## References

# [1] A. B. Owen, “Latin hypercube sampling,” in Monte Carlo theory, methods and examples, 2023, ch. 10.8, pp. 8–12. [Online]. Available: https://artowen.su.domains/mc/ (accessed Feb. 14, 2026).
# 
# 
# [2] “Latin Hypercube Sampling Overview,” @emergentmind, 2025. https://www.emergentmind.com/topics/latin-hypercube-sampling-lhs (accessed Feb. 14, 2026).
# ‌
# 
# [3] M. Giles, “Numerical Methods II.” Accessed: Feb. 14, 2026. [Online]. Available: https://people.maths.ox.ac.uk/gilesm/mc/mc/lec4.pdf
# 
# 
# [4] “金融工学でのモンテカルロ法(11/23)：層別化法,” 「大人の教養・知識・気付き」を伸ばすブログ, Oct. 06, 2021. https://power-of-awareness.com/entry/2021/10/07/050000#631%E3%83%A9%E3%83%86%E3%83%B3%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%82%AD%E3%83%A5%E3%83%BC%E3%83%96%E6%B3%95 (accessed Feb. 14, 2026).
# ‌
# 
# [5] nabenabe0928, “高次元乱数ベクトル生成〜Latin Hypercube Sampling・ラテン超方格サンプリング〜,” Qiita, Feb. 03, 2020. https://qiita.com/nabenabe0928/items/906ce2ad511833609b38#fn-1 (accessed Feb. 14, 2026).
# ‌
# ‌

# 
