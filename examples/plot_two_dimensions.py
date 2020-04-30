import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import matplotlib.tri as mtri
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import cmasher as cmr
import seaborn as sns
import matplotlib as mplt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams["font.size"] = 20.
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"


file = 'example_results_test.pkl'
tmp = pickle.load(open(os.path.join(file), 'rb'))
df = pd.DataFrame(tmp)
df = df.fillna(0)

df = pd.DataFrame(tmp)
df = df.fillna(0)

x = np.array(df['W_EDGE'])
y = np.array(df['SYMP_PROB'])
z = np.array(df['reff'])
z_1 = np.array([1] * len(z))

triang = mtri.Triangulation(x, y)

xi, yi = np.meshgrid(np.linspace(min(x), max(x), 20), np.linspace(min(y), max(y), 20))

# # Triangular interpolation
# interp_lin = mtri.LinearTriInterpolator(triang, z)
# zi_lin = interp_lin(xi, yi)

# Gaussian smoothing
X = np.vstack((x, y)).T
Y = z
kernel = 1.0 * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, Y)
zi_lin = gp.predict(np.vstack((xi.flatten(), yi.flatten())).T).reshape(xi.shape)

### use single cmap
# cmap = sns.cm.rocket
# colormap = sns.cm.rocket

### use two colormaps
# cmap1 = sns.cm.rocket
# cmap2 = sns.cm.mako_r
# new_cmap_name = 'rocket_mako'

# cmap1 = sns.cm.rocket
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'rocket_freeze'

# cmap1 = plt.get_cmap('cmr.ember')
# cmap2 = plt.get_cmap('cmr.lavender_r')
# new_cmap_name = 'ember_lavender'

# cmap1 = plt.get_cmap('cmr.gem')
# cmap2 = sns.cm.mako_r
# new_cmap_name = 'gem_mako'

# cmap1 = plt.get_cmap('cmr.gem')
# cmap2 = cmr.cm.sunburst_r
# new_cmap_name = 'gem_sunburst'

# cmap1 = sns.cm.rocket
# cmap2 = cmr.cm.ocean_r
# new_cmap_name = 'rocket_ocean'

# cmap1 = cmr.cm.flamingo
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'flamingo_freeze'

# cmap1 = mplt.cm.magma
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'magma_freeze'

# cmap1 = mplt.cm.magma
# cmap2 = mplt.cm.YlGnBu
# new_cmap_name = 'magma_YlGnBu'

# cmap1 = mplt.cm.Oranges_r
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'Oranges_freeze'r

# cmap1 = mplt.cm.OrRd_r
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'OrRd_freeze'

# cmap1 = mplt.cm.plasma
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'plasma_freeze'

# cmap1 = mplt.cm.RdPu_r
# cmap2 = cmr.cm.freeze_r
# new_cmap_name = 'RdPu_freeze'

# cmap1 = cmr.cm.heat
cmap2 = cmr.cm.freeze_r
new_cmap_name = 'heat_freeze'

colors1 = cmap1(np.linspace(0., 1, 128))  # to truncate darker end of cmap1 change 0 to a value greater than 0, less than 1
colors2 = cmap2(np.linspace(0., 1, 128))  # to truncate darker end of cmap2 change 1 to a value less than 1, greater than 0

transition_steps = 0  # heat+freeze
# transition_steps = 4 # increase if closest ends of the color maps are far apart, values to try: 4, 8, 16
transition = mplt.colors.LinearSegmentedColormap.from_list("transition", [cmap1(1.), cmap2(0)])(np.linspace(0,1,transition_steps))
colors = np.vstack((colors1, transition, colors2))
colors = np.flipud(colors)


new_cmap = mplt.colors.LinearSegmentedColormap.from_list(new_cmap_name, colors)

fig = plt.figure(figsize=(8, 6), facecolor=colors[0])

im=plt.contourf(xi, yi, zi_lin, cmap=new_cmap, levels=np.linspace(0., 2., 50))
plt.xlim([min(x), max(x)])
plt.ylim([min(y), max(y)])
plt.ylabel('Percent Symptomatic')
plt.xlabel('R0')

plt.title('No work tracing')

cs = plt.contour(xi, yi, zi_lin, levels=[0, 1], colors='k', linestyles='dashed', linewidths=3)
plt.clabel(cs, fmt=r'$R_{e}=%2.1d$', colors='k', fontsize=18)

plt.subplots_adjust(right=0.9, top=0.9, left=0.15, bottom=0.15)
cbar = plt.colorbar(im, ticks=np.linspace(0, 2.0, 9))
cbar.ax.set_title('$R_{e}$', rotation=0)
plt.savefig('figures/test_' + new_cmap_name + '.png')