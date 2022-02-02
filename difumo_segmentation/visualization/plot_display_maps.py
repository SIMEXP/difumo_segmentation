import os
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from nilearn.image import iter_img
from nilearn import plotting


def _get_values(maps_img, percent):
    """Get new intensities in each probabilistic map specified by
       value in percent. Useful for visualizing less non-overlapping
       between each probabilistic map.

    Parameters
    ----------
    maps_img : 4D Nifti image
        A 4D image which contains each dictionary/map in 3D.

    percent : float
        A value which is multiplied by true values in each probabilistic
        map.

    Returns
    -------
    values : list
        Values after multiplied by percent.
    """
    values = []
    for img in iter_img(maps_img):
        values.append(img.get_data().max() * percent)
    return values


root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
percent = 0.33
cmap = matplotlib.colors.ListedColormap('k', name='from_list', N=256)

components_to_display = [64, 128, 256, 512, 1024]
resolutions_to_display = [2, 3]

for i, n_components in enumerate(components_to_display):
    for res in resolutions_to_display:
        maps_img = os.path.join(root_dir, "data", "processed", "segmented_difumo_atlases", str(
            n_components), str(res), "maps.nii.gz")
        if percent is not None:
            threshold = _get_values(maps_img, percent)
        else:
            threshold = None
        display = plotting.plot_prob_atlas(maps_img,
                                           threshold=threshold, dim=0.1,
                                           draw_cross=False,
                                           cmap=cmap, linewidths=1.5)
        save_dir = join(root_dir, "reports", "imgs", "display_maps")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(join(save_dir, f"{n_components}_{res}.jpg"),
                    bbox_inches='tight')
