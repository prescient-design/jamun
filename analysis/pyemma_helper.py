from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from pyemma.plots.plots2d import get_histogram, plot_map, _to_free_energy


def compute_2D_histogram(xall: np.ndarray, yall: np.ndarray, weights=None, nbins=100, avoid_zero_count=False):
    """Compute a 2D histogram from scattered data."""
    return get_histogram(xall, yall, nbins=nbins, weights=weights, avoid_zero_count=avoid_zero_count)


def compute_1D_histogram(xyzall: np.ndarray, n_bins: int = 50) -> Dict[str, np.ndarray]:
    """Compute a 1D histogram from scattered data."""
    all_hists = []
    all_edges = []

    for coordinate in xyzall.T:
        hist, edges = np.histogram(coordinate, bins=n_bins)
        all_hists.append(hist)
        all_edges.append(edges)

    all_hists = np.array(all_hists)
    all_edges = np.array(all_edges)
    return {
        "histograms": all_hists,
        "edges": all_edges,
    }


def plot_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ax=None,
    ncontours: int = 100,
    minener_zero=True,
    kT=1.0,
    vmin=None,
    vmax=None,
    cmap="nipy_spectral",
    cbar=True,
    cbar_label="free energy / kT",
    cax=None,
    levels=None,
    legacy=True,
    cbar_orientation="vertical",
    **kwargs,
):
    """Plot a two-dimensional free energy map using a histogram of
    scattered data.

    Parameters
    ----------
    x : ndarray(nbins, nbins)
        The bins' x-coordinates in meshgrid format.
    y : ndarray(nbins, nbins)
        The bins' y-coordinates in meshgrid format.
    z : ndarray(nbins, nbins)
        Histogram counts in meshgrid format.
    weights : ndarray(T), optional, default=None
        Sample weights; by default all samples have the same weight.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
        Number of contour levels.
    nbins : int, optional, default=100
        Number of histogram bins used in each dimension.
    ncontours : int, optional, default=100
        Number of contour levels.
    avoid_zero_count : bool, optional, default=False
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.
    minener_zero : boolean, optional, default=True
        Shifts the energy minimum to zero.
    kT : float, optional, default=1.0
        The value of kT in the desired energy unit. By default,
        energies are computed in kT (setting 1.0). If you want to
        measure the energy in kJ/mol at 298 K, use kT=2.479 and
        change the cbar_label accordingly.
    vmin : float, optional, default=None
        Lowest free energy value to be plotted.
        (default=0.0 in legacy mode)
    vmax : float, optional, default=None
        Highest free energy value to be plotted.
    cmap : matplotlib colormap, optional, default='nipy_spectral'
        The color map to use.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cbar_label : str, optional, default='free energy / kT'
        Colorbar label string; use None to suppress it.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    legacy : boolean, optional, default=True
        Switch to use the function in legacy mode (deprecated).
    ncountours : int, optional, default=None
        Legacy parameter (typo) for number of contour levels.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    """
    if legacy:
        warn(
            "Legacy mode is deprecated is will be removed in the next major release. Until then use legacy=False",
            DeprecationWarning,
        )
        if vmin is None:
            vmin = 0.0

    f = _to_free_energy(z, minener_zero=minener_zero) * kT
    fig, ax, misc = plot_map(
        x,
        y,
        f,
        ax=ax,
        cmap=cmap,
        ncontours=ncontours,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cbar=cbar,
        cax=cax,
        cbar_label=cbar_label,
        cbar_orientation=cbar_orientation,
        norm=None,
        **kwargs,
    )
    if legacy:
        return fig, ax
    return fig, ax, misc


def plot_feature_histograms(
    all_hists,
    all_edges,
    feature_labels=None,
    ax=None,
    ylog=False,
    outfile=None,
    **kwargs,
):
    r"""Feature histogram plot

    Parameters
    ----------
    all_hists : iterable of np.ndarray
        List of histograms to be plotted.
    all_edges : iterable of np.ndarray
        List of edges for the histograms.
    feature_labels : iterable of str or pyemma.Featurizer, optional, default=None
        Labels of histogramed features, defaults to feature index.
    ax : matplotlib.Axes object, optional, default=None.
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    ylog : boolean, default=False
        If True, plot logarithm of histogram values.
    n_bins : int, default=50
        Number of bins the histogram uses.
    outfile : str, default=None
        If not None, saves plot to this file.
    ignore_dimwarning : boolean, default=False
        Enable plotting for more than 50 dimensions (on your own risk).
    **kwargs: kwargs passed to pyplot.fill_between. See the doc of pyplot for options.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the historams were plotted.

    """

    if feature_labels is not None:
        if not isinstance(feature_labels, list):
            from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer as _MDFeaturizer

            if isinstance(feature_labels, _MDFeaturizer):
                feature_labels = feature_labels.describe()
            else:
                raise ValueError("feature_labels must be a list of feature labels, a pyemma featurizer object or None.")

    # make nice plots if user does not decide on color and transparency
    if "color" not in kwargs.keys():
        kwargs["color"] = "b"
    if "alpha" not in kwargs.keys():
        kwargs["alpha"] = 0.25

    # check input
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    hist_offset = -0.2
    for h, (hist, edges) in enumerate(zip(all_hists, all_edges)):
        if not ylog:
            y = hist / hist.max()
        else:
            y = np.zeros_like(hist) + np.NaN
            pos_idx = hist > 0
            y[pos_idx] = np.log(hist[pos_idx]) / np.log(hist[pos_idx]).max()
        ax.fill_between(edges[:-1], y + h + hist_offset, y2=h + hist_offset, **kwargs)
        ax.axhline(y=h + hist_offset, xmin=0, xmax=1, color="k", linewidth=0.2)
    ax.set_ylim(hist_offset, h + hist_offset + 1)

    # formatting
    if feature_labels is None:
        feature_labels = [str(n) for n in range(len(all_hists))]
        ax.set_ylabel("Feature histograms")

    ax.set_yticks(np.array(range(len(feature_labels))) + 0.3)
    ax.set_yticklabels(feature_labels[::-1])
    ax.set_xlabel("Feature values")

    # save
    if outfile is not None:
        fig.savefig(outfile)
    return fig, ax
