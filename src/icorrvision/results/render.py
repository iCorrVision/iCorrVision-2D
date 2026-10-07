import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from icorrvision.contracts import HeatmapPayload

from icorrvision.results.payload import ColorScale


class HeatmapFigure:
    """An image axes and a dedicated colorbar axes inside one Figure."""

    _AX_RECT = (0.10, 0.08, 0.66, 0.78)
    _CAX_RECT = (0.79, 0.10, 0.022, 0.76)

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.ax = figure.add_axes(self._AX_RECT)
        self.cax = figure.add_axes(self._CAX_RECT)
        self._norm: Normalize | None = None
        self._cmap: Colormap | None = None

    @property
    def has_content(self) -> bool:
        return bool(self.ax.images) or bool(self.ax.collections)

    def set_scale(self, scale: ColorScale) -> None:
        self._norm = Normalize(vmin=scale.vmin, vmax=scale.vmax)
        self._cmap = colormaps[scale.cmap]
        self.cax.clear()
        self.cax.set_axes_locator(None)
        self.figure.colorbar(
            ScalarMappable(norm=self._norm, cmap=self._cmap),
            cax=self.cax,
            extend=scale.extend,
            label=scale.colorbar_label,
        )

    def draw(
        self,
        image: np.ndarray,
        payload: HeatmapPayload,
        opacity: float,
        keep_view: bool = False,
    ) -> None:
        if self._norm is None:
            raise RuntimeError("HeatmapFigure.set_scale() must be called before draw()")

        limits = (
            (self.ax.get_xlim(), self.ax.get_ylim())
            if keep_view and self.has_content
            else None
        )
        self.ax.clear()
        self.ax.imshow(image, cmap="gray")
        self.ax.pcolormesh(
            payload.x,
            payload.y,
            payload.magnitude,
            cmap=self._cmap,
            norm=self._norm,
            alpha=opacity,
            shading="nearest",
        )
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(f"{payload.value_label}\n{payload.frame_name}", fontsize=10)
        if limits is not None:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])


def render_heatmap(
    figure: Figure,
    image: np.ndarray,
    payload: HeatmapPayload,
    scale: ColorScale,
    opacity: float = 0.7,
) -> HeatmapFigure:
    """One-shot render into a fresh figure, for static exports."""
    heatmap = HeatmapFigure(figure)
    heatmap.set_scale(scale)
    heatmap.draw(image, payload, opacity)
    return heatmap
