"""Application theme and colour palettes.

Colour schemes for the console log, the splash screen and histograms, and the
DARK and LIGHT presets. Colours are hex strings for Qt stylesheets.
"""

# Only the console uses this theme so far; connecting it to the View menu is future work.

from dataclasses import dataclass


@dataclass(frozen=True)
class LogColors:
    """Console log colour for each severity level."""

    debug: str
    info: str
    warning: str
    error: str
    critical: str


@dataclass(frozen=True)
class SplashColors:
    """Splash-screen background and text colours."""

    background: str
    text: str


@dataclass(frozen=True)
class HistogramColors:
    """Histogram colours."""

    gray_bar: str
    secondary_gray_bar: str
    channel_bars: tuple[str, str, str]
    background: str


@dataclass(frozen=True)
class ThemePallete:
    """Complete palette of a theme preset: log, splash-screen and histogram colours."""

    log: LogColors
    splash: SplashColors
    histogram: HistogramColors


DARK = ThemePallete(
    log=LogColors(
        debug="#555e6e",
        info="#c5c9d6",
        warning="#e5c07b",
        error="#e06c75",
        critical="#ff5f57",
    ),
    splash=SplashColors(
        background="#000000",
        text="#FFFFFF",
    ),
    histogram=HistogramColors(
        gray_bar="#B4C8C8C8",
        secondary_gray_bar="#A0FF5050",
        channel_bars=("#550000FF", "#5500FF00", "#55FF0000"),
        background="#FF1E1E1E",
    ),
)

LIGHT = ThemePallete(
    log=LogColors(
        debug="#888c96",
        info="#2d2f36",
        warning="#7d5a00",
        error="#c0292b",
        critical="#8b0000",
    ),
    splash=SplashColors(
        background="#FFFFFF",
        text="#000000",
    ),
    histogram=HistogramColors(
        gray_bar="#B4505050",
        secondary_gray_bar="#A0FF5050",
        channel_bars=("#800000AA", "#80008800", "#80AA0000"),
        background="#FFF0F0F0",
    ),
)

THEME = LIGHT
