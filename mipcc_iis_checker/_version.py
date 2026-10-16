"""The checker's version, read from the installed package's metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mipcc-iis-checker")
except PackageNotFoundError:  # imported from a checkout that was never installed
    __version__ = "unknown"
