from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prescient-jamun")
except PackageNotFoundError:
    __version__ = "unknown version"
