from enum import Enum
import rasterio
import rasterio.mask
from pathlib import Path
import numpy as np
from typing import List, Sequence
import matplotlib.pyplot as plt
import geopandas

class Band(str, Enum):
    """A collection of different Sentinel-2 band names."""

    B01 = "B01"
    B02 = "B02"
    B03 = "B03"
    B04 = "B04"
    B05 = "B05"
    B06 = "B06"
    B07 = "B07"
    B08 = "B08"
    B8A = "B8A"
    B09 = "B09"
    B10 = "B10"
    B11 = "B11"
    B12 = "B12"

    def __str__(self):
        return self.value


def read_s2_jp2_data(jp2_data_path: Path) -> np.ndarray:
    """
    Read band from Sentinel-2 jp2 file.
    """
    with rasterio.open(jp2_data_path) as data:
        # rasterio is 1-indexed
        return data.read(1)

def read_s2_jp2_data_with_clipping(
    band_data_path: Path, clip_geoseries: geopandas.GeoSeries, envelope: bool = True
) -> np.ndarray:
    with rasterio.open(band_data_path) as data:
        # ensure that the data is using the same coordinate reference system
        reprojected_geoseries = clip_geoseries.to_crs(data.crs)
        reprojected_geoseries = (
            reprojected_geoseries.envelope if envelope else reprojected_geoseries
        )
        out_img, _out_transform = rasterio.mask.mask(data, reprojected_geoseries, crop=True)
        # drop singleton axes
        out_img = out_img.squeeze()
    return out_img


def _get_all_jp2_files(source_dir: Path, parent_dir: str = "IMG_DATA/R60m") -> List[Path]:
    """
    Given a Sentinel-2 source directory, find all jp2 files that have
    a parent folder named `parent_dir`.
    Usually, it should be the folder `IMG_DATA`, other possible source
    would be the quality masks in `QI_DATA`.
    To not load band multiple times at different resolutions, by default
    the lowest 60m band is loaded.

    Note: Depending on the acquisition date and data type, the structure might be different
    and no sub-directory within `IMG_DATA` exists!
    """
    image_files = list(source_dir.glob(f"**/{parent_dir}/*.jp2"))
    assert len(image_files) > 0
    return image_files


class S2_TileReader:
    def __init__(self, safe_directory: Path, img_data_parent_dir: str = "IMG_DATA/R60m"):
        self.image_files = _get_all_jp2_files(safe_directory, parent_dir=img_data_parent_dir)

    def _get_band_path(self, band: Band) -> Path:
        return [f for f in self.image_files if f"_{band}_" in f.name][0]

    def read_band_data(self, band: Band) -> np.ndarray:
        band_path = self._get_band_path(band)
        return read_s2_jp2_data(band_path)

    def read_band_data_with_clipping(
        self, band: Band, clip_geoseries: geopandas.GeoSeries, envelope: bool = True
    ) -> np.ndarray:
        band_path = self._get_band_path(band)
        return read_s2_jp2_data_with_clipping(band_path, clip_geoseries, envelope=envelope)


def quant_norm_data(
    data: np.ndarray, lower_quant: float = 0.01, upper_quant: float = 0.99
) -> np.ndarray:
    """
    Normalize the data by quantiles `lower_quant/upper_quant`.
    The quantiles are calculated globally/*across all channels*.
    """
    masked_data = np.ma.masked_equal(data, 0)
    lq, uq = np.quantile(masked_data.compressed(), (lower_quant, upper_quant))
    data = np.clip(data, a_min=lq, a_max=uq)
    data = (data - lq) / (uq - lq)
    return data


def vis(data: np.ndarray, quant_norm: bool = True):
    """
    Visualize an array by calling `imshow` with `cmap="gray"`.
    By default, the image is normalized through `quant_norm_data`.
    """
    if quant_norm:
        data = quant_norm_data(data)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(data, cmap="gray")