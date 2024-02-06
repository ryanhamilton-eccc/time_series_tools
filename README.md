# time_series_tools
Tools for doing Time Series Analysis in Google Earth Engine

## Installation
```bash
pip install git+https://github.com/ryanhmailton-eccc/time_series_tools.git
```

## Usage
```python
import time_series_tools as tst
import ee

ee.Initialize()

# Create a time series of NDVI for senitnel 2 data
def mask_clouds(self, image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask)

dataset = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(ee.Geometry.Point(-77.258, 44.0671))
    .filterDate('2018-01-01', '2021-12-31')
    .map(self.mask_clouds)
    .map(lambda image: image.addBands(image.normalizedDifference(['B8', 'B4']).rename('ndvi')))
)

fourier = tst.FourierTransform(
     dataset=dataset,
    dependent_variable='ndvi',
    modes=3
)

product = fourier.process()

```