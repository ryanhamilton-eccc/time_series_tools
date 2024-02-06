from unittest import TestCase
import ee

from time_series_tools import HarmonicTimeSeries, FourierTransform

class TestTimeSeriesTools(TestCase):
    
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

    
    def setUp(self):
        ee.Initialize()
        dataset = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(ee.Geometry.Point(-77.258, 44.0671))
            .filterDate('2018-01-01', '2021-12-31')
            .map(self.mask_clouds)
            .map(lambda image: image.addBands(image.normalizedDifference(['B8', 'B4']).rename('ndvi')))
        )
        
        self.time_series = HarmonicTimeSeries(
            dataset=dataset,
            dependent_variable='ndvi',
            modes=3
        )
    
    def test_add_constant(self):
        self.time_series.add_constant()
        self.assertIsInstance(self.time_series.dataset, ee.ImageCollection)
        print(self.time_series.dataset.first().bandNames().getInfo())
        print(self.time_series._independnet)
    
    def test_add_time(self):
        self.time_series.add_time()
        self.assertIsInstance(self.time_series.dataset, ee.ImageCollection)
        print(self.time_series.dataset.first().bandNames().getInfo())
        print(self.time_series._independnet)
        
    
    def test_add_harmonics(self):
        self.time_series.add_time().add_harmonics()
        self.assertIsInstance(self.time_series.dataset, ee.ImageCollection)
        print(self.time_series.dataset.first().bandNames().getInfo())
        print(self.time_series._independnet)
    
    def test_add_compute_trend(self):
        self.time_series.add_constant().add_time().add_harmonics().compute_trend()
        self.assertIsInstance(self.time_series.dataset, ee.ImageCollection)
        print(self.time_series.trend.bandNames().getInfo())

    def test_add_compute_coef(self):
        self.time_series.add_constant().add_time().add_harmonics().compute_trend().compute_coefficients()
        print(self.time_series.coefficients.bandNames().getInfo())
        print(self.time_series.dataset.first().bandNames().getInfo())

    def test_process(self):
        self.time_series.process()
        print(self.time_series.dataset.first().bandNames().getInfo())
        

class TestFourerTransform(TestCase):
    
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

    
    def setUp(self):
        ee.Initialize()
        dataset = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(ee.Geometry.Point(-77.258, 44.0671))
            .filterDate('2018-01-01', '2021-12-31')
            .map(self.mask_clouds)
            .map(lambda image: image.addBands(image.normalizedDifference(['B8', 'B4']).rename('ndvi')))
        )
        
        self.time_series = FourierTransform(
            dataset=dataset,
            dependent_variable='ndvi',
            modes=3
        )
    
    def test_compute_phase(self):
        self.time_series.add_constant().add_time().add_harmonics().compute_trend().compute_coefficients()
        self.time_series.compute_phase(1)
        print(self.time_series.dataset.first().bandNames().getInfo())
    
    def test_compute_amplitude(self):
        self.time_series.add_constant().add_time().add_harmonics().compute_trend().compute_coefficients()
        self.time_series.compute_amplitude(1)
        print(self.time_series.dataset.first().bandNames().getInfo())
    
    def test_process(self):
        fourier = self.time_series.process()
        print(fourier.bandNames().getInfo())