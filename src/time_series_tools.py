from __future__ import annotations

import ee
from math import pi
from ee.imagecollection import ImageCollection

# inspo: https://developers.google.com/earth-engine/tutorials/community/time-series-modeling


class HarmonicTimeSeries:
    def __init__(self, dataset: ee.ImageCollection, dependent_variable: str, modes: list[int]):
        """A pipeline to process time series data

        Args:
            dataset (ee.ImageCollection): A preprocessed image collection (cloud mask, etc.)
            dependent_variable (str): The name of dependent variable to be modeled
            modes (list[int]): The number of terms to be used in the model
        """
        self.dataset = dataset.select(dependent_variable)
        self.dependent_variable = dependent_variable
        self.modes = modes
        self._independnet = []
        self.trend = None
        self.coefficients = None
    
    @property
    def frequencies(self):
        return [mode for mode in range(1, self.modes + 1)]
    
    def add_constant(self) -> HarmonicTimeSeries:
        self.dataset = self.dataset.map(lambda image: image.addBands(ee.Image(1)))
        self._independnet.append('constant')
        return self
    
    def add_time(self) -> HarmonicTimeSeries:
        def _add_time(image):
            date = date = ee.Date(image.get('system:time_start'))
            year = date.difference(ee.Date('1970-01-01'), 'year')
            time_radians = ee.Image(year.multiply(2 * pi * 1))
            return image.addBands(time_radians.rename('t').float())
        self.dataset = self.dataset.map(_add_time)
        self._independnet.append('t')
        return self
    
    def add_harmonics(self) -> HarmonicTimeSeries:
        cosine = self._get_names('cos', self.modes)
        sine = self._get_names('sin', self.modes)
        def _add_harmoncis(image):
            freq = ee.Image.constant(self.frequencies)
            time = image.select('t')
            cosines = time.multiply(freq).cos().rename(cosine)
            sines = time.multiply(freq).sin().rename(sine)
            return image.addBands(cosines).addBands(sines)
        self.dataset = self.dataset.map(_add_harmoncis)
        self._independnet.extend(cosine + sine)
        return self
            
    def compute_trend(self) -> HarmonicTimeSeries:
        self.trend = (
            self.dataset.select([self.dependent_variable] + self._independnet)
            .reduce(ee.Reducer.linearRegression(len(self._independnet), 1))    
        )
        return self
    
    def compute_coefficients(self) -> HarmonicTimeSeries:
        self.coefficients = (
            self.trend.select('coefficients')
            .arrayFlatten([self._independnet, ['coef']])
        )
        self.dataset = self.dataset.map(lambda image: image.addBands(self.coefficients))
        return self

    def process(self) -> HarmonicTimeSeries:
        (
            self.add_constant()
            .add_time()
            .add_harmonics()
            .compute_trend()
            .compute_coefficients()
        )
        return self

    @staticmethod
    def _get_names(prefix, modes):
        return [f'{prefix}_{mode}' for mode in range(1, modes + 1)]


class FourierTransform(HarmonicTimeSeries):
    def __init__(self, dataset: ImageCollection, dependent_variable: str, modes: list[int]):
        super().__init__(dataset, dependent_variable, modes)
    
    def compute_phase(self, mode: int):
        def compute(image):
            cos = image.select(f"cos_{mode}_coef")
            sin = image.select(f"sin_{mode}_coef")
            arctan = cos.atan2(sin)
            return image.addBands(arctan.rename(f"phase_{mode}"))
        self.dataset = self.dataset.map(compute)
        return self
        
    def compute_amplitude(self, mode: int):
        def compute_amplitude(image):
            cos = image.select(f"cos_{mode}_coef")
            sin = image.select(f"sin_{mode}_coef")
            amplitude = cos.hypot(sin)
            return image.addBands(amplitude.rename(f"amplitude_{mode}"))
        self.dataset = self.dataset.map(compute_amplitude)
        return self
    
    def tranform(self) -> ee.Image:
        return self.dataset.median().unitScale(-1, 1)
    
    def process(self) -> ee.Image:
        dataset = super().process()
        
        for mode in range(1, self.modes + 1):
            dataset = dataset.compute_phase(mode)
            dataset = dataset.compute_amplitude(mode)
        pattern = f"{self.dependent_variable}|.*coef|amp.*|phase.*"
        return self.tranform().select(pattern)
