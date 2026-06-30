"""Python implementation of the CROPGRO-Strawberry crop growth model.

This module contains a simplified, purely Python implementation of the
CROPGRO strawberry model.  The structure mirrors the original Fortran
code but trades some complexity for readability.  All major calculation
steps are implemented as small functions decorated with ``@njit`` to keep
them fast when the optional ``numba`` dependency is available.
"""

# CROPGRO-Strawberry Model Implementation in Python
# This is a simplified implementation of the CROPGRO model for strawberries

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from numba import njit


@dataclass
class PlantState:
    biomass: float = 0.0
    leaf_area_index: float = 0.1
    root_depth: float = 5.0
    fruit_number: float = 0.0
    fruit_biomass: float = 0.0
    leaf_biomass: float = 0.0
    stem_biomass: float = 0.0
    root_biomass: float = 0.0
    phenological_stage: str = "GERMINATION"
    development_rate: float = 0.0
    crown_number: float = 1.0
    runner_number: float = 0.0


@njit
def _calc_daylength(latitude, day_of_year):
    """Return length of the day in hours for a given latitude and date."""
    declination = 23.45 * np.sin(np.deg2rad(360 * (day_of_year - 80) / 365))
    lat_rad = np.deg2rad(latitude)
    term = -np.tan(lat_rad) * np.tan(np.deg2rad(declination))
    if term >= 1.0:
        return 0.0
    elif term <= -1.0:
        return 24.0
    else:
        return 24.0 * np.arccos(term) / np.pi


@njit
def _thermal_time(tmin, tmax, tbase, topt, tmax_th):
    """Calculate thermal time accumulation for a single day."""
    tavg = (tmin + tmax) / 2.0
    if tavg <= tbase:
        return 0.0
    elif tavg <= topt:
        return tavg - tbase
    elif tavg <= tmax_th:
        return (topt - tbase - (tavg - topt) * 
                ((topt - tbase) / (tmax_th - topt)))
    else:
        return 0.0


@njit
def _photosynthesis(solar_radiation, tmax, tmin, rue, tbase, topt, 
                    k_light, lai, co2):
    """Estimate daily photosynthesis based on temperature and light."""
    tavg = (tmax + tmin) / 2.0
    if tavg <= tbase:
        temp_effect = 0.0
    elif tavg >= topt:
        temp_effect = 1.0
    else:
        temp_effect = (tavg - tbase) / (topt - tbase)
    co2_effect = 1.0 + 0.11 * np.log(co2 / 400.0)
    light_interception = 1.0 - np.exp(-k_light * lai)
    return (solar_radiation * rue * temp_effect * co2_effect * 
            light_interception)


@njit
def _transpiration(solar_radiation, tmax, tmin, rh, lai):
    """Compute potential plant transpiration using a simple ET0 approach."""
    tavg = (tmax + tmin) / 2.0
    et0 = 0.0023 * solar_radiation * np.sqrt(tmax - tmin) * (tavg + 17.8)
    kc = 0.3 + 0.7 * (1.0 - np.exp(-0.7 * lai))
    return et0 * kc


@njit
def _water_stress(field_capacity, wilting_point, root_depth, rainfall, 
                  transpiration):
    """Derive a water stress factor from soil moisture balance."""
    available_water = (field_capacity - wilting_point) * root_depth
    effective_rainfall = rainfall * 0.7
    deficit = max(0.0, transpiration - effective_rainfall)
    if deficit == 0.0:
        return 0.0
    else:
        stress_factor = min(1.0, deficit / available_water)
        return stress_factor


@njit
def _maintenance_resp(leaf_biomass, stem_biomass, root_biomass, 
                      fruit_biomass, tmin, tmax):
    """Calculate maintenance respiration of all plant organs."""
    tavg = (tmin + tmax) / 2.0
    temp_factor = 2.0 ** ((tavg - 20.0) / 10.0)
    resp_leaf = leaf_biomass * 0.03 * temp_factor
    resp_stem = stem_biomass * 0.015 * temp_factor
    resp_root = root_biomass * 0.01 * temp_factor
    resp_fruit = fruit_biomass * 0.01 * temp_factor
    return resp_leaf + resp_stem + resp_root + resp_fruit


class CropgroStrawberry:
    """
    A Python implementation of the CROPGRO-Strawberry crop model.
    
    This model simulates strawberry growth and development based on 
    environmental conditions, plant characteristics, and management practices.
    """
    
    def __init__(self, latitude, planting_date, soil_properties, 
                 cultivar_params):
        """
        Initialize the CROPGRO-Strawberry model.
        
        Parameters:
        -----------
        latitude : float
            Site latitude in decimal degrees
        planting_date : str
            Planting date in format 'YYYY-MM-DD'
        soil_properties : dict
            Dictionary containing soil properties (depth, texture, 
            water holding capacity, etc.)
        cultivar_params : dict
            Dictionary containing cultivar-specific parameters
        """
        self.latitude = latitude
        self.planting_date = datetime.strptime(planting_date, '%Y-%m-%d')
        self.soil = soil_properties
        self.cultivar = cultivar_params
        
        self.days_after_planting = 0
        self.plant_state = PlantState()
        
        self.thermal_time = 0.0
        
        self.phenology_stages = {
            'GERMINATION': 0,
            'EMERGENCE': 50,
            'JUVENILE': 100, 
            'VEGETATIVE': 200,
            'FLORAL_INDUCTION': 400,
            'FLOWERING': 600,
            'FRUIT_SET': 700,
            'FRUIT_DEVELOPMENT': 800,
            'FRUIT_MATURITY': 1000,
            'SENESCENCE': 1500
        }
        
        self.results = []
        
    def calculate_daylength(self, day_of_year):
        """
        Calculate daylength based on latitude and day of year.
        
        Parameters:
        -----------
        day_of_year : int
            Day of year (1-366)
            
        Returns:
        --------
        float
            Daylength in hours
        """
        return _calc_daylength(self.latitude, day_of_year)
    
    def calculate_thermal_time(self, tmin, tmax):
        """
        Calculate thermal time (degree-days) based on daily temperatures.
        
        Parameters:
        -----------
        tmin : float
            Minimum daily temperature (°C)
        tmax : float
            Maximum daily temperature (°C)
            
        Returns:
        --------
        float
            Thermal time accumulation for the day (degree-days)
        """
        tbase = self.cultivar['tbase']
        topt = self.cultivar['topt']
        tmax_th = self.cultivar['tmax_th']
        
        return _thermal_time(tmin, tmax, tbase, topt, tmax_th)
    
    def update_phenology(self, thermal_time_today):
        """
        Update plant phenological stage based on accumulated thermal time.
        
        Parameters:
        -----------
        thermal_time_today : float
            Thermal time accumulated for the current day
        """
        self.thermal_time += thermal_time_today
        
        current_stage = self.plant_state.phenological_stage
        stages = list(self.phenology_stages.keys())
        current_index = stages.index(current_stage)

        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            if self.thermal_time >= self.phenology_stages[next_stage]:
                self.plant_state.phenological_stage = next_stage
    
    def calculate_photosynthesis(self, solar_radiation, tmax, tmin, co2=400):
        """
        Calculate daily photosynthesis rate.
        
        Parameters:
        -----------
        solar_radiation : float
            Daily solar radiation (MJ/m²)
        tmax : float
            Maximum daily temperature (°C)
        tmin : float
            Minimum daily temperature (°C)
        co2 : float, optional
            Atmospheric CO2 concentration (ppm)
            
        Returns:
        --------
        float
            Daily photosynthesis rate (g CH2O/m²)
        """
        rue = self.cultivar['rue']
        lai = self.plant_state.leaf_area_index
        return _photosynthesis(
            solar_radiation,
            tmax,
            tmin,
            rue,
            self.cultivar['tbase'],
            self.cultivar['topt'],
            self.cultivar['k_light'],
            lai,
            co2,
        )
    
    def calculate_transpiration(self, solar_radiation, tmax, tmin, rh, wind_speed):
        """
        Calculate plant transpiration using a simplified 
        Penman-Monteith approach.
        
        Parameters:
        -----------
        solar_radiation : float
            Daily solar radiation (MJ/m²)
        tmax : float
            Maximum daily temperature (°C)
        tmin : float
            Minimum daily temperature (°C)
        rh : float
            Relative humidity (%)
        wind_speed : float
            Wind speed (m/s)
            
        Returns:
        --------
        float
            Daily transpiration (mm)
        """
        lai = self.plant_state.leaf_area_index
        base_transpiration = _transpiration(solar_radiation, tmax, tmin, rh, lai)
        
        wind_modifier = 1.0 + 0.1 * (wind_speed - 2.0)
        wind_modifier = max(0.5, min(2.0, wind_modifier))
        
        return base_transpiration * wind_modifier
    
    def partition_biomass(self, daily_biomass):
        """
        Partition new biomass to plant organs based on development stage.
        
        Parameters:
        -----------
        daily_biomass : float
            Daily biomass production (g/plant)
        """
        stage = self.plant_state.phenological_stage
        
        if stage in ['GERMINATION', 'EMERGENCE', 'JUVENILE']:
            root_fraction = 0.4
            leaf_fraction = 0.4
            stem_fraction = 0.2
            fruit_fraction = 0.0
        elif stage in ['VEGETATIVE', 'FLORAL_INDUCTION']:
            root_fraction = 0.2
            leaf_fraction = 0.5
            stem_fraction = 0.3
            fruit_fraction = 0.0
        elif stage == 'FLOWERING':
            root_fraction = 0.1
            leaf_fraction = 0.4
            stem_fraction = 0.3
            fruit_fraction = 0.2
        elif stage in ['FRUIT_SET', 'FRUIT_DEVELOPMENT']:
            root_fraction = 0.05
            leaf_fraction = 0.25
            stem_fraction = 0.2
            fruit_fraction = 0.5
        elif stage == 'FRUIT_MATURITY':
            root_fraction = 0.0
            leaf_fraction = 0.1
            stem_fraction = 0.1
            fruit_fraction = 0.8
        else:
            root_fraction = 0.0
            leaf_fraction = 0.0
            stem_fraction = 0.0
            fruit_fraction = 0.0
            
        self.plant_state.root_biomass += daily_biomass * root_fraction
        self.plant_state.leaf_biomass += daily_biomass * leaf_fraction
        self.plant_state.stem_biomass += daily_biomass * stem_fraction
        self.plant_state.fruit_biomass += daily_biomass * fruit_fraction
        
        self.plant_state.biomass = (
            self.plant_state.root_biomass
            + self.plant_state.leaf_biomass
            + self.plant_state.stem_biomass
            + self.plant_state.fruit_biomass
        )
        
        sla = self.cultivar['sla']
        if stage in ['FRUIT_DEVELOPMENT', 'FRUIT_MATURITY', 'SENESCENCE']:
            sla *= 0.8
            
        self.plant_state.leaf_area_index = self.plant_state.leaf_biomass * sla
        
        max_root_growth_rate = 0.5
        max_root_depth = self.soil['max_root_depth']
        
        potential_root_growth = max_root_growth_rate * root_fraction
        current_root_depth = self.plant_state.root_depth
        
        if current_root_depth < max_root_depth:
            self.plant_state.root_depth = min(
                current_root_depth + potential_root_growth, max_root_depth)
    
    def update_runners(self):
        """Update the number of runners based on development stage 
        and conditions."""
        if self.plant_state.phenological_stage in ['VEGETATIVE', 
                                                   'FLORAL_INDUCTION']:
            self.plant_state.runner_number += (
                0.1 * self.plant_state.crown_number)
    
    def update_crowns(self):
        """Update the number of crowns based on development stage 
        and conditions."""
        if self.plant_state.phenological_stage in ['VEGETATIVE', 
                                                   'FLORAL_INDUCTION', 
                                                   'FLOWERING']:
            self.plant_state.crown_number += (
                0.02 * self.plant_state.crown_number)
    
    def update_fruits(self):
        """Update fruit number and individual fruit weight."""
        stage = self.plant_state.phenological_stage
        
        if stage == 'FLOWERING':
            new_fruits = (self.cultivar['potential_fruits_per_crown'] * 
                          self.plant_state.crown_number * 0.1)
            self.plant_state.fruit_number += new_fruits
        elif stage == 'FRUIT_SET':
            new_fruits = (self.cultivar['potential_fruits_per_crown'] * 
                          self.plant_state.crown_number * 0.2)
            self.plant_state.fruit_number += new_fruits
    
    def simulate_day(self, weather_data):
        """
        Simulate one day of strawberry growth.
        
        Parameters:
        -----------
        weather_data : dict
            Dictionary containing weather data for the day:
            - tmax: Maximum temperature (°C)
            - tmin: Minimum temperature (°C)
            - solar_radiation: Solar radiation (MJ/m²)
            - rainfall: Rainfall (mm)
            - rh: Relative humidity (%)
            - wind_speed: Wind speed (m/s)
            - date: Date in 'YYYY-MM-DD' format
        """
        self.days_after_planting += 1
        
        current_date = datetime.strptime(weather_data['date'], '%Y-%m-%d')
        day_of_year = current_date.timetuple().tm_yday
        
        daylength = self.calculate_daylength(day_of_year)
        
        thermal_time_today = self.calculate_thermal_time(
            weather_data['tmin'], weather_data['tmax'])
        
        self.update_phenology(thermal_time_today)
        
        photosynthesis = self.calculate_photosynthesis(
            weather_data['solar_radiation'],
            weather_data['tmax'],
            weather_data['tmin']
        )
        
        transpiration = self.calculate_transpiration(
            weather_data['solar_radiation'],
            weather_data['tmax'],
            weather_data['tmin'],
            weather_data['rh'],
            weather_data['wind_speed']
        )
        
        water_stress = self.calculate_water_stress(
            weather_data['rainfall'], transpiration)
        
        photosynthesis *= (1 - water_stress)
        
        plant_density = 5.0
        daily_biomass = photosynthesis / plant_density
        
        maintenance_resp = self.calculate_maintenance_respiration(
            weather_data['tmin'], weather_data['tmax'])
        daily_biomass = max(0, daily_biomass - maintenance_resp)
        
        self.partition_biomass(daily_biomass)
        
        self.update_runners()
        self.update_crowns()
        self.update_fruits()
        
        self.results.append({
            'date': weather_data['date'],
            'dap': self.days_after_planting,
            'stage': self.plant_state.phenological_stage,
            'thermal_time': self.thermal_time,
            'biomass': self.plant_state.biomass,
            'leaf_area_index': self.plant_state.leaf_area_index,
            'root_depth': self.plant_state.root_depth,
            'fruit_number': self.plant_state.fruit_number,
            'fruit_biomass': self.plant_state.fruit_biomass,
            'leaf_biomass': self.plant_state.leaf_biomass,
            'stem_biomass': self.plant_state.stem_biomass,
            'root_biomass': self.plant_state.root_biomass,
            'crown_number': self.plant_state.crown_number,
            'runner_number': self.plant_state.runner_number,
            'water_stress': water_stress,
            'daylength': daylength,
            'photosynthesis': photosynthesis,
            'transpiration': transpiration
        })
    
    def calculate_water_stress(self, rainfall, transpiration):
        """
        Calculate water stress factor (0-1) based on soil water balance.
        
        Parameters:
        -----------
        rainfall : float
            Daily rainfall (mm)
        transpiration : float
            Potential transpiration (mm)
            
        Returns:
        --------
        float
            Water stress factor (0 = no stress, 1 = maximum stress)
        """
        field_capacity = self.soil['field_capacity']
        wilting_point = self.soil['wilting_point']
        root_depth = self.plant_state.root_depth / 100.0
        return _water_stress(field_capacity, wilting_point, root_depth, 
                           rainfall, transpiration)
    
    def calculate_maintenance_respiration(self, tmin, tmax):
        """
        Calculate maintenance respiration based on biomass and temperature.
        
        Parameters:
        -----------
        tmin : float
            Minimum daily temperature (°C)
        tmax : float
            Maximum daily temperature (°C)
            
        Returns:
        --------
        float
            Maintenance respiration (g/plant)
        """
        return _maintenance_resp(
            self.plant_state.leaf_biomass,
            self.plant_state.stem_biomass,
            self.plant_state.root_biomass,
            self.plant_state.fruit_biomass,
            tmin,
            tmax,
        )
    
    def simulate_growth(self, weather_data_df):
        """
        Simulate strawberry growth for a period defined by the weather data.
        
        Parameters:
        -----------
        weather_data_df : pandas.DataFrame
            DataFrame containing daily weather data with the following columns:
            - date: Date in 'YYYY-MM-DD' format
            - tmax: Maximum temperature (°C)
            - tmin: Minimum temperature (°C)
            - solar_radiation: Solar radiation (MJ/m²)
            - rainfall: Rainfall (mm)
            - rh: Relative humidity (%)
            - wind_speed: Wind speed (m/s)
        """
        self.results = []
        
        for row in weather_data_df.itertuples(index=False):
            weather_day = {
                'date': row.date,
                'tmax': row.tmax,
                'tmin': row.tmin,
                'solar_radiation': row.solar_radiation,
                'rainfall': row.rainfall,
                'rh': row.rh,
                'wind_speed': row.wind_speed,
            }
            self.simulate_day(weather_day)
        
        self.results_df = pd.DataFrame(self.results)
        return self.results_df
    
    def plot_results(self):
        """Plot key simulation results."""
        if not hasattr(self, 'results_df') or len(self.results_df) == 0:
            print("No simulation results to plot. "
                  "Run simulate_growth() first.")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        
        axs[0, 0].plot(self.results_df['dap'], self.results_df['biomass'], 
                      'b-', label='总生物量')
        axs[0, 0].plot(self.results_df['dap'], 
                      self.results_df['leaf_biomass'], 'g-', label='叶片')
        axs[0, 0].plot(self.results_df['dap'], 
                      self.results_df['stem_biomass'], 'k-', label='茎秆')
        axs[0, 0].plot(self.results_df['dap'], 
                      self.results_df['root_biomass'], 'r-', label='根系')
        axs[0, 0].plot(self.results_df['dap'], 
                      self.results_df['fruit_biomass'], 'm-', label='果实')
        axs[0, 0].set_xlabel('种植后天数')
        axs[0, 0].set_ylabel('生物量 (g/株)')
        axs[0, 0].set_title('植物生物量')
        axs[0, 0].legend()
        
        axs[0, 1].plot(self.results_df['dap'], 
                      self.results_df['leaf_area_index'], 'g-')
        axs[0, 1].set_xlabel('种植后天数')
        axs[0, 1].set_ylabel('叶面积指数 (m^2/m^2)')
        axs[0, 1].set_title('叶面积指数')
        
        axs[1, 0].plot(self.results_df['dap'], 
                      self.results_df['fruit_number'], 'm-')
        axs[1, 0].set_xlabel('种植后天数')
        axs[1, 0].set_ylabel('果实数量 (个/株)')
        axs[1, 0].set_title('果实数量')
        
        axs[1, 1].plot(self.results_df['dap'], 
                      self.results_df['crown_number'], 'b-', label='冠数')
        axs[1, 1].plot(self.results_df['dap'], 
                      self.results_df['runner_number'], 'r-', 
                      label='匍匐茎数')
        axs[1, 1].set_xlabel('种植后天数')
        axs[1, 1].set_ylabel('数量 (个/株)')
        axs[1, 1].set_title('冠数和匍匐茎数')
        axs[1, 1].legend()
        
        axs[2, 0].plot(self.results_df['dap'], 
                      self.results_df['water_stress'], 'r-')
        axs[2, 0].set_xlabel('种植后天数')
        axs[2, 0].set_ylabel('水分胁迫因子 (0-1)')
        axs[2, 0].set_title('水分胁迫因子')
        
        stages = list(self.phenology_stages.keys())
        stage_values = [stages.index(stage) 
                       for stage in self.results_df['stage']]
        
        stage_labels_cn = [
            '发芽期', '出苗期', '幼苗期', '营养生长期', '花芽分化期',
            '开花期', '坐果期', '果实发育期', '果实成熟期', '衰老期'
        ]
        
        axs[2, 1].plot(self.results_df['dap'], stage_values, 'b-')
        axs[2, 1].set_xlabel('种植后天数')
        axs[2, 1].set_ylabel('发育阶段')
        axs[2, 1].set_yticks(range(len(stages)))
        axs[2, 1].set_yticklabels(stage_labels_cn)
        axs[2, 1].set_title('物候发育阶段')
        
        plt.tight_layout()
        return fig


def parse_dssat_date(code: str) -> str:
    """Convert DSSAT YYDDD date code to YYYY-MM-DD string."""
    year = 2000 + int(code[:2])
    doy = int(code[2:])
    return datetime.strptime(f"{year} {doy}", "%Y %j").strftime("%Y-%m-%d")


def parse_srx_file(path: str):
    """Extract planting date and weather station code from SRX file."""
    planting_code = None
    wsta = None
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("@L ID_FIELD") and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) >= 3:
                wsta = parts[2]
        if line.startswith("@P PDATE") and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                planting_code = parts[1]
    planting_date = parse_dssat_date(planting_code) if planting_code else None
    return planting_date, wsta


def read_wth_file(path: str) -> pd.DataFrame:
    """Parse DSSAT .WTH file into pandas DataFrame."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("@DATE"))
    header = lines[start].split()
    indices = {h: idx for idx, h in enumerate(header)}
    records = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.startswith("*"):
            continue
        parts = line.split()
        code = parts[0]
        date = parse_dssat_date(code)
        rec = {
            "date": date,
            "tmax": float(parts[indices["TMAX"]]),
            "tmin": float(parts[indices["TMIN"]]),
            "solar_radiation": float(parts[indices["SRAD"]]),
            "rainfall": float(parts[indices["RAIN"]]) if "RAIN" in indices and len(parts) > indices["RAIN"] else 0.0,
            "rh": float(parts[indices["RHUM"]]) if "RHUM" in indices and len(parts) > indices["RHUM"] else 70.0,
            "wind_speed": float(parts[indices["WIND"]]) if "WIND" in indices and len(parts) > indices["WIND"] else 2.0,
        }
        records.append(rec)
    return pd.DataFrame(records)


def run_with_dssat_data(srx_path: str, weather_dir: str = "dssat-csm-data-develop/Weather"):
    """Run model using real DSSAT experiment data."""
    import os
    planting_date, wsta = parse_srx_file(srx_path)
    if planting_date is None or wsta is None:
        raise ValueError("Could not parse SRX file")
    
    year = planting_date[:4]
    matches = [f for f in os.listdir(weather_dir) if f.startswith(f"{wsta}{year[2:]}") and f.endswith(".WTH")]
    if not matches:
        raise FileNotFoundError(f"Weather file not found for station {wsta} and year {year}")
    
    wth_path = os.path.join(weather_dir, matches[0])
    weather_df = read_wth_file(wth_path)
    
    soil_properties = {
        'max_root_depth': 50.0,
        'field_capacity': 200.0,
        'wilting_point': 50.0,
    }
    
    cultivar_params = {
        'name': 'Generic',
        'tbase': 4.0,
        'topt': 22.0,
        'tmax_th': 35.0,
        'rue': 2.5,
        'k_light': 0.6,
        'sla': 0.02,
        'potential_fruits_per_crown': 10.0,
    }
    
    model = CropgroStrawberry(
        latitude=40.0,
        planting_date=planting_date,
        soil_properties=soil_properties,
        cultivar_params=cultivar_params
    )
    
    results = model.simulate_growth(weather_df)
    fig = model.plot_results()
    
    return model, results, fig, planting_date, wsta


def run_example_simulation():
    """Run the model with synthetic weather data and return results."""
    soil_properties = {
        'max_root_depth': 50.0,
        'field_capacity': 200.0,
        'wilting_point': 50.0,
    }
    
    cultivar_params = {
        'name': 'Albion',
        'tbase': 4.0,
        'topt': 22.0,
        'tmax_th': 35.0,
        'rue': 2.5,
        'k_light': 0.6,
        'sla': 0.02,
        'potential_fruits_per_crown': 10.0
    }
    
    start_date = '2023-05-01'
    end_date = '2023-10-31'
    
    dates = pd.date_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    np.random.seed(42)
    
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_component = 10 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
    
    tmax = 25.0 + seasonal_component + np.random.normal(0, 3, n_days)
    tmin = 10.0 + seasonal_component + np.random.normal(0, 2, n_days)
    
    solar_rad = (15.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 172) / 365) 
                + np.random.normal(0, 2, n_days))
    solar_rad = np.maximum(1.0, solar_rad)
    
    rainfall = np.zeros(n_days)
    rain_events = np.random.rand(n_days) < 0.3
    rainfall[rain_events] = np.random.exponential(5, np.sum(rain_events))
    
    rh = 70.0 + np.random.normal(0, 10, n_days)
    rh = np.clip(rh, 20, 100)
    
    wind_speed = 2.0 + np.random.exponential(1, n_days)
    
    weather_df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'tmax': tmax,
        'tmin': tmin,
        'solar_radiation': solar_rad,
        'rainfall': rainfall,
        'rh': rh,
        'wind_speed': wind_speed
    })
    
    model = CropgroStrawberry(
        latitude=40.0,
        planting_date=start_date,
        soil_properties=soil_properties,
        cultivar_params=cultivar_params
    )
    
    results = model.simulate_growth(weather_df)
    fig = model.plot_results()
    
    return model, results, fig


if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except:
        pass
    
    import os
    
    srx_files = [
        "dssat-csm-data-develop/Strawberry/UFBA1401.SRX",
        "dssat-csm-data-develop/Strawberry/UFBA1601.SRX",
        "dssat-csm-data-develop/Strawberry/UFBA1701.SRX",
        "dssat-csm-data-develop/Strawberry/UFWM1401.SRX",
    ]
    
    output_dir = "测试"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, srx_path in enumerate(srx_files):
        if os.path.exists(srx_path):
            experiment_name = os.path.basename(srx_path).replace('.SRX', '')
            print(f"\n=== Processing {experiment_name} ===")
            
            try:
                model, results, fig, planting_date, wsta = run_with_dssat_data(srx_path)
                
                print(f"📅 Planting date: {planting_date}")
                print(f"🌍 Weather station: {wsta}")
                print(f"📊 Final biomass: {results['biomass'].iloc[-1]:.2f} g/plant")
                print(f"🍓 Final fruit biomass: {results['fruit_biomass'].iloc[-1]:.2f} g/plant")
                print(f"🌿 Final LAI: {results['leaf_area_index'].iloc[-1]:.2f} m^2/m^2")
                print(f"📈 Final stage: {results['stage'].iloc[-1]}")
                print(f"📅 Days simulated: {len(results)}")
                
                if i == 0:
                    output_file = os.path.join(output_dir, f"{experiment_name}.png")
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"💾 Chart saved to: {output_file}")
                else:
                    print(f"⚠️  Skipping chart generation (similar to {srx_files[0]})")
                
                csv_file = os.path.join(output_dir, f"{experiment_name}.csv")
                results.to_csv(csv_file, index=False)
                print(f"💾 Data saved to: {csv_file}")
                
                plt.close(fig)
                
                all_results.append({
                    'name': experiment_name,
                    'year': int(experiment_name[4:8]),
                    'weather': wsta,
                    'results': results,
                    'planting_date': planting_date
                })
                
            except Exception as e:
                print(f"❌ Error processing {experiment_name}: {str(e)}")
        else:
            print(f"\n⚠️  SRX file not found: {srx_path}")
    
    print(f"\n=== ✅ All experiments completed ===")
    print(f"📁 Results saved in '{output_dir}' directory")
    
    if len(all_results) > 0:
        print("\n=== Generating year comparison chart ===")
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('UFBA1401/1601/1701 草莓生长关键指标对比图', fontsize=14, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
        
        for i, exp in enumerate(all_results):
            name = exp['name']
            results = exp['results']
            dap = results['dap']
            color = colors[i % len(colors)]
            
            axes[0, 0].plot(dap, results['biomass'], label=f'{name} 生物量', color=color, linewidth=2)
            axes[0, 1].plot(dap, results['leaf_area_index'], label=f'{name} 叶面积指数', color=color, linewidth=2)
            axes[1, 0].plot(dap, results['root_depth'], label=f'{name} 根系深度', color=color, linewidth=2)
            axes[1, 1].plot(dap, results['photosynthesis'], label=f'{name} 光合速率', color=color, linewidth=2)
        
        axes[0, 0].set_title('生物量随天数变化')
        axes[0, 0].set_xlabel('天数')
        axes[0, 0].set_ylabel('生物量')
        axes[0, 0].legend(loc='upper left', fontsize=8)
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        axes[0, 1].set_title('叶面积指数随天数变化')
        axes[0, 1].set_xlabel('天数')
        axes[0, 1].set_ylabel('叶面积指数')
        axes[0, 1].legend(loc='upper left', fontsize=8)
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        axes[1, 0].set_title('根系深度随天数变化')
        axes[1, 0].set_xlabel('天数')
        axes[1, 0].set_ylabel('根系深度')
        axes[1, 0].legend(loc='upper left', fontsize=8)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        axes[1, 1].set_title('光合速率随天数变化')
        axes[1, 1].set_xlabel('天数')
        axes[1, 1].set_ylabel('光合速率')
        axes[1, 1].legend(loc='upper left', fontsize=8)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        comparison_file = os.path.join(output_dir, "年份对比图.png")
        fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"💾 Comparison chart saved to: {comparison_file}")
        plt.close(fig)
    
    if len(all_results) > 0:
        print("\n=== Generating biomass chart ===")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('UFBA1401 草莓生物量随天数变化', fontsize=16, fontweight='bold')
        
        exp = all_results[0]
        results = exp['results']
        dap = results['dap']
        
        ax.plot(dap, results['biomass'], 'b-', label='总生物量', linewidth=3)
        ax.plot(dap, results['leaf_biomass'], 'g-', label='叶片', linewidth=2.5)
        ax.plot(dap, results['stem_biomass'], 'k-', label='茎秆', linewidth=2.5)
        ax.plot(dap, results['root_biomass'], 'r-', label='根系', linewidth=2.5)
        ax.plot(dap, results['fruit_biomass'], 'm-', label='果实', linewidth=2.5)
        
        ax.set_xlabel('天数', fontsize=12, fontweight='bold')
        ax.set_ylabel('生物量 (g/株)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        biomass_file = os.path.join(output_dir, "UFBA1401_生物量变化.png")
        fig.savefig(biomass_file, dpi=150, bbox_inches='tight')
        print(f"💾 Biomass chart saved to: {biomass_file}")
        plt.close(fig)
    
    if len(all_results) > 0:
        print("\n=== Generating photosynthesis chart ===")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('UFBA1401 草莓光合速率随天数变化', fontsize=16, fontweight='bold')
        
        exp = all_results[0]
        results = exp['results']
        dap = results['dap']
        
        ax.plot(dap, results['photosynthesis'], 'g-', label='光合速率', linewidth=3)
        
        ax.set_xlabel('天数', fontsize=12, fontweight='bold')
        ax.set_ylabel('光合速率', fontsize=12, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        photosynthesis_file = os.path.join(output_dir, "UFBA1401_光合速率变化.png")
        fig.savefig(photosynthesis_file, dpi=150, bbox_inches='tight')
        print(f"💾 Photosynthesis chart saved to: {photosynthesis_file}")
        plt.close(fig)
    
    if len(all_results) > 0:
        print("\n=== Generating LAI chart ===")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('UFBA1401 草莓叶面积指数随天数变化', fontsize=16, fontweight='bold')
        
        exp = all_results[0]
        results = exp['results']
        dap = results['dap']
        
        ax.plot(dap, results['leaf_area_index'], '#2ca02c', label='叶面积指数', linewidth=3)
        
        ax.set_xlabel('天数', fontsize=12, fontweight='bold')
        ax.set_ylabel('叶面积指数 (m^2/m^2)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        lai_file = os.path.join(output_dir, "UFBA1401_叶面积指数变化.png")
        fig.savefig(lai_file, dpi=150, bbox_inches='tight')
        print(f"💾 LAI chart saved to: {lai_file}")
        plt.close(fig)
    
    if len(all_results) > 0:
        print("\n=== Generating individual charts for all experiments ===")
        
        chart_configs = [
            {'name': '生物量变化', 'column': 'biomass', 'ylabel': '生物量 (g/株)', 'color': '#1f77b4'},
            {'name': '光合速率变化', 'column': 'photosynthesis', 'ylabel': '光合速率', 'color': '#2ca02c'},
            {'name': '叶面积指数变化', 'column': 'leaf_area_index', 'ylabel': '叶面积指数 (m^2/m^2)', 'color': '#2ca02c'},
            {'name': '根系深度变化', 'column': 'root_depth', 'ylabel': '根系深度', 'color': '#d62728'},
        ]
        
        for chart_config in chart_configs:
            for exp in all_results:
                name = exp['name']
                results = exp['results']
                dap = results['dap']
                
                plt.close('all')
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.suptitle(f'{name} 草莓{chart_config["name"]}', fontsize=16, fontweight='bold')
                
                ax.plot(dap, results[chart_config['column']], chart_config['color'], 
                        label=chart_config['name'].replace('变化', ''), linewidth=3)
                
                ax.set_xlabel('天数', fontsize=12, fontweight='bold')
                ax.set_ylabel(chart_config['ylabel'], fontsize=12, fontweight='bold')
                ax.legend(fontsize=12, loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                
                output_file = os.path.join(output_dir, f"{name}_{chart_config['name']}.png")
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"💾 {name}_{chart_config['name']}.png")
                plt.close(fig)
    
    print("\n=== Opening generated charts ===")
    first_chart = os.path.join(output_dir, "UFBA1401.png")
    if os.path.exists(first_chart):
        try:
            os.startfile(first_chart)
            print(f"🖼️  Opened: {first_chart}")
        except:
            try:
                import subprocess
                subprocess.run(['start', first_chart], shell=True)
                print(f"🖼️  Opened: {first_chart}")
            except Exception as e:
                print(f"⚠️  Could not open chart: {str(e)}")
                print(f"Please manually open: {first_chart}")
    
    comparison_chart = os.path.join(output_dir, "年份对比图.png")
    if os.path.exists(comparison_chart):
        try:
            os.startfile(comparison_chart)
            print(f"🖼️  Opened: {comparison_chart}")
        except:
            try:
                import subprocess
                subprocess.run(['start', comparison_chart], shell=True)
                print(f"🖼️  Opened: {comparison_chart}")
            except Exception as e:
                print(f"⚠️  Could not open comparison chart: {str(e)}")
                print(f"Please manually open: {comparison_chart}")
