"""Simplified CROPGRO-Strawberry model in pure Python (no external dependencies)."""

import math
from datetime import datetime, timedelta


class PlantState:
    def __init__(self):
        self.biomass = 0.0
        self.leaf_area_index = 0.1
        self.root_depth = 5.0
        self.fruit_number = 0.0
        self.fruit_biomass = 0.0
        self.leaf_biomass = 0.0
        self.stem_biomass = 0.0
        self.root_biomass = 0.0
        self.phenological_stage = "GERMINATION"
        self.development_rate = 0.0
        self.crown_number = 1.0
        self.runner_number = 0.0


def calc_daylength(latitude, day_of_year):
    declination = 23.45 * math.sin(math.radians(360 * (day_of_year - 80) / 365))
    lat_rad = math.radians(latitude)
    term = -math.tan(lat_rad) * math.tan(math.radians(declination))
    if term >= 1.0:
        return 0.0
    elif term <= -1.0:
        return 24.0
    else:
        return 24.0 * math.acos(term) / math.pi


def thermal_time(tmin, tmax, tbase, topt, tmax_th):
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


def photosynthesis(solar_radiation, tmax, tmin, rue, tbase, topt, k_light, lai, co2):
    tavg = (tmax + tmin) / 2.0
    if tavg <= tbase:
        temp_effect = 0.0
    elif tavg >= topt:
        temp_effect = 1.0
    else:
        temp_effect = (tavg - tbase) / (topt - tbase)
    co2_effect = 1.0 + 0.11 * math.log(co2 / 400.0)
    light_interception = 1.0 - math.exp(-k_light * lai)
    return solar_radiation * rue * temp_effect * co2_effect * light_interception


def transpiration(solar_radiation, tmax, tmin, rh, lai):
    tavg = (tmax + tmin) / 2.0
    et0 = 0.0023 * solar_radiation * math.sqrt(tmax - tmin) * (tavg + 17.8)
    kc = 0.3 + 0.7 * (1.0 - math.exp(-0.7 * lai))
    return et0 * kc


def water_stress(field_capacity, wilting_point, root_depth, rainfall, transpiration):
    available_water = (field_capacity - wilting_point) * root_depth
    effective_rainfall = rainfall * 0.7
    deficit = max(0.0, transpiration - effective_rainfall)
    if deficit == 0.0:
        return 0.0
    else:
        stress_factor = min(1.0, deficit / available_water)
        return stress_factor


def maintenance_resp(leaf_biomass, stem_biomass, root_biomass, fruit_biomass, tmin, tmax):
    tavg = (tmin + tmax) / 2.0
    temp_factor = 2.0 ** ((tavg - 20.0) / 10.0)
    resp_leaf = leaf_biomass * 0.03 * temp_factor
    resp_stem = stem_biomass * 0.015 * temp_factor
    resp_root = root_biomass * 0.01 * temp_factor
    resp_fruit = fruit_biomass * 0.01 * temp_factor
    return resp_leaf + resp_stem + resp_root + resp_fruit


class CropgroStrawberry:
    
    def __init__(self, latitude, planting_date, soil_properties, cultivar_params):
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
        return calc_daylength(self.latitude, day_of_year)
    
    def calculate_thermal_time(self, tmin, tmax):
        return thermal_time(tmin, tmax, 
                           self.cultivar['tbase'], 
                           self.cultivar['topt'], 
                           self.cultivar['tmax_th'])
    
    def update_phenology(self, thermal_time_today):
        self.thermal_time += thermal_time_today
        current_stage = self.plant_state.phenological_stage
        stages = list(self.phenology_stages.keys())
        current_index = stages.index(current_stage)
        
        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            if self.thermal_time >= self.phenology_stages[next_stage]:
                self.plant_state.phenological_stage = next_stage
    
    def calculate_photosynthesis(self, solar_radiation, tmax, tmin, co2=400):
        return photosynthesis(
            solar_radiation, tmax, tmin,
            self.cultivar['rue'],
            self.cultivar['tbase'],
            self.cultivar['topt'],
            self.cultivar['k_light'],
            self.plant_state.leaf_area_index,
            co2
        )
    
    def calculate_transpiration(self, solar_radiation, tmax, tmin, rh, wind_speed):
        lai = self.plant_state.leaf_area_index
        base_transpiration = transpiration(solar_radiation, tmax, tmin, rh, lai)
        wind_modifier = 1.0 + 0.1 * (wind_speed - 2.0)
        wind_modifier = max(0.5, min(2.0, wind_modifier))
        return base_transpiration * wind_modifier
    
    def partition_biomass(self, daily_biomass):
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
        if self.plant_state.phenological_stage in ['VEGETATIVE', 'FLORAL_INDUCTION']:
            self.plant_state.runner_number += 0.1 * self.plant_state.crown_number
    
    def update_crowns(self):
        if self.plant_state.phenological_stage in ['VEGETATIVE', 'FLORAL_INDUCTION', 'FLOWERING']:
            self.plant_state.crown_number += 0.02 * self.plant_state.crown_number
    
    def update_fruits(self):
        stage = self.plant_state.phenological_stage
        
        if stage == 'FLOWERING':
            new_fruits = self.cultivar['potential_fruits_per_crown'] * self.plant_state.crown_number * 0.1
            self.plant_state.fruit_number += new_fruits
        elif stage == 'FRUIT_SET':
            new_fruits = self.cultivar['potential_fruits_per_crown'] * self.plant_state.crown_number * 0.2
            self.plant_state.fruit_number += new_fruits
    
    def calculate_water_stress(self, rainfall, transpiration):
        field_capacity = self.soil['field_capacity']
        wilting_point = self.soil['wilting_point']
        root_depth = self.plant_state.root_depth / 100.0
        return water_stress(field_capacity, wilting_point, root_depth, rainfall, transpiration)
    
    def calculate_maintenance_respiration(self, tmin, tmax):
        return maintenance_resp(
            self.plant_state.leaf_biomass,
            self.plant_state.stem_biomass,
            self.plant_state.root_biomass,
            self.plant_state.fruit_biomass,
            tmin, tmax
        )
    
    def simulate_day(self, weather_data):
        self.days_after_planting += 1
        current_date = datetime.strptime(weather_data['date'], '%Y-%m-%d')
        day_of_year = current_date.timetuple().tm_yday
        
        daylength = self.calculate_daylength(day_of_year)
        thermal_time_today = self.calculate_thermal_time(weather_data['tmin'], weather_data['tmax'])
        self.update_phenology(thermal_time_today)
        
        photosynthesis_val = self.calculate_photosynthesis(
            weather_data['solar_radiation'], weather_data['tmax'], weather_data['tmin'])
        
        transpiration_val = self.calculate_transpiration(
            weather_data['solar_radiation'], weather_data['tmax'], weather_data['tmin'],
            weather_data['rh'], weather_data['wind_speed'])
        
        water_stress_val = self.calculate_water_stress(weather_data['rainfall'], transpiration_val)
        photosynthesis_val *= (1 - water_stress_val)
        
        plant_density = 5.0
        daily_biomass = photosynthesis_val / plant_density
        
        maintenance_resp_val = self.calculate_maintenance_respiration(weather_data['tmin'], weather_data['tmax'])
        daily_biomass = max(0, daily_biomass - maintenance_resp_val)
        
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
            'water_stress': water_stress_val,
            'daylength': daylength,
            'photosynthesis': photosynthesis_val,
            'transpiration': transpiration_val
        })


def generate_weather_data(start_date, end_date, seed=42):
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    n_days = len(dates)
    
    import random
    random.seed(seed)
    
    weather_data = []
    for i, date in enumerate(dates):
        day_of_year = datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday
        seasonal_component = 10 * math.sin(2 * math.pi * (day_of_year - 172) / 365)
        
        tmax = 25.0 + seasonal_component + random.gauss(0, 3)
        tmin = 10.0 + seasonal_component + random.gauss(0, 2)
        
        solar_rad = 15.0 + 10.0 * math.sin(2 * math.pi * (day_of_year - 172) / 365) + random.gauss(0, 2)
        solar_rad = max(1.0, solar_rad)
        
        rainfall = 0.0
        if random.random() < 0.3:
            rainfall = random.expovariate(1/5)
        
        rh = 70.0 + random.gauss(0, 10)
        rh = max(20, min(100, rh))
        
        wind_speed = 2.0 + random.expovariate(1)
        
        weather_data.append({
            'date': date,
            'tmax': tmax,
            'tmin': tmin,
            'solar_radiation': solar_rad,
            'rainfall': rainfall,
            'rh': rh,
            'wind_speed': wind_speed
        })
    
    return weather_data


def run_example_simulation():
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
    
    weather_data = generate_weather_data(start_date, end_date)
    
    model = CropgroStrawberry(
        latitude=40.0,
        planting_date=start_date,
        soil_properties=soil_properties,
        cultivar_params=cultivar_params
    )
    
    for weather_day in weather_data:
        model.simulate_day(weather_day)
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("CROPGRO-Strawberry Model Simulation")
    print("=" * 60)
    
    model = run_example_simulation()
    
    print(f"\nSimulation completed for {model.days_after_planting} days")
    print(f"Planting date: {model.planting_date.strftime('%Y-%m-%d')}")
    print(f"Final phenological stage: {model.plant_state.phenological_stage}")
    print(f"Final thermal time: {model.thermal_time:.2f} degree-days")
    
    print("\n--- Final Biomass Results (g/plant) ---")
    print(f"  Total biomass: {model.plant_state.biomass:.2f}")
    print(f"  Fruit biomass: {model.plant_state.fruit_biomass:.2f}")
    print(f"  Leaf biomass: {model.plant_state.leaf_biomass:.2f}")
    print(f"  Stem biomass: {model.plant_state.stem_biomass:.2f}")
    print(f"  Root biomass: {model.plant_state.root_biomass:.2f}")
    
    print("\n--- Other Plant Indicators ---")
    print(f"  Leaf Area Index (LAI): {model.plant_state.leaf_area_index:.2f} m^2/m^2")
    print(f"  Root depth: {model.plant_state.root_depth:.2f} cm")
    print(f"  Fruit number: {model.plant_state.fruit_number:.2f} fruits/plant")
    print(f"  Crown number: {model.plant_state.crown_number:.2f} crowns/plant")
    print(f"  Runner number: {model.plant_state.runner_number:.2f} runners/plant")
    
    print("\n--- Phenological Development Timeline ---")
    stages_seen = []
    for result in model.results:
        if result['stage'] not in stages_seen:
            stages_seen.append(result['stage'])
            print(f"  {result['stage']}: reached at day {result['dap']}")
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)