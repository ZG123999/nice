print("步骤1: 导入库")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
print("库导入成功")

print("\n步骤2: 使用 datetime 创建日期范围")
start_date = datetime(2023, 5, 1)
end_date = datetime(2023, 10, 31)
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += timedelta(days=1)
print(f"日期范围创建成功，共 {len(dates)} 天")

print("\n步骤3: 生成随机天气数据")
n_days = len(dates)
np.random.seed(42)

day_of_year = np.array([d.timetuple().tm_yday for d in dates])
print(f"day_of_year 生成成功: {len(day_of_year)}")

seasonal_component = 10 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
print(f"seasonal_component 生成成功")

tmax = 25.0 + seasonal_component + np.random.normal(0, 3, n_days)
print(f"tmax 生成成功")

tmin = 10.0 + seasonal_component + np.random.normal(0, 2, n_days)
print(f"tmin 生成成功")

solar_rad = (15.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 172) / 365) 
            + np.random.normal(0, 2, n_days))
solar_rad = np.maximum(1.0, solar_rad)
print(f"solar_rad 生成成功")

rainfall = np.zeros(n_days)
rain_events = np.random.rand(n_days) < 0.3
rainfall[rain_events] = np.random.exponential(5, np.sum(rain_events))
print(f"rainfall 生成成功")

rh = 70.0 + np.random.normal(0, 10, n_days)
rh = np.clip(rh, 20, 100)
print(f"rh 生成成功")

wind_speed = 2.0 + np.random.exponential(1, n_days)
print(f"wind_speed 生成成功")

print("\n步骤4: 创建 DataFrame")
weather_df = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d') for d in dates],
    'tmax': tmax,
    'tmin': tmin,
    'solar_radiation': solar_rad,
    'rainfall': rainfall,
    'rh': rh,
    'wind_speed': wind_speed
})
print(f"DataFrame 创建成功，形状: {weather_df.shape}")

print("\n步骤5: 测试迭代 DataFrame")
count = 0
for _, row in weather_df.iterrows():
    count += 1
    if count <= 3:
        print(f"第 {count} 天: {row['date']}, tmax={row['tmax']:.1f}")
print(f"迭代完成，共 {count} 行")

print("\n所有步骤通过!")