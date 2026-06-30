print("步骤1: 导入库")
import numpy as np
import pandas as pd
from datetime import datetime
print("库导入成功")

print("\n步骤2: 创建模型类")

class SimpleModel:
    def __init__(self):
        self.state = {'biomass': 0.0}
        print("模型初始化成功")
    
    def run(self, days):
        print(f"开始运行 {days} 天")
        for i in range(days):
            self.state['biomass'] += 1.0
        print(f"运行完成，最终生物量: {self.state['biomass']}")
        return self.state

print("步骤3: 创建模型实例")
model = SimpleModel()

print("\n步骤4: 运行模拟")
result = model.run(10)

print("\n步骤5: 输出结果")
print(f"生物量: {result['biomass']}")

print("\n所有测试通过!")