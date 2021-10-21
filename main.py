import datetime

import yaml

from src import model

with open("optimization_config.yaml", "rb") as h:
    config = yaml.safe_load(h)

optim = model.OptimizeDay(config)
optim.run_optimization(datetime.datetime(*config['task_config']['start_date']),
                       datetime.datetime(*config['task_config']['end_date']))
solution = optim.get_solution()
solution.to_csv(config['output_solution_path'])

solution_check = model.OptimizationSolution(data_folder=config['data_folder_path'], solution_path=config['output_solution_path'])
solution_check.calculate_all_data()
status = solution_check.solution_fullcheck()
