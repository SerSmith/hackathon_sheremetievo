import datetime

import yaml
import cloudpickle
from src import model

with open("optimization_config.yaml", "rb") as h:
    config = yaml.safe_load(h)


if not config['optimization_parameters']['model_saver']['load_model']:
    # запускаем оптимизацию
    optim = model.OptimizeDay(config)
    optim.make_problem_obj(datetime.datetime(*config['task_config']['start_date']),
                        datetime.datetime(*config['task_config']['end_date']))

    if config['optimization_parameters']['model_saver']['save_model']:
        model_to_save = optim.get_model()
        with open(config['optimization_parameters']['model_saver']['model_path'], 'wb') as h:
            cloudpickle.dump(model_to_save, h)
else:
    with open(config['optimization_parameters']['model_saver']['model_path'], 'rb') as h:
        loaded_model = cloudpickle.load(h)
        optim = model.OptimizeDay(config)
        optim.set_model(loaded_model)
        print('модель загружена')

optim.set_warm_start()
optim.solve_model()
solution = optim.get_solution()
solution.to_csv(config['optimization_parameters']['output_solution_path'])

# проверяем рассчет 
solution_check = model.OptimizationSolution(data_folder=config['optimization_parameters']['data_folder_path'], solution_path=config['optimization_parameters']['output_solution_path'])
solution_check.calculate_all_data()
status = solution_check.solution_fullcheck()
result = solution_check.calculate_all_data()
result.to_csv(config['optimization_parameters']['output_results_path'])

final_solution = solution_check.get_solution_send_format()
final_solution.to_csv(config['optimization_parameters']['final_solution_path'])