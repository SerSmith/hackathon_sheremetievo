# Цифровой прорыв Шереметьево

## Реализованная функциональность
   1. Находит оптимальное расположение мест стоянок для ВС
   2. Строит на полученном решении интерактивный дашборд

## Особенность проекта
   1. Решение полностью напиисано на open source - бесплатных библиотеках
   2. Постановка задачи оптимизации, ее решение и визуализация осуществляются независимо друг от друга
   3. Солвер оценивает нижнюю границу потенциального значения целевой
   4. Возможно ограничить время решения задачи
   5. Можно проводить анализ “что-если”
   6. Возможно получить ускорение решения - за счет использования проприетарных солверов, более грубой агрегации по времени или использование более мощного оборудования
   7. Возможность использования теплного старта


## Cтек технологий
   1. Формулировка задачи: Pyomo
   2. Оптимизация: cbc solver
   3. Визуалииизация: streamlit 

## Среда запуска
   Теоритически CBC солвер можно установить куда угодно, но удобнее это сделать на linux

## Установка

1. Для запуска оптимизации Вам надо:
   1. Установить requirements: pip install -r requirements.txt
   2. Установить cbc солвер: https://github.com/coin-or/Cbc
   3. Заполнить конфиг optimization_config.yaml
   4. Запустить main.py
2. Для запуска дэшборда Вам надо:
   1. Установить requirements: pip install -r requirements.txt
   2. Запустить дэшборд (из корня запустить streamlit run src/dashboard.py)


### Разработчики
Кузнецов Сергей Юрьевич fullstack https://t.me/test@just_nickname

Мошков Николай Евгеньевич fullstack https://t.me/test@Affernus

Кузнецова Марина Юрьевна fullstack https://t.me/test@Mila_601


