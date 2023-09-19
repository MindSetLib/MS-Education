## Дообучение больших языковых моделей (LLM) в задачах суммаризации и детоксификации

### Загрузка данных и весов моделей

- Скачайте zip-файл по ссылке https://disk.yandex.ru/d/_qImUZdCGdBMtQ
- Распакуйте его в директории проекта

### Установка окружения

Выполнить из рабочей директории:

```
pyenv local 3.9.15
poetry shell
poetry install
ipython kernel install --user --name nlp --display-name="nlp"
```

### Демонстрационный ноутбук

[mset.space - LLM_ft.ipynb](./notebooks/mset.space-LLM_ft.ipynb)

### Скрипты для различных этапов

- [расчет метрик для сравнения полного файнтьюнинга и PEFT](./scriptss/ft_rouge.py)
- [полный файнтьюнинг LLM](./scriptss/instruct_ft.py)
- [PEFT](./scriptss/peft.py)
- [применение RL (PPO) для детоксификации LLM](./scriptss/rlhf_detox.py)