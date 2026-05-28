# Лабораторная работа №5


- код анализа CSV находится в `src/metetl/analysis/aggregations.py`;
- код скачивания и обработки изображений находится в `src/metetl/images/processing.py`;
- `print()` заменены на логгер из `src/metetl/logging_config.py`;
- добавлен CLI в `src/metetl/cli.py`;
- добавлен запуск пакета через `python -m metetl`;
- добавлена сборка через `pyproject.toml`;
- добавлены тесты `unittest`.


## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Если PowerShell ругается на запуск скриптов:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Команды CLI

### Справка

```powershell
metetl --help
```

### Подготовка JSON с метаданными изображений

```powershell
metetl prepare --csv .\data\MetObjects.txt --output .\data\to_download.json --num 1 --max-attempts 200
```

Команда берет случайные `Object ID` из CSV, обращается к MET API, пропускает объекты без изображений и останавливается, когда нашла нужное количество картинок. Для быстрой сдачи достаточно `--num 1`.

### Скачивание и обработка одной картинки

```powershell
metetl process --input .\data\to_download.json --output .\images --num 1 --workers 1
```

Команда скачивает картинку, делает grayscale, размытие Гаусса, Sobel-границы и сохраняет результат.

### Анализ CSV

```powershell
metetl analyze --csv .\data\MetObjects.txt --output-dir .\data\plots
```

Команда запускает анализ и сохраняет таблицы и графики в `data/plots`.

## Логирование

Логи пишутся в:

```text
logs/app.log
```

В консоль выводятся краткие сообщения уровня `INFO`, а в файл пишутся подробные сообщения уровня `DEBUG` с временем, файлом и строкой.

## Тесты

```powershell
python -m unittest discover -s tests -v
```

## Сборка

```powershell
pip install build
python -m build
```

После сборки появится папка `dist` с файлами `.tar.gz` и `.whl`.

## Установка wheel

```powershell
pip install .\dist\metetl_lab5_from_lab34-0.1.0-py3-none-any.whl
```

Имя файла может немного отличаться, его можно посмотреть в папке `dist`.
