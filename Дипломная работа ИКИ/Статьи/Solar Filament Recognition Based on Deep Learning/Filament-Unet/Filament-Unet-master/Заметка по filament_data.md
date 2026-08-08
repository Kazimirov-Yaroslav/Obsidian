## Импорты

### Эпоха и совместимость (особенности)
- `from __future__ import print_function` — совместимость с Python 2; в Py3 ничего не делает. Маркер: код писали в эпоху Py2.
- `from keras import ...` — standalone **Keras 2**, а не `tf.keras`. Особенность: в современном Keras 3 модуля `keras.preprocessing.image` больше нет → код не запустится без старого окружения (Python ≤ 3.11 + TF 2.15). Для чтения окружение не нужно.

### Кто за что отвечает (проверено по использованию в файле)
| Импорт | Роль | Где используется |
|---|---|---|
| `ImageDataGenerator` (keras) | потоковая подгрузка батчей + аугментация на лету | `trainGenerator`, `validationGenerator` |
| `numpy as np` | все операции с массивами | `adjustData`, `geneTrainNpy`, `labelVisualize`, `COLOR_DICT` |
| `os` | пути, обход папок | `testGenerator`, `saveResult`, `geneTrainNpy` |
| `glob` | поиск файлов по маске `*.jpg` | `geneTrainNpy` |
| `skimage.io as io` | чтение/сохранение картинок | `io.imread` (`testGenerator`, `geneTrainNpy`), `io.imsave` (`saveResult`) |
| `skimage.transform as trans` | resize | `testGenerator` |

### Мёртвые импорты — в этом файле не используются
- `array_to_img` — ни одного вызова;
- `matplotlib.pyplot as plt` — ни одного вызова;
- `keras.backend as K` — в этом файле ни одного вызова.
Наследие оригинала zhixuhao/unet. Вывод: не тратить время на поиск их использования здесь.
- [ ] Проверить, где оживает `K` — вероятно, в других файлах (loss/метрики).

### Особенности, которые импорты дают по сути
- **Две библиотеки для картинок делят роли**: Keras — потоковое чтение с аугментацией для train/valid (генераторы на `yield`); skimage — точечное чтение/сохранение для test и отладки. То есть обучение — стриминг батчей, тест — простой перебор файлов.
- Стек типичен для Keras-сегментации 2017–2019 (файл — адаптация zhixuhao/unet): неясные места гуглятся по именам функций оригинала.



### COLOR_DICT — палитра классов (наследие multiclass-режима)
- `obj1 = [128,128,128]` (серый) — класс 0, `obj2 = [128,0,0]` (тёмно-красный) — класс 1.
- Используется только в `saveResult` и только при `flag_multi_class = True`.
- В проекте сегментация бинарная, `flag_multi_class = False` → предсказания сохраняются как есть, палитра не работает. Наследие оригинала zhixuhao/unet.
- [ ] Проверить по вызовам в `filament_training.py` / `filament_predict.py`, что `flag_multi_class` действительно нигде не True.

# Функции

## adjustData

«Нормализатор» данных: приводит снимок и маску к виду, который ожидает сеть. Два режима: мультикласс (наследие) и бинарный (рабочий).

### Сигнатура и аргументы
```python
def adjustData(img,mask,flag_multi_class,num_class):
```

| Аргумент | Значение |
|---|---|
| `img` | батч снимков, форма (B, 512, 512, 1). Значения зависят от вызывающего: сырые 0–255 (`trainGenerator`, `geneTrainNpy`) или уже [0,1] (`validationGenerator`) |
| `mask` | соответствующий батч масок той же формы |
| `flag_multi_class` | переключатель режима: True — мультикласс, False — бинарный. В проекте — False |
| `num_class` | число классов; нужен только мультикласс-ветке (здесь = 2) |

### Блок 1 — мультикласс-ветка (наследие, в проекте не работает)
```python
if(flag_multi_class):
    img = img / 255.0
```
Нормализация снимка без всяких проверок (в отличие от бинарной ветки — там есть условие).

```python
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
```
Отбрасываем канальное измерение: батч (B,H,W,1) → (B,H,W), одиночная картинка (H,W,1) → (H,W). Нужно, потому что для one-hot у каждого пикселя должен быть «номер класса», а не канал.

```python
    new_mask = np.zeros(mask.shape + (num_class,))
    for i in range(num_class):
        new_mask[mask == i,i] = 1
```
One-hot кодирование: нулевой массив формы (…, H, W, num_class); пиксели, равные i, получают 1 в канале i. `mask == i` — булева маска по пиксельным осям, `i` выбирает канал.

```python
    new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
    mask = new_mask
```
Reshape (B,H,W,C) → (B, H*W, C): старый формат «категориальный по пикселям» для softmax-кроссэнтропии на каждый пиксель. Наблюдение: ветка `else` тернарника — мёртвый код: мы уже внутри `if flag_multi_class`, условие всегда истинно. След копипаста из оригинала zhixuhao/unet.

### Блок 2 — бинарная ветка (рабочая)
```python
elif(np.max(img) > 1):
```
Триггер: ветка срабатывает, только если данные ещё **не** нормализованы. Это негласный контракт с вызывающими: кто уже сделал rescale — тот ветку пропускает (см. таблицу ниже).

```python
    img = img / 255.0
    mask = mask /255.0
```
Приводим оба массива к [0, 1].

```python
    mask[mask > 0.19] = 1
    mask[mask <= 0.19] = 0
```
Бинаризация маски порогом 0.19 (≈ 48/255). Зачем: маски лежат в `.jpg`, и JPEG-сжатие оставляет «серые» пиксели вокруг нитей; порог превращает всё, что светлее ~48, в 1, остальное — в 0. На выходе маска строго 0/1.

### Блок 3 — возврат
```python
return (img,mask)
```
Готовая пара. В бинарном режиме: `img` в [0,1], `mask` строго 0/1, формы не меняются — (B, 512, 512, 1).

### Кто вызывает и с какими данными
| Вызывающий | `img` на входе | `elif` срабатывает? | Маска после вызова |
|---|---|---|---|
| `trainGenerator` | сырые 0–255 | да | строго 0/1 |
| `geneTrainNpy` | сырые 0–255 | да | строго 0/1 |
| `validationGenerator` | уже [0,1] (`rescale=1/255`) | **нет** | «мягкая», с JPEG-шумом |

- [ ] Проверить по `filament_training.py` / `filament_predict.py`, что `flag_multi_class = False` во всех вызовах.

Примечание: функция — чистый numpy, без Keras. Когда появится окружение, её можно будет прогнать на одной паре файлов за секунду — идеальный первый эксперимент.


## trainGenerator

Фабрика обучающего потока: бесконечный генератор батчей «снимок + маска» с аугментацией на лету.

### Сигнатура и аргументы
```python
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
flag_multi_class = False,num_class = 2,save_to_dir = 'data/filament/train/test/',target_size = (512,512),seed = 1):
```

| Аргумент | Значение |
|---|---|
| `batch_size` | сколько пар «снимок+маска» в одном батче |
| `train_path` | корневая папка, внутри которой лежат подпапки |
| `image_folder` | имя подпапки со снимками |
| `mask_folder` | имя подпапки с масками |
| `aug_dict` | словарь аугментаций: ключи — параметры `ImageDataGenerator` (повороты, отражения…). Один и тот же словарь идёт в оба генератора — основа парности |
| `image_color_mode` / `mask_color_mode` = `"grayscale"` | чтение в один канал; батч имеет форму (B, 512, 512, 1) |
| `image_save_prefix` / `mask_save_prefix` = `"image"` / `"mask"` | префиксы имён, под которыми аугментированные пары сбрасываются на диск |
| `flag_multi_class` = `False` | режим бинарной сегментации; транзитом в `adjustData` |
| `num_class` = `2` | число классов; транзитом в `adjustData` |
| `save_to_dir` | папка, куда генератор сохраняет аугментированные пары (отладка/визуализация; вероятно, так в репозитории появилась папка `aug`) |
| `target_size` = `(512,512)` | все картинки приводятся к размеру 512×512 — это и есть вход сети |
| `seed` = `1` | зерно случайности; одинаковое у обоих генераторов — вторая половина парности |

### Блок 1 — два генератора с одинаковыми настройками
```python
image_datagen = ImageDataGenerator(**aug_dict)
mask_datagen = ImageDataGenerator(**aug_dict)
```
`**aug_dict` распаковывает словарь в именованные параметры конструктора. Создаются два **независимых** объекта, но с идентичными настройками аугментаций — первое условие парности.

### Блок 2 — поток снимков
```python
image_generator = image_datagen.flow_from_directory(
    train_path,
    classes = [image_folder],
    class_mode = None,
    color_mode = image_color_mode,
    target_size = target_size,
    batch_size = batch_size,
    save_to_dir = save_to_dir,
    save_prefix  = image_save_prefix,
    seed = seed)
```
`flow_from_directory` возвращает бесконечный итератор батчей. Параметры: `classes=[image_folder]` — читать только из указанной подпапки; `class_mode=None` — не возвращать метки (обычно меткой служит имя папки, но здесь «метки» приедут отдельным потоком масок); `color_mode`, `target_size`, `batch_size` — приведение к форме (B, 512, 512, 1); `save_to_dir`+`save_prefix` — дамп аугментированных батчей на диск; `seed` — зерно перемешивания и трансформаций. Файлы читаются в алфавитном порядке.

### Блок 3 — поток масок
```python
mask_generator = mask_datagen.flow_from_directory(
    train_path,
    classes = [mask_folder],
    class_mode = None,
    color_mode = mask_color_mode,
    target_size = target_size,
    batch_size = batch_size,
    save_to_dir = save_to_dir,
    save_prefix  = mask_save_prefix,
    seed = seed)
```
Тот же код, но для масок. Важно: имена файлов в папках снимков и масок **совпадают**, значит алфавитный порядок одинаков и i-й снимок соответствует i-й маске.

### Блок 4 — спаривание и выдача
```python
train_generator = zip(image_generator, mask_generator)
for (img,mask) in train_generator:
    img,mask = adjustData(img,mask,flag_multi_class,num_class)
    yield (img,mask)
```
- `zip` соединяет два потока в пары батч-в-батч.
- Почему трансформации совпадают: у обоих генераторов одинаковые `seed` и параметры → одинаковые последовательности случайных поворотов/отражений, применяемых к соответственным батчам. Итог: снимок и его маска искажаются **одинаково** — та самая парная аугментация из заметки про данные.
- `adjustData` нормализует и бинаризует пару. Здесь `ImageDataGenerator` создан **без** `rescale`, поэтому приходят сырые 0–255 и бинарная ветка `adjustData` срабатывает.
- `yield` делает функцию генератором: данные не грузятся все сразу, а стримятся по батчу — экономия памяти; Keras-обучение тянет из него, пока не кончится эпоха.
- [ ] Проверить в `filament_training.py`: с какими аргументами вызывается (какая папка масок, что в `aug_dict`).

## validationGenerator

Тот же шаблон, но **без аугментации** — валидация должна измерять качество на чистых данных.

### Сигнатура и аргументы
```python
def validationGenerator(batch_size,valid_path,image_folder,mask_folder,image_color_mode = "grayscale",mask_color_mode = "grayscale",flag_multi_class = False,num_class = 2,target_size = (512,512),seed = 1):
```
Отличия от `trainGenerator`: нет `aug_dict` (аугментаций нет), нет `save_to_dir`/префиксов (ничего не сохраняется). Остальные аргументы те же.

### Блок 1 — генераторы только с рескейлом
```python
validimg_datagen = ImageDataGenerator(rescale=1. / 255)
validmask_datagen= ImageDataGenerator(rescale=1. / 255)
```
Единственное преобразование — деление на 255 сразу в генераторе.

### Блок 2 — два потока
```python
validimg_generator= validimg_datagen.flow_from_directory(
    valid_path,
    classes = [image_folder],
    color_mode=image_color_mode,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=seed)
validmask_generator= validmask_datagen.flow_from_directory(
    valid_path,
    classes = [mask_folder],
    color_mode=mask_color_mode,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=seed)
```
`shuffle=True` с одинаковым `seed` у обоих — пары не рассинхронизируются при перемешивании.

### Блок 3 — спаривание и выдача
```python
valid_generator = zip(validimg_generator, validmask_generator)
for (img1,mask1) in valid_generator:
    img1,mask1 = adjustData(img1,mask1,flag_multi_class,num_class)
    yield (img1,mask1)
```
Следствие (уже отмечено в разделе про `adjustData`): данные приходят уже в [0,1], условие `np.max(img) > 1` ложно → бинаризация маски **не выполняется** → в валидационных целях остаётся JPEG-шум. Несостыковка train/valid.

### Сводка train vs valid
| | train | valid |
|---|---|---|
| Аугментация | да, из `aug_dict` | нет |
| Rescale в генераторе | нет (сырые 0–255) | да (1/255) |
| Бинаризация масок в `adjustData` | срабатывает | не срабатывает |
| Сохранение пар на диск | да (`save_to_dir`) | нет |