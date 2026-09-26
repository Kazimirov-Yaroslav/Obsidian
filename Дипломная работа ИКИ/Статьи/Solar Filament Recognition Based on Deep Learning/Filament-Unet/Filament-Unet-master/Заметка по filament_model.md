## filament_model.py — импорты

```python
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as keras
from keras.utils import plot_model
```
- **Wildcard-импорты** (`import *`): всё содержимое модулей Keras высыпаются в пространство имён без префиксов — поэтому ниже работают «голые» `Input`, `Conv2D`, `MaxPooling2D`, `Model`, `Adam`. Для чтения удобно, в продакшне — дурной тон (конфликты имён).
- `numpy as np` (дважды!), `os`, `skimage.io`, `skimage.transform` — в этом файле **не используются**: наследие оригинала zhixuhao/unet, где функции работы с данными жили в том же файле.
- `from keras import backend as keras` — тоже не используется; обратите внимание, алиас `keras` здесь указывает на backend, а не на пакет.
- `plot_model` — рисование схемы архитектуры (`plot_model(model, show_shapes=True)`); требует pydot+graphviz — вероятно, под это в training/predict импортирован `pydotplus`.

## filament_unet — сигнатура и аргументы

```python
def filament_unet(pretrained_weights = None,input_size = (512,512,1)):
```

| Аргумент | Значение |
|---|---|
| `pretrained_weights` = `None` | путь к файлу весов; если не None — в конце грузятся в построенный граф. **В проекте не используется**: `filament_predict.py` вызывает `model.load_weights(...)` вручную |
| `input_size` = `(512,512,1)` | форма входного тензора: 512×512 из `target_size` генераторов + 1 канал grayscale. Контур «данные ↔ модель» замыкается |

Функция возвращает **скомпилированную** модель: граф строится цепочкой вызовов слоёв (Functional API), затем `Model(...)` и `compile(...)`.

## Блок 1 — энкодер: 4 уровня «conv×2 → dropout → pool»

```python
inputs = Input(input_size)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
```
Разбор аргументов `Conv2D` (паттерн повторится во всём файле):
- `64` — число фильтров: слой учит 64 карт признаков, каждая — свёртка ядра 3×3 по входу + bias;
- `3` — ядро 3×3;
- `activation='relu'` — нелинейность max(0,x); без неё стопка свёрок схлопнулась бы в одно линейное преобразование;
- `padding='same'` — нули по краям: H и W **не меняются**, размер уменьшает только пулинг;
- `kernel_initializer='he_normal'` — старт весов ~ N(0, sqrt(2/fan_in)): масштаб, подобранный под relu, чтобы сигнал не затухал и не взрывался по слоям.

- `Dropout(0.5)`: на обучении случайно обнуляет 50% активаций (остальные масштабируются ×2), мешая сети опираться на отдельные нейроны; на инференсе выключен. **В классическом U-Net и в оригинале zhixuhao dropout нет вообще** — здесь он на каждом уровне энкодера и в bottleneck: сильная регуляризация как ответ на 30 обучающих снимков.
- `MaxPooling2D((2,2))`: максимум в окне 2×2 → пространственное разрешение /2, каналы не трогаются; даёт инвариантность к малым сдвигам и растит рецептивное поле следующих свёрток.
- Логика уровня: две свёртки извлекают признаки → dropout → пулинг сжимает. Каналы при этом ×2 (ниже), компенсируя /2 по пространству: «информационная ёмкость» уровня примерно сохраняется.

Уровни 2–4 — тот же шаблон с удвоением каналов:
```python
conv2 = Conv2D(128, 3, ...)(pool1);  conv2 = Conv2D(128, 3, ...)(conv2)
drop2 = Dropout(0.5)(conv2);         pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
conv3 = Conv2D(256, 3, ...)(pool2);  conv3 = Conv2D(256, 3, ...)(conv3)
drop3 = Dropout(0.5)(conv3);         pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
conv4 = Conv2D(512, 3, ...)(pool3);  conv4 = Conv2D(512, 3, ...)(conv4)
drop4 = Dropout(0.5)(conv4);         pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
```
(в заметке аргументы свёрток сокращены до `...` — они идентичны блоку 1)

## Блок 2 — bottleneck: самая глубокая точка

```python
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)
```
Пулинга после conv5 **нет**: это дно сети, 32×32×1024. Здесь максимум рецептивное поле и максимум абстракции: каждый «пиксель» этого тензора видит всё исходное изображение. Обратите внимание: именно эти два слоя + первый блок декодера съедают ~75% всех параметров сети (см. таблицу).

## Блок 3 — декодер: upsample → skip-concat → conv×2, четыре раза

```python
up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)#drop4
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
```
- `UpSampling2D((2,2))`: nearest-neighbor ×2 — **без обучаемых параметров**, каждый пиксель дублируется квадратом 2×2; пространственный размер возвращаем, каналов не касаемся.
- `Conv2D(512, 2, ...)` — свёртка с ядром **2×2** сразу после апсемплинга: наследная причуда оригинала; содержательно «сглаживает» дубликаты nearest-neighbor (каждый квадрат 2×2 одинаковых значений перемалывается в новое) и одновременно减半 каналы 1024→512. В современных U-Net вместо этой пары обычно `Conv2DTranspose` или апсемплинг + conv 3×3.
- `merge6 = merge([conv4, up6], mode='concat', concat_axis=3)` — **skip connection, суть U-Net**: конкатенация по каналам признаковой карты энкодера того же разрешения (conv4 — тонкие детали) с картой декодера (up6 — семантика из глубины): (64,64,512) ⊕ (64,64,512) → (64,64,1024). Декодеру возвращаются детали, которые убил пулинг.
  - `concat_axis=3` — ось каналов (channels last).
  - `merge(..., mode='concat')` — **API Keras 1.x**; в Keras 2 заменено на `Concatenate(axis=3)([...])` / `concatenate([...], axis=3)` и позже удалено → ещё один маркер, что файл «как есть» на современном окружении не запустится.
  - Комментарий `#drop4` — след правки: в оригинале zhixuhao skip шёл из-под dropout (`drop4`); здесь заменён на `conv4` (до dropout), комментарий остался. У merge7/8/9 комментарии (#conv3, #conv2, #conv1) просто помечают энкодерного партнёра.
- `conv6×2`: свёртки учатся **смешивать** склеенные признаки и возвращают каналы к норме уровня (1024→512).

Дальше паттерн зеркально повторяется: каналы /2, пространство ×2 — до полного разрешения:
```python
up7 = Conv2D(256, 2, ...)(UpSampling2D(size = (2,2))(conv6))
merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
conv7 = Conv2D(256, 3, ...)(merge7);  conv7 = Conv2D(256, 3, ...)(conv7)
up8 = Conv2D(128, 2, ...)(UpSampling2D(size = (2,2))(conv7))
merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
conv8 = Conv2D(128, 3, ...)(merge8);  conv8 = Conv2D(128, 3, ...)(conv8)
up9 = Conv2D(64, 2, ...)(UpSampling2D(size = (2,2))(conv8))
merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
conv9 = Conv2D(64, 3, ...)(merge9);   conv9 = Conv2D(64, 3, ...)(conv9)
```

## Блок 4 — голова: вероятность на пиксель

```python
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
```
- `Conv2D(2, 3, relu)` — избыточный промежуточный слой на 2 канала; наследие оригинала (выросло из головы на 2 класса). Для понимания архитектуры не обязателен.
- `Conv2D(1, 1, sigmoid)` — смысловой выход. Свёртка 1×1 = поточечная линейная комбинация каналов (здесь двух) без касания пространства; сигмоида даёт вероятность «пиксель — филамент» в [0,1]. Выход: **(B, 512, 512, 1)** — ровно форма бинарной маски из `adjustData`: контур «модель ↔ данные» замыкается окончательно.

## Блок 5 — сборка и компиляция

```python
model = Model(input = inputs, output = conv10)
model.compile(optimizer = Adam(lr = 1e-4,beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])#,beta_1=0.9, beta_2=0.999,
#model.summary()
```
- `Model(input=..., output=...)` — Functional API: граф уже построен цепочкой вызовов выше, здесь лишь упаковывается; именованные аргументы `input=`/`output=` — старый стиль (сейчас `inputs=`/`outputs=`).
- `Adam(lr=1e-4)`: адаптивный оптимизатор; `beta_1/beta_2` выписаны явно, но равны значениям по умолчанию — избыточно. (`lr=` в новых Keras заменён на `learning_rate=`.)
- `loss='binary_crossentropy'`: поточечная бинарная кросс-энтропия сигмоидного выхода против 0/1-маски. Важная стыковка: эта loss ожидает цель формы (B,512,512,1) — ровно выход **бинарной** ветки `adjustData`; мультиклассовая (B, H*W, C) сюда не подошла бы → финальное доказательство, что мультикласс-ветка мёртвый код.
- `metrics=['accuracy']`: поточечная точность. **Осторожно для диплома**: на масках, где филамент — единицы процентов пикселей, accuracy оптимистична (тривиальный прогноз «всё ноль» даёт ~95%+). Реальные метрики статьи (dice/IoU) считаются вне этого кода.
- `#model.summary()` закомментирован; раскомментируйте в окружении — получите таблицу форм и параметров (или сверьтесь с посчитанной ниже).

## Блок 6 — веса и возврат

```python
if(pretrained_weights):
    model.load_weights(pretrained_weights)
return model
```
Если путь передан — веса загружаются в построенный граф (имена слоёв должны совпасть). В проекте аргумент не используется: `filament_predict.py` делает `filament_unet()` и затем `load_weights("filament_unet.hdf5")` вручную.

## Таблица форм (то, что дал бы model.summary())

| Блок | Слои | Форма выхода |
|---|---|---|
| inputs | Input | (B, 512, 512, 1) |
| conv1×2 | Conv2D 64, 3×3 | (B, 512, 512, 64) |
| pool1 | MaxPool 2×2 | (B, 256, 256, 64) |
| conv2×2 | Conv2D 128 | (B, 256, 256, 128) |
| pool2 | | (B, 128, 128, 128) |
| conv3×2 | Conv2D 256 | (B, 128, 128, 256) |
| pool3 | | (B, 64, 64, 256) |
| conv4×2 | Conv2D 512 | (B, 64, 64, 512) |
| pool4 | | (B, 32, 32, 512) |
| conv5×2 | Conv2D 1024 | (B, 32, 32, 1024) |
| up6 | UpSampling + Conv2D 512, 2×2 | (B, 64, 64, 512) |
| merge6 | concat([conv4, up6], axis 3) | (B, 64, 64, 1024) |
| conv6×2 | Conv2D 512 | (B, 64, 64, 512) |
| up7 + merge7 | concat([conv3, up7]) | (B, 128, 128, 512) |
| conv7×2 | Conv2D 256 | (B, 128, 128, 256) |
| up8 + merge8 | concat([conv2, up8]) | (B, 256, 256, 256) |
| conv8×2 | Conv2D 128 | (B, 256, 256, 128) |
| up9 + merge9 | concat([conv1, up9]) | (B, 512, 512, 128) |
| conv9×2 | Conv2D 64 | (B, 512, 512, 64) |
| голова | Conv2D 2, 3×3 → Conv2D 1, 1×1 sigmoid | (B, 512, 512, 1) |

## Параметры: формула, итог и сверка с hdf5

Формула для Conv2D: `(k_h·k_w·C_in + 1)·C_out` (+1 — bias на каждый фильтр).

| Блок | Параметров |
|---|---|
| conv1×2 | 37 568 |
| conv2×2 | 221 440 |
| conv3×2 | 885 248 |
| conv4×2 | 3 539 968 |
| conv5×2 (bottleneck) | 14 157 824 |
| up6 + conv6×2 | 9 176 576 |
| up7 + conv7×2 | 2 294 784 |
| up8 + conv8×2 | 426 368 |
| up9 + conv9×2 + голова | 107 845 |
| **Итого** | **30 847 621 ≈ 31 млн** |

**Сверка с файлом весов:** `save_weights_only=True` → в hdf5 лежат только веса. 30 847 621 × 4 байта (float32) ≈ 123,4 МБ ≈ 120 498 КиБ; файл `filament_unet.hdf5` весит 121 308 КБ (Windows) — сходится с точностью до служебных данных hdf5. Те самые «121 МБ» из первого скриншота — это веса ровно этой сети. Контур, открытый в день знакомства с папками, замкнулся.
Наблюдение: bottleneck + первый блок декодера = ~23 млн из 31 млн (~75%) — цена каналов 1024 и склейки 1024→512 на высоких разрешениях.

## Реликты файла (почему «как есть» не запустится сегодня)
- `merge(..., mode='concat', concat_axis=3)` — Keras 1.x, удалено в Keras 2.x;
- `Model(input=, output=)` — старые именованные аргументы;
- `Adam(lr=...)` — в новых Keras `learning_rate=`;
- standalone `keras`, а не `tf.keras`.
Всё это согласуется с прежним выводом: код писали под TF 1.x / ранний TF 2.x и Python ≤ 3.11.