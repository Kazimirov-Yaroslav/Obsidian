---
term: Пулинг
aliases:
  - Pooling
  - пулинг
  - max pooling
  - global average pooling
tags:
  - ml/dictionary
category: архитектура
---
# Пулинг (Pooling)

**Пулинг** (pooling layer) — операция в свёрточных нейронных сетях, выполняющая **понижающую дискретизацию (downsampling)** и агрегацию признаков из предыдущего слоя. Операция уменьшает пространственные размеры признаковых карт (feature maps), увеличивая receptive field и снижая вычислительную сложность.

> [!abstract] Коротко
> Пулинг решает три задачи одновременно:
>
> 1. **Уменьшение размерности** — снижает потребление памяти и FLOPs.
> 2. **Трансляционная инвариантность** — признаки становятся устойчивее к малым сдвигам.
> 3. **Увеличение receptive field** — позволяет сети охватывать более крупные объекты.
>
> Для сегментации солнечных пятен пулинг играет двоякую роль: с одной стороны, он помогает охватывать крупные группы пятен (через иерархию), с другой — может терять информацию о мелких порах и тонких границах полутени.

---

## 📑 Содержание

- [[#Что такое пулинг и зачем он нужен]]
- [[#Базовые типы пулинга]]
- [[#Глобальный пулинг (Global Pooling)]]
- [[#Пространственно-пирамидальный пулинг (SPP)]]
- [[#Адаптивный пулинг (Adaptive Pooling)]]
- [[#Fractional Max Pooling]]
- [[#Атруc-пространственно-пирамидальный пулинг (ASPP)]]
- [[#Pyramid Pooling Module (PPM)]]
- [[#Attention Pooling]]
- [[#Альтернативы традиционному пулингу]]
- [[#Сравнительная таблица методов]]
- [[#Математика: влияние на receptive field и градиенты]]
- [[#Специфика солнечных изображений]]
- [[#Практика в PyTorch]]
- [[#Масштабирование и перспектива]]
- [[#Частые ошибки]]
- [[#Литература и источники]]
- [[#Вывод]]
- [[#Связанные термины]]

---

## Что такое пулинг и зачем он нужен

### Формальное определение

Пулинг — это отображение:

$$
\text{Pool}: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H' \times W'}
$$

где $H' \leq H$, $W' \leq W$. Обычно $H' = \lfloor (H - k) / s \rfloor + 1$ для ядра $k$ и шага $s$.

### Роль в современной архитектуре

> [!tip] 1. Иерархия признаков
> В [[Энкодер (Encoder)|энкодере]] последовательные пулинги формируют иерархию: от низкоуровневых деталей (границы, текстуры) до высокоуровневой семантики (объекты, контекст).

> [!tip] 2. Вычислительная эффективность
> Без пулинга каждый следующий слой работал бы с картой исходного размера — FLOPs росли бы экспоненциально с глубиной.

> [!tip] 3. Трансляционная инвариантность
> Пулинг делает признаки устойчивее к малым сдвигам: если максимум в окне смещается, выход не меняется.

> [!tip] 4. Расширение receptive field
> Каждый пулинг вдвое увеличивает receptive field всех последующих слоёв.

---

## Базовые типы пулинга

### Max Pooling (максимальный пулинг)

**Max Pooling** выбирает максимальное значение в каждом окне:

$$
y_{i,j,c} = \max_{(p,q) \in R_{i,j}} x_{p,q,c}
$$

где $R_{i,j}$ — окно (обычно $k \times k$) с позицией $(i, j)$, $c$ — канал.

Для стандартного пулинга с размером ядра $k$ и шагом $s$:
- Выходной размер: $H_{out} = \lfloor (H_{in} - k) / s \rfloor + 1$.
- Типичный выбор: $k = 2, s = 2$ → уменьшение вдвое.

**Свойства:**

> [!success] Преимущества
> - Сохраняет **наиболее сильные активации** — «самые важные» признаки.
> - **Инвариантен к малым сдвигам** — если максимум смещается в пределах окна, выход не меняется.
> - **Не имеет обучаемых параметров** — детерминированная операция.
> - **Индуцирует разреженность** — большинство значений отбрасываются.

> [!note] Интерпретация как envelope extraction
> Max pooling можно интерпретировать как **извлечение огибающей** признаковых карт. Он сохраняет только самые яркие отклики в каждом регионе, что полезно для задач, где важна детекция наиболее выраженных признаков (например, границ пятен).

### Average Pooling (средний пулинг)

**Average Pooling** вычисляет среднее значение в каждом окне:

$$
y_{i,j,c} = \frac{1}{|R_{i,j}|} \sum_{(p,q) \in R_{i,j}} x_{p,q,c}
$$

**Свойства:**

> [!success] Преимущества
> - Сохраняет **усреднённую информацию** обо всех пикселях окна.
> - Работает как **низкочастотный фильтр** с последующим даунсемплингом.
> - Менее чувствителен к выбросам, чем max pooling.

> [!warning] Недостатки
> - Размывает признаки, что может приводить к потере резких границ.
> - Слабые признаки «разбавляются» сильными.

**Интерпретация:** average pooling эквивалентен применению **равномерного фильтра** $1/k^2$ ко всем пикселям окна с последующей субдискретизацией.

### Сравнение Max и Average Pooling

| Характеристика | Max Pooling | Average Pooling |
|---|---|---|
| **Сохраняет** | Сильнейшие активации | Усреднённый отклик |
| **Чувствительность к шуму** | Устойчив | Чувствителен |
| **Сохранение границ** | Хорошо (резкие переходы) | Плохо (размытие) |
| **Информация** | Теряет слабые признаки | Сохраняет все признаки |
| **Инвариантность** | К сдвигам в пределах окна | К малым вариациям |

**Эмпирический вывод:** для задач **классификации** max pooling обычно предпочтительнее, для **сегментации** — зависит от задачи. Для солнечных пятен max pooling даёт более чёткие границы объектов.

---

## Глобальный пулинг (Global Pooling)

### Global Average Pooling (GAP)

**Global Average Pooling** вычисляет среднее по всему пространственному размеру для каждого канала:

$$
y_c = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,c}
$$

Впервые предложен в **Network-in-Network** (Lin et al., 2013) как замена полносвязным слоям.

> [!success] Преимущества
> - **Резкое снижение параметров:** убирает полносвязные слои.
> - **Регуляризация:** каждый канал «отвечает» за один семантический признак.
> - **Инвариантность к размеру входа:** можно подавать изображения разного размера.
> - **Интерпретируемость:** каждый канал — признак конкретного класса.

**Применение:**
- Последний слой классификаторов (ResNet, EfficientNet).
- Bottleneck в автоэнкодерах.
- Модули SE (Squeeze-and-Excitation).

### Global Max Pooling (GMP)

Аналогично GAP, но берёт максимум:

$$
y_c = \max_{i,j} x_{i,j,c}
$$

Менее распространён, но иногда даёт лучшие результаты для задач, где важен именно наиболее выраженный признак.

### Сравнение с полносвязными слоями

| Характеристика | GAP/GMP | FC-слои |
|---|---|---|
| Параметры | 0 | $H \cdot W \cdot C_{in} \cdot C_{out}$ |
| Риск переобучения | Низкий | Высокий |
| Размер входа | Любой | Фиксированный |
| Интерпретируемость | Высокая | Низкая |

---

## Пространственно-пирамидальный пулинг (SPP)

**Spatial Pyramid Pooling** (He et al., 2014) — ключевая работа, устранившая требование фиксированного размера входа.

### Идея

Вместо одного пулинга применяется **пирамида пулингов** на разных уровнях:

- Уровень 1: $1 \times 1$ пулинг (GAP/GMP).
- Уровень 2: $2 \times 2$ пулинг.
- Уровень 3: $3 \times 3$ пулинг.
- Уровень 4: $6 \times 6$ пулинг.

Каждый уровень выдаёт фиксированное число признаков, независимо от размера входа:

$$
\text{Размер выхода уровня } l = l^2 \cdot C
$$

Общий выход:

$$
z = \text{Concat}\left[\text{Pool}_{1\times1}(x), \text{Pool}_{2\times2}(x), \text{Pool}_{3\times3}(x), \ldots\right]
$$

### Математика

Для уровня с $n_l \times n_l$ бинов размер ядра и шаг вычисляются адаптивно:

$$
k_l = \left\lceil \frac{H}{n_l} \right\rceil, \quad s_l = \left\lfloor \frac{H}{n_l} \right\rfloor
$$

где $H$ — размер входа. Это гарантирует, что на выходе всегда будет $n_l \times n_l$ бинов.

### Значение для сегментации

> [!tip] Мультимасштабность
> SPP стал предшественником всех современных мультимасштабных модулей: **ASPP** в DeepLab, **PPM** в PSPNet, **FPN** в детекции. Идея «агрегировать информацию на разных масштабах» оказалась критически важной для сегментации объектов разного размера.

---

## Адаптивный пулинг (Adaptive Pooling)

**Adaptive Pooling** — обобщение пулинга, где задаётся **желаемый размер выхода**, а размер ядра и шаг вычисляются автоматически.

### Математика

Для целевого размера выхода $n_{out} \times n_{out}$ и размера входа $n_{in} \times n_{in}$:

$$
k = \left\lfloor \frac{n_{in}}{n_{out}} \right\rfloor + \left\lceil \frac{n_{in} \bmod n_{out}}{n_{out}} \right\rceil
$$

$$
s = \left\lfloor \frac{n_{in}}{n_{out}} \right\rfloor
$$

Это позволяет применять пулинг к входам **любого размера**, получая выход фиксированного размера.

### Применение

```python
import torch.nn as nn

pool = nn.AdaptiveAvgPool2d((1, 1))  # эквивалент GAP
pool = nn.AdaptiveAvgPool2d((7, 7))  # выход 7x7 для любого входа
```

**Когда использовать:**
- В bottleneck перед полносвязными слоями.
- В пирамидальных модулях (PPM, SPP).
- Когда размер входа заранее неизвестен.

---

## Fractional Max Pooling

**Fractional Max Pooling** (Graham, 2014) — пулинг с **нецелым** коэффициентом уменьшения.

### Идея

Вместо фиксированного ядра $k \times k$, используется переменный размер окна и стохастическое разбиение признаковой карты:

$$
\text{Уменьшение} = \alpha \in (1, 2) \text{ (например, } \sqrt{2} \approx 1.41\text{)}
$$

Пулинг-регионы выбираются случайно с определёнными ограничениями, что даёт:

- **Стохастичность** — разные прогоны дают разные разбиения.
- **Более плавное уменьшение** — не всегда в 2 раза.
- **Регуляризационный эффект** — снижает переобучение.

### Математика

Разбиение признаковой карты генерируется случайно из последовательности $p_0 < p_1 < \ldots < p_{N_{out}}$, где:

$$
p_i = \left\lfloor \frac{i \cdot (N_{in} - 1)}{N_{out} - 1} + u_i \right\rfloor
$$

где $u_i$ — случайное возмущение.

### Преимущества

> [!success] Почему это работает
> Работа Грэхема показала, что FMP значительно снижает переобучение на малых датасетах, улучшая state-of-the-art на CIFAR-10 и CIFAR-100. Для солнечных изображений с малым датасетом это может быть полезно как форма регуляризации.

---

## Атруc-пространственно-пирамидальный пулинг (ASPP)

**Atrous Spatial Pyramid Pooling** (Chen et al., 2016, 2017) — ключевой модуль семейства DeepLab.

### Идея

Вместо пространственного пулинга на разных уровнях используется **атрус-свёртка** с разными dilation rates:

- $1 \times 1$ свёртка (обычная).
- $3 \times 3$ свёртка с dilation rate 6 (receptive field 13×13).
- $3 \times 3$ свёртка с dilation rate 12 (receptive field 25×25).
- $3 \times 3$ свёртка с dilation rate 18 (receptive field 37×37).
- $1 \times 1$ свёртка после GAP (глобальный контекст).

### Математика atrous-свёртки

Для atrous свёртки с dilation rate $r$:

$$
y_{i,j} = \sum_{p,q} x_{i + r \cdot p,\; j + r \cdot q} \cdot w_{p,q}
$$

Receptive field растёт без увеличения числа параметров и без потери пространственного разрешения.

### DeepLabv3: усовершенствованный ASPP

В DeepLabv3 (Chen et al., 2017) добавлен **image-level pooling** с последующей 1×1 свёрткой и batch normalization, что улучшило глобальный контекст.

> [!tip] Для солнечных изображений
> ASPP идеально подходит для сегментации солнечных пятен:
> - Мелкие поры видны при dilation rate 1.
> - Средние пятна — при rate 6.
> - Крупные группы — при rate 12–18.
> - Глобальный контекст диска — через image-level pooling.

---

## Pyramid Pooling Module (PPM)

**Pyramid Pooling Module** (Zhao et al., 2017) — центральный модуль **PSPNet**.

### Архитектура

PPM применяет **адаптивный average pooling** на 4 уровнях, затем 1×1 свёртку для снижения размерности и апсемплинг обратно:

1. Уровень 1×1 (GAP).
2. Уровень 2×2.
3. Уровень 3×3.
4. Уровень 6×6.

Каждый уровень:
- Adaptive Average Pooling до размера $n \times n$.
- 1×1 свёртка (reduce channels).
- BatchNorm + ReLU.
- Bilinear upsample до исходного размера.

Финальная конкатенация:

$$
y = \text{Concat}[x,\; \text{Up}(P_1),\; \text{Up}(P_2),\; \text{Up}(P_3),\; \text{Up}(P_6)]
$$

### Зачем нужен PPM

> [!note] Проблема контекста
> Стандартные [[Энкодер (Encoder)|энкодеры]] теряют глобальный контекст из-за последовательных даунсемплингов. PPM восстанавливает **контекстную информацию разных масштабов**, что критично для корректной классификации пикселей.
>
> Для солнечных пятен это означает, что модель может отличать:
> - Тёмную область на краю диска (limb darkening) от пятна.
> - Мелкую пору от шума.
> - Одиночное пятно от группы пятен.

---

## Attention Pooling

**Attention Pooling** — пулинг, который учится **взвешивать** разные регионы в зависимости от их важности.

### Attentive Pooling Networks

**Attentive Pooling** (Santos et al., 2016) использует двухсторонний attention для взвешенного агрегирования.

Для матрицы признаков $X \in \mathbb{R}^{H \times W \times C}$:

$$
\alpha = \text{softmax}(\text{MLP}(X)) \in \mathbb{R}^{H \times W}
$$

$$
y = \sum_{i,j} \alpha_{i,j} \cdot x_{i,j}
$$

### Self-Attentive Pooling

**Self-Attentive Pooling** (Chen, 2022) использует self-attention для выбора информативных регионов:

1. Patch embedding признаков.
2. Multi-head self-attention.
3. Sigmoid для генерации attention mask.
4. Взвешенный пулинг.

### Mix-Pooling Strategy

**SPEM** (Zhong, 2022) комбинирует global max и average pooling для создания более богатого контекста:

$$
y = \text{MLP}(\text{Concat}[\text{GAP}(X),\; \text{GMP}(X)]) \odot X
$$

Это основа современных attention-модулей (CBAM, SE, ECA).

---

## Альтернативы традиционному пулингу

### Strided Convolution

**Strided Convolution** — свёртка с шагом $s > 1$, которая выполняет даунсемплинг и извлечение признаков одновременно:

$$
y_{i,j} = \sum_{p,q} x_{i \cdot s + p,\; j \cdot s + q} \cdot w_{p,q}
$$

> [!success] Преимущества перед max pooling
> - **Обучаемость:** веса адаптируются под задачу.
> - **Не теряет информацию:** учитывает все пиксели в окне.
> - **Единая операция:** даунсемплинг совмещён с извлечением признаков.

> [!warning] Недостатки
> - Больше параметров.
> - Может быть менее устойчив к сдвигам.

**Современная тенденция:** ResNet, EfficientNet, ConvNeXt используют strided convolution вместо max pooling для даунсемплинга.

### CARAFE: Content-Aware ReAssembly of FEatures

**CARAFE** (Wang et al., 2019) — обучаемый оператор для апсемплинга (обратной операции).

**Идея:** вместо билинейной интерполяции CARAFE **учится собирать** апсемплированные пиксели из соседних признаков с учётом контента:

1. Генерирует kernel для каждого выходного пикселя (content-aware).
2. Применяет kernel к локальной окрестности.
3. Получает более точные признаки.

**Для солнечных пятен:** CARAFE может улучшить качество границ полутени и мелких пор за счёт контента-зависимого апсемплинга.

### Subsampling через MLP

**Multi-Layer Pooling** (2020) заменяет классический пулинг небольшими MLP:

$$
y = \text{MLP}(\text{Flatten}(\text{Patch}(X)))
$$

Это даёт более гибкое агрегирование, но требует больше параметров.

### Soft Pooling

**Soft Pooling** использует softmax-взвешивание для плавного агрегирования:

$$
y_{i,j} = \frac{\sum_{(p,q) \in R} x_{p,q} \cdot e^{x_{p,q}}}{\sum_{(p,q) \in R} e^{x_{p,q}}}
$$

Сохраняет больше информации, чем max pooling, и более «гладкий», чем average.

---

## Сравнительная таблица методов

| Метод | Параметры | Детерминированный | Сохраняет детали | Когда использовать |
|---|---|---|---|---|
| **Max Pooling** | Нет | Да | Сильные признаки | Классика, детекция границ |
| **Average Pooling** | Нет | Да | Усреднённый контекст | Сглаживание, классификация |
| **Global Average** | Нет | Да | Один вектор | Классификаторы, bottleneck |
| **Adaptive Pooling** | Нет | Да | Заданный размер | Произвольный вход |
| **Fractional Max** | Нет | **Нет** (стохастичен) | Плавное уменьшение | Регуляризация |
| **SPP** | Нет | Да | Мультимасштаб | Классификация разного размера |
| **ASPP** | Да (свёртки) | Да | Мультимасштаб + контекст | Сегментация |
| **PPM** | Да (1×1 conv) | Да | Пирамида контекста | Сегментация |
| **Attention Pooling** | Да (MLP/attention) | Да (после обучения) | Важные регионы | Избирательная агрегация |
| **Strided Conv** | Да | Да | Обучаемый даунсемплинг | Современные архитектуры |
| **CARAFE** | Да | Да | Content-aware | Качественный апсемплинг |

---

## Математика: влияние на receptive field и градиенты

### Влияние на receptive field

Каждый пулинг с stride $s$ увеличивает receptive field всех последующих слоёв. Для $L$ уровней с пулингом stride 2:

$$
RF_L = RF_0 \cdot 2^L
$$

То есть после 4 уровней даунсемплинга receptive field вырастает в 16 раз.

### Градиенты max pooling

Max pooling **недифференцируем** в обычном смысле, но имеет субградиент. Для выхода $y_{i,j}$:

$$
\frac{\partial y_{i,j}}{\partial x_{p,q}} = \begin{cases}
1, & \text{если } x_{p,q} = \max_{R_{i,j}} x \\
0, & \text{иначе}
\end{cases}
$$

То есть градиент идёт только по **одному** элементу — максимуму. Это делает max pooling разреженным оператором.

### Градиенты average pooling

Average pooling дифференцируем:

$$
\frac{\partial y_{i,j}}{\partial x_{p,q}} = \frac{1}{|R_{i,j}|}, \quad \forall (p,q) \in R_{i,j}
$$

Градиент распределяется равномерно по всем элементам окна.

### Сравнение потока градиента

| Метод | Градиент | Разреженность |
|---|---|---|
| **Max Pooling** | Идёт только через максимум | Высокая |
| **Average Pooling** | Равномерно по всем элементам | Низкая |
| **Strided Conv** | Распределяется с весами | Средняя |

> [!note] Следствие для обучения
> Max pooling создаёт более разреженный поток градиента, что может приводить к «мёртвым» нейронам при неудачной инициализации. Average pooling распределяет градиент равномернее, но может «размывать» сигналы.

---

## Специфика солнечных изображений

### Особенности данных

Для сегментации солнечных пятен важно учитывать:

- **Мультимасштабность объектов:** поры (несколько пикселей) vs крупные группы (сотни пикселей).
- **Нечёткие границы:** полутень плавно переходит в фон.
- **Limb darkening:** градиент яркости к краю диска.
- **Малый датасет:** несколько сотен размеченных изображений.

### Рекомендуемые стратегии

> [!success] Для энкодера (даунсемплинг)
> - **Strided convolution** (ResNet-стиль) или **max pooling 2×2** (классический U-Net).
> - **Не слишком агрессивный даунсемплинг** на первых уровнях — сохранить мелкие поры.
> - **Stride 2** на каждом уровне (не 4).

> [!success] Для bottleneck (мультимасштабность)
> - **ASPP** — лучший выбор для солнечных пятен:
>   - Dilation rates 1, 6, 12, 18 покрывают поры → группы.
>   - Image-level pooling даёт глобальный контекст диска.
> - **PPM** — альтернатива с более лёгкой реализацией.

> [!success] Для декодера (апсемплинг)
> - **Bilinear + 3×3 conv** — стандарт, без артефактов.
> - **CARAFE** — если нужно лучшее качество границ.
> - **НЕ** использовать transposed convolution (checkerboard artifacts).

> [!success] Для регуляризации
> - **Fractional Max Pooling** на одном из уровней — как форма стохастической регуляризации.
> - Полезно при малом датасете солнечных изображений.

### Что используют в работах

- **U-Net с max pooling 2×2** — стандарт в работах по сегментации солнечных пятен (Mourato et al., 2024).
- **DeepLabv3+ с ASPP** — для сегментации крупных структур (корональные дыры, филаменты).
- **PSPNet с PPM** — реже, но встречается в работах по активным регионам.
- **SIPNet с SAHI** (Fan et al., 2023) — мультимасштабный подход для солнечных пятен.

---

## Практика в PyTorch

### Классический Max/Average Pooling

```python
import torch.nn as nn

# Max pooling 2x2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling 2x2
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling
gap = nn.AdaptiveAvgPool2d((1, 1))

# Adaptive pooling с заданным выходом
adaptive = nn.AdaptiveAvgPool2d((7, 7))
```

> [!note]+ Построчный разбор
> 1. `nn.MaxPool2d(kernel_size=2, stride=2)` — стандартный max pooling, уменьшающий разрешение вдвое.
> 2. `nn.AvgPool2d(kernel_size=2, stride=2)` — аналогичный average pooling.
> 3. `nn.AdaptiveAvgPool2d((1, 1))` — эквивалент Global Average Pooling, всегда выдаёт вектор.
> 4. `nn.AdaptiveAvgPool2d((7, 7))` — адаптивный пулинг, подстраивающий размер ядра под вход.

### ASPP модуль

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        modules = []

        # 1x1 свёртка
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        ))

        # Атруc-свёртки с разными rates
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                         padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image-level features (GAP + 1x1 conv)
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2),
                     out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Image-level: upsample обратно к размеру входа
        img_level = self.convs[-1](x)
        img_level = F.interpolate(img_level, size=size,
                                  mode='bilinear', align_corners=False)
        res.append(img_level)

        x = torch.cat(res, dim=1)
        return self.project(x)
```

> [!note]+ Построчный разбор
> 1. Четыре параллельные ветви: 1×1 conv, три atrous conv (rates 6, 12, 18), image-level pooling.
> 2. `padding=rate, dilation=rate` — обеспечивает одинаковый размер выхода для всех ветвей.
> 3. Image-level pooling → GAP → 1×1 conv → upsample к размеру входа.
> 4. `project` — финальная проекция и dropout для регуляризации.

### PPM (Pyramid Pooling Module)

```python
class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )
            for bin in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(bins),
                     out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        priors = [x]
        for stage in self.stages:
            prior = stage(x)
            prior = F.interpolate(prior, size=size,
                                 mode='bilinear', align_corners=False)
            priors.append(prior)
        x = torch.cat(priors, dim=1)
        return self.bottleneck(x)
```

> [!note]+ Построчный разбор
> 1. Четыре уровня адаптивного average pooling: 1×1, 2×2, 3×3, 6×6.
> 2. Каждый уровень: adaptive pool → 1×1 conv (reduce channels) → norm → ReLU.
> 3. Апсемплинг каждого уровня обратно к размеру входа через bilinear interpolation.
> 4. Конкатенация всех уровней с исходными признаками.
> 5. Финальная 3×3 свёртка — bottleneck, агрегирующий мультимасштабную информацию.

### Fractional Max Pooling (упрощённая реализация)

```python
def fractional_max_pool2d(x, output_size):
    """Упрощённая реализация Fractional Max Pooling."""
    N, C, H, W = x.shape
    out_h, out_w = output_size

    # Вычисляем индексы разбиений (в оригинале стохастические)
    h_seq = torch.linspace(0, H, out_h + 1).long()
    w_seq = torch.linspace(0, W, out_w + 1).long()

    # Применяем max pooling в каждом регионе
    output = torch.zeros(N, C, out_h, out_w, device=x.device)
    for i in range(out_h):
        for j in range(out_w):
            region = x[:, :, h_seq[i]:h_seq[i+1], w_seq[j]:w_seq[j+1]]
            output[:, :, i, j] = region.amax(dim=(-2, -1))

    return output
```

> [!warning] Ограничение
> Это детерминированная реализация. Настоящий Fractional Max Pooling использует стохастические разбиения, что даёт регуляризационный эффект. Для полноценной реализации используйте специализированные библиотеки или PyTorch через `torch.nn.FractionalMaxPool2d` (если доступен).

---

## Масштабирование и перспектива

### Если появится больше вычислительных ресурсов

- **Большие входы** (1024×1024, 2048×2048) → можно использовать ASPP с большими dilation rates.
- **Больше данных** → можно применять более сложные модули (PPM + ASPP одновременно).
- **Многокарточное обучение** → можно экспериментировать с CARAFE и другими обучаемыми операторами.

### Если появится больше данных

- **Больше данных** → меньше потребность в регуляризации через Fractional Max Pooling.
- Можно использовать **трансформерные блоки внимания** вместо простого attention pooling.
- Можно обучать **контент-зависимые операторы** (CARAFE) с нуля.

### Перспективные направления

> [!note]+ На будущее
> - **Обучаемые операторы** (CARAFE, CARAFE++): замена статических пулингов на обучаемые.
> - **Attention-based pooling**: интеграция self-attention для адаптивного агрегирования.
> - **Мультимодальный пулинг**: разные типы пулинга для разных каналов (например, для магнитограмм и континуумов).
> - **Физически-обоснованный пулинг**: использование знаний о физике Солнца (например, симметрия, limb darkening) для специализированных операторов.

---

## Частые ошибки

> [!danger]- 1. Слишком агрессивный даунсемплинг на первых уровнях
> Теряются мелкие объекты (поры, малые пятна).
> **Решение:** использовать stride 2 (а не 4) на каждом уровне.

> [!danger]- 2. Использование transposed convolution для апсемплинга
> Checkerboard-артефакты в масках, особенно при высоких скоростях обучения.
> **Решение:** использовать bilinear upsample + 3×3 conv.

> [!danger]- 3. Отсутствие мультимасштабности в bottleneck
> Модель не может одновременно сегментировать мелкие и крупные объекты.
> **Решение:** использовать ASPP или PPM.

> [!danger]- 4. Использование только max pooling в encoder
> Теряется информация о слабых признаках (например, полутени).
> **Решение:** комбинировать с average pooling или использовать strided conv.

> [!danger]- 5. Игнорирование image-level pooling в ASPP
> Модель не видит глобальный контекст, что критично для крупных структур.
> **Решение:** всегда добавлять GAP-ветвь в ASPP.

> [!danger]- 6. Использование BatchNorm при малых батчах
> Нестабильные статистики портят признаки.
> **Решение:** использовать GroupNorm или SyncBatchNorm.

> [!danger]- 7. Фиксированные dilation rates для всех изображений
> Оптимальные rates зависят от разрешения входа.
> **Решение:** адаптировать rates под размер входа или использовать Adaptive ASPP.

---

## Литература и источники

### Общие обзоры

1. **Gholamalinezhad, Khosravi (2020).** *Pooling Methods in Deep Neural Networks, a Review.* [arXiv:2009.07485](https://arxiv.org/abs/2009.07485)
   Подробный обзор методов пулинга с таксономией.

2. **Zafar et al. (2022).** *A Comparison of Pooling Methods for Convolutional Neural Networks.* [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/12/17/8643)
   Эмпирическое сравнение различных методов пулинга.

3. **Zhao et al. (2024).** *A improved pooling method for convolutional neural networks.* [Nature Scientific Reports](https://www.nature.com/articles/s41598-024-51258-6)
   Современный обзор с предложениями улучшений.

### Классические работы

4. **Lin et al. (2013).** *Network In Network.* [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
   Введение Global Average Pooling.

5. **He et al. (2014).** *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.* [arXiv:1406.4729](https://arxiv.org/abs/1406.4729)
   Пространственно-пирамидальный пулинг (SPP).

6. **Graham (2014).** *Fractional Max-Pooling.* [arXiv:1412.6071](https://arxiv.org/abs/1412.6071)
   Стохастический пулинг с нецелым коэффициентом уменьшения.

### Пулинг в сегментации

7. **Chen et al. (2016).** *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.* [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)
   Введение ASPP (Atrous Spatial Pyramid Pooling).

8. **Chen et al. (2017).** *Rethinking Atrous Convolution for Semantic Image Segmentation (DeepLabv3).* [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)
   Улучшенный ASPP с image-level pooling.

9. **Zhao et al. (2016).** *Pyramid Scene Parsing Network (PSPNet).* [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
   Pyramid Pooling Module (PPM).

### Attention Pooling

10. **Santos et al. (2016).** *Attentive Pooling Networks.* [arXiv:1602.03609](https://arxiv.org/abs/1602.03609)
    Двухстороннее внимание для пулинга.

11. **Chen (2022).** *Self-Attentive Pooling for Efficient Deep Learning.* [arXiv:2209.07659](https://arxiv.org/abs/2209.07659)
    Self-attention для эффективного пулинга.

12. **Zhong (2022).** *Mix-Pooling Strategy for Attention Mechanism.* [arXiv:2208.10322](https://arxiv.org/abs/2208.10322)
    Комбинирование max и average pooling в attention.

### Альтернативы и улучшения

13. **Wang et al. (2019).** *CARAFE: Content-Aware ReAssembly of FEatures.* [arXiv:1905.02188](https://arxiv.org/abs/1905.02188)
    Content-aware оператор для апсемплинга.

14. **Strided Convolution Instead of Max Pooling for Memory Efficiency.** [ResearchGate](https://www.researchgate.net/publication/334399497)
    Сравнение strided convolution и max pooling.

### Солнечная физика

15. **Mourato et al. (2024).** *Automatic sunspot detection through semantic and instance segmentation.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197623018201)
    U-Net с max pooling для сегментации пятен.

16. **Sayez et al. (2023).** *SunSCC: Segmenting, Grouping and Classifying Sunspots.* [AGU](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JA031548)
    Автоматическая сегментация солнечных пятен.

17. **Fan et al. (2023).** *SIPNet & SAHI: Multiscale Sunspot Extraction for High-Resolution Images.* [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/14/1/7)
    Мультимасштабный подход для солнечных пятен.

---

## Вывод

> [!success] Итого
> **Пулинг** — операция понижающей дискретизации и агрегации признаков, играющая ключевую роль в формировании иерархии представлений.
>
> Ключевые принципы:
>
> - **Базовые типы:** Max Pooling (сильные активации), Average Pooling (усреднённый контекст), Global Pooling (один вектор на канал).
> - **Мультимасштабные модули:** SPP (классика), ASPP (для сегментации), PPM (пирамида контекста).
> - **Альтернативы:** Strided Convolution (обучаемый даунсемплинг), CARAFE (content-aware апсемплинг), Attention Pooling (взвешенное агрегирование).
> - **Регуляризация:** Fractional Max Pooling как стохастический даунсемплинг.
>
> Математические основы:
>
> - Каждый пулинг увеличивает receptive field всех последующих слоёв.
> - Max pooling — разреженный оператор (градиент только через максимум).
> - Average pooling распределяет градиент равномерно.
> - Адаптивный пулинг подстраивает размер ядра под желаемый выход.
>
> Для задачи сегментации солнечных пятен:
>
> ```text
> Энкодер:
>   - Strided conv или max pooling 2×2 на каждом уровне
>   - Сохранять skip-connections всех уровней
>
> Bottleneck:
>   - ASPP с rates (6, 12, 18) + image-level pooling
>   - Альтернатива: PPM с бинами (1, 2, 3, 6)
>
> Декодер:
>   - Bilinear upsample + 3×3 conv на каждом уровне
>   - НЕ использовать transposed convolution
>
> Специальные приёмы:
>   - Fractional Max Pooling для регуляризации (опционально)
>   - Attention gates на skip-connections
> ```

---

## Связанные термины

- [[Энкодер (Encoder)|Энкодер]] — использует пулинг для даунсемплинга и формирования иерархии
- [[Декодер (Decoder)|Декодер]] — использует обратные операции (апсемплинг) для восстановления разрешения
- [[Сегментация (Segmentation)|Сегментация]] — задача, для которой пулинг формирует многоуровневые признаки
- [[Батч (Batch)|Батч]] — размер батча влияет на выбор нормализации после пулинга
- [[Нормализация (Normalization)|Нормализация]] — часто применяется после пулинга
- [[Маска (Mask)|Маска]] — целевая переменная, качество которой зависит от правильного пулинга
- [[Функция потерь (Loss Function)|Функция потерь]] — определяет, как обучать сеть с пулингом