---
term: Декодер
aliases:
  - Decoder
  - декодер
  - segmentation head
  - upsampling decoder
tags:
  - ml/dictionary
category: архитектура
---

# Декодер (Decoder)

**Декодер** — это «расширяющая» часть архитектуры encoder-decoder, отвечающая за восстановление пространственного разрешения и построение финальной попиксельной маски (или карты классов) из сжатого признакового представления энкодера.

> [!abstract] Коротко
> Декодер — это «зеркало» энкодера: там, где энкодер сжимает, декодер восстанавливает.
>
> Его три главные функции:
> 1. **Восстановление пространственного разрешения** от компактного bottleneck до полного размера входа.
> 2. **Интеграция мультимасштабных признаков** через skip-connections.
> 3. **Формирование финальной маски** (бинарной или многоклассовой).
>
> В современных архитектурах декодер часто называют **segmentation head** — головкой, превращающей признаки бэкбоуна в маску. Для сегментации солнечных пятен на слабом GPU важен баланс: декодер должен быть выразительным, но не избыточным по параметрам.

---

## 📑 Содержание

- [[#Что такое декодер и зачем он нужен]]
- [[#История: от FCN к современным декодерам]]
- [[#Математика декодера]]
- [[#Апсемплинг: способы повышения разрешения]]
- [[#Skip-connections: конкатенация vs сложение]]
- [[#Bottleneck: роль самого узкого места]]
- [[#Головка декодера (Head)]]
- [[#Типы современных декодеров]]
- [[#Современные трансформерные и универсальные декодеры]]
- [[#Promptable декодеры (SAM и SAM 2)]]
- [[#Выбор декодера под ограничения железа]]
- [[#Специфика солнечных изображений]]
- [[#Практика в PyTorch]]
- [[#Масштабирование и перспектива]]
- [[#Частые ошибки]]
- [[#Литература и источники]]
- [[#Вывод]]
- [[#Связанные термины]]

---

## Что такое декодер и зачем он нужен

### Формальное определение

Декодер — это отображение:

$$
D: \mathcal{Z} \to \mathcal{Y}, \quad \hat{y} = D(z; \theta_D)
$$

где:
- $z$ — признаки из bottleneck [[Энкодер (Encoder)|энкодера]] (и промежуточных уровней через skip-connections);
- $\mathcal{Y}$ — пространство выходов (маска сегментации, карта вероятностей);
- $\theta_D$ — обучаемые параметры декодера.

В полной системе: $\hat{y} = D(E(x))$, где $E$ — энкодер.

### Три главные функции декодера

> [!tip] 1. Восстановление пространственного разрешения
> Декодер последовательно повышает разрешение признаков от компактного bottleneck ($H/32 \times W/32$) до полного разрешения входа ($H \times W$).

> [!tip] 2. Интеграция мультимасштабных признаков
> Через skip-connections декодер получает признаки разных уровней энкодера, комбинируя семантику (высокий уровень) с геометрией (низкий уровень).

> [!tip] 3. Формирование финального предсказания
> На выходе декодера — маска сегментации: бинарная (фон/пятно) или многоклассовая (фон/умбра/полутень).

### Две роли декодера

**Роль 1: часть encoder-decoder архитектуры.** В U-Net, SegNet, DeepLabv3+, MaskFormer декодер — явно выделенная часть, зеркальная энкодеру.

**Роль 2: segmentation head.** В более общих системах (ResNet + FPN, Swin + UPerNet) декодером называют «голову», превращающую признаки бэкбоуна в маску. Эта терминология преобладает в библиотеках типа `segmentation_models.pytorch` и `MMSegmentation`.

---

## История: от FCN к современным декодерам

### FCN (Long et al., 2015) — первый декодер для сегментации

**Fully Convolutional Networks** первыми заменили полносвязные слои свёрточными и предложили идею декодера для сегментации:

- **FCN-32s:** один апсемплинг в 32 раза → грубая маска.
- **FCN-16s:** комбинация признаков pool5 и pool4, апсемплинг в 16 раз.
- **FCN-8s:** добавление pool3, апсемплинг в 8 раз → лучшая детализация.

Именно здесь родилась концепция skip-connections между энкодером и декодером (хотя авторы называли их «skip connections» в более узком смысле).

### SegNet (Badrinarayanan et al., 2015) — декодер с передачей индексов

SegNet ввёл явную пару encoder-decoder, где декодер использует **индексы max-pooling из энкодера** для точного восстановления границ:

- В энкодере запоминаются позиции максимумов.
- В декодере эти позиции используются для «разреженного» апсемплинга (unpooling).

**Плюс:** точные границы объектов.
**Минус:** не восстанавливает детали, которые были потеряны при max-pooling.

### U-Net (Ronneberger et al., 2015) — канонический декодер

U-Net предложил **зеркальную** структуру:

- Декодер имеет столько же уровней, сколько энкодер.
- На каждом уровне есть **skip-connection**, передающая признаки того же разрешения из энкодера.
- Признаки **конкатенируются** (а не складываются), что даёт декодеру и семантику, и детали.

Это стало стандартом для медицинской и астрофизической сегментации. По данным обзора 2025 года, U-Net и его варианты остаются наиболее широко используемой архитектурой в биомедицинской и научной сегментации.

### U-Net++ (Zhou et al., 2018) — вложенные связи

**U-Net++** предложил плотные вложенные skip-connections между всеми уровнями энкодера и декодера, что улучшает качество границ и упрощает обучение.

### DeepLabv3+ (Chen et al., 2018) — атрус-декодер

**DeepLabv3+** использует декодер с atrous (dilated) свёртками и модулем ASPP, позволяющий увеличивать receptive field без потери разрешения. Это особенно важно для крупных объектов.

### FPN (Lin et al., 2017) — пирамидальный декодер

**Feature Pyramid Network** формализовал идею **пирамиды признаков**: декодер строит иерархию карт разного разрешения, где каждая карта получается комбинацией апсемплированной карты сверху и признаков энкодера того же уровня.

Хотя FPN был предложен для детекции, идея пирамиды широко используется в сегментации (UPerNet, PSPNet-подобные головы).

### Трансформерные декодеры

- **SETR** (Zheng et al., 2021) — первый чистый трансформерный декодер для сегментации.
- **MaskFormer** (Cheng et al., 2021) — перенёс идею из DETR в сегментацию.
- **Mask2Former** (Cheng et al., 2022) — улучшенная версия с маскированным вниманием.
- **OneFormer** (Jain et al., 2023) — единый декодер для semantic/instance/panoptic сегментации.

### Современные тенденции (2024–2025)

Появились работы по **self-regularized U-Net** (Zhu et al., 2024), **мультимасштабным трансформерным декодерам** (2024) и **enhanced encoder-decoder для уменьшения потерь информации** (2024). Эти работы подчёркивают, что декодер — активная область исследований, а не устоявшийся компонент.

---

## Математика декодера

### Общая схема

Типичный декодер сегментации на $L$ уровнях можно записать рекурсивно:

$$
D_L = z_L \quad \text{(признаки bottleneck)}
$$

$$
D_{l} = \text{Conv}_{l}\left(\text{Concat}\left[\text{Up}(D_{l+1}),\; E_l\right]\right), \quad l = L-1, \ldots, 1
$$

$$
\hat{y} = \text{Head}(D_1)
$$

где:
- $E_l$ — признаки уровня $l$ энкодера (skip);
- $\text{Up}(\cdot)$ — операция апсемплинга;
- $\text{Concat}[\cdot]$ — конкатенация по каналам (U-Net) или сложение (ResNet-стиль);
- $\text{Conv}_l$ — свёрточный блок (обычно 2×(3×3 conv + norm + act));
- $\text{Head}$ — финальная 1×1 свёртка в число классов.

### Параметры, FLOPs и память декодера

Для уровня $l$ декодера с $C_{in}$ входами (апсемплированные признаки + skip) и $C_{out}$ выходами:

$$
\text{Параметры}_l = 2 \cdot (C_{in} \cdot C_{out} \cdot 3^2 + C_{out}) \quad \text{(два блока 3×3 conv + norm)}
$$

$$
\text{FLOPs}_l \approx 2 \cdot C_{in} \cdot C_{out} \cdot 9 \cdot H_l \cdot W_l
$$

> [!info] Декодер обычно тяжелее энкодера
> Из-за большего числа каналов (удвоение при конкатенации) декодер U-Net часто имеет больше параметров, чем энкодер. При подключении предобученного ResNet в качестве энкодера декодер становится основной обучаемой частью.

---

## Апсемплинг: способы повышения разрешения

Апсемплинг — обратная операция даунсемплингу. Существуют несколько способов.

### 1. Transposed Convolution (Deconvolution)

**Transposed convolution** — «обратная свёртка», которая повышает разрешение. Математически:

$$
y = W^{\top} \ast x
$$

где $\ast$ обозначает полную свёртку (full convolution) с stride 2 и padding, подобранным для увеличения размера.

> [!warning] Проблема шахматных артефактов (checkerboard artifacts)
> Работа Одей и соавт. (2016) показала, что transposed convolution со stride > 1 создаёт неравномерное «перекрытие» вкладов, что приводит к характерным шахматным артефактам в масках. Это особенно заметно при тонкой настройке и высоких скоростях обучения.
>
> **Решение:** использовать nearest/bilinear upsample + обычная свёртка вместо transposed conv.

### 2. Nearest/Bilinear Upsample + Conv

Двухэтапный подход:
1. **Upsample** — увеличивает разрешение методом ближайшего соседа или билинейной интерполяцией (без обучаемых параметров).
2. **Conv** — свёртка, уточняющая признаки.

**Преимущества:**
- Нет шахматных артефактов.
- Меньше параметров, чем у transposed conv.
- Работает стабильнее на практике.

**Это современный стандарт в U-Net, DeepLabv3+ и большинстве других декодеров.**

### 3. Pixel Shuffle (Sub-pixel Convolution)

**PixelShuffle** (Shi et al., 2016) — метод повышения разрешения, при котором свёртка создаёт $r^2$ каналов (где $r$ — коэффициент повышения), а затем каналы перегруппировываются в пространственные координаты:

$$
\text{PixelShuffle}(X)_{i,j,c} = X_{\lfloor i/r \rfloor, \lfloor j/r \rfloor, c \cdot r^2 + (i \bmod r) \cdot r + (j \bmod r)}
$$

**Плюсы:** нет артефактов, быстрое выполнение на GPU.
**Применение:** часто используется в задачах суперразрешения, реже в сегментации.

### 4. Адаптивный интерполятор (CARAFE)

**CARAFE** (Wang et al., 2019) — Content-Aware ReAssembly of FEatures: учится собирать признаки с учётом контента, что улучшает качество границ. Используется в современных детекторах и сегментаторах.

### Сравнение способов апсемплинга

| Метод | Параметры | Артефакты | Скорость | Качество |
|---|---|---|---|---|
| **Transposed conv** | Да | Есть (checkerboard) | Средняя | Средняя |
| **Upsample + Conv** | Да (только conv) | Нет | Высокая | **Хорошее** |
| **PixelShuffle** | Да | Нет | Высокая | Хорошее |
| **CARAFE** | Да (контентные) | Нет | Средняя | Отличное |

> [!success] Рекомендация
> Для сегментации солнечных пятен используйте **nearest/bilinear upsample + 3×3 conv** — это современный стандарт, свободный от артефактов и хорошо работающий на слабом GPU.

### Nearest vs Bilinear upsampling

**Nearest (ближайший сосед):**
- Сохраняет чёткие границы.
- Может давать блочные артефакты при высоком коэффициенте.
- Быстрее, не требует интерполяции.

**Bilinear (билинейная интерполяция):**
- Более гладкий результат.
- Может размывать границы.
- Чуть медленнее, но разница на GPU незначительна.

> [!tip] Для солнечных пятен
> Bilinear upsampling обычно предпочтительнее: границы пятен и так нечёткие (полутень), а bilinear даёт более гладкие признаки для последующей обработки свёрткой.

---

## Skip-connections: конкатенация vs сложение

### Конкатенация (U-Net стиль)

$$
D_l = \text{Conv}\left(\text{Concat}\left[\text{Up}(D_{l+1}), E_l\right]\right)
$$

Число каналов удваивается: $C_l^{D} = C_{l+1}^{D} + C_l^E$.

**Плюсы:** декодер получает полную информацию обоих источников.
**Минусы:** больше параметров и вычислений.

### Сложение (ResNet / FPN стиль)

$$
D_l = \text{Conv}\left(\text{Up}(D_{l+1}) + E_l\right)
$$

Число каналов сохраняется.

**Плюсы:** меньше параметров, эффективнее.
**Минусы:** требует совпадения числа каналов и разрешения.

### Attention Gates (Attention U-Net)

**Attention U-Net** (Oktay et al., 2018) добавляет **attention gates** на skip-connections:

$$
g_l = \sigma\left(\text{Conv}\left(\text{Up}(D_{l+1})\right) + \text{Conv}(E_l)\right)
$$

$$
D_l = \text{Conv}\left(\text{Concat}\left[\text{Up}(D_{l+1}), g_l \odot E_l\right]\right)
$$

где $\odot$ — поэлементное умножение. Attention gate учится «включать» только те области skip-признаков, которые релевантны задаче.

**Преимущества:**
- Улучшает фокус на целевых объектах.
- Подавляет шум из несущественных областей skip-connections.
- Полезно при сильном дисбалансе классов.

> [!tip] Для солнечных пятен
> Attention gates могут быть полезны: они фокусируют декодер на пятнах, игнорируя несущественные области (например, limb darkening).

---

## Bottleneck: роль самого узкого места

**Bottleneck** — самый глубокий уровень энкодера с минимальным пространственным разрешением и максимальным числом каналов. Именно его признаки декодер получает первым.

Формально bottleneck $z = E_L(x)$, где $L$ — число уровней энкодера.

### Роль bottleneck

- **Максимальная семантика:** самый большой receptive field.
- **Минимальная пространственная информация:** детали уже потеряны.
- **Главный регуляризатор:** узкое горлышко не позволяет сети переобучаться (см. [[Энкодер (Encoder)#Information bottleneck: механика сжатия информации|information bottleneck]]).

### Обработка bottleneck

В U-Net bottleneck — это просто два свёрточных блока. В более сложных декодерах:

- **ASPP** (DeepLabv3+): параллельные atrous свёртки с разными dilation rates, дающие мультимасштабные признаки.
- **PPM** (PSPNet, Zhao et al., 2017): pyramid pooling module с адаптивным пулингом на разных уровнях.
- **Transformer block** (SegFormer): несколько слоёв self-attention для глобального контекста.

---

## Головка декодера (Head)

На выходе декодера — финальная 1×1 свёртка, проецирующая признаки в пространство масок:

$$
\hat{y}_{i,j,c} = \sum_{k} W_{c,k} \cdot D_{1,i,j,k} + b_c
$$

Для бинарной сегментации: $C = 1$, применяется sigmoid.
Для многоклассовой: $C$ каналов, применяется softmax по классам.

### Deep supervision

**Deep supervision** (Lee et al., 2015) добавляет **вспомогательные выходы** на промежуточных уровнях декодера:

$$
\mathcal{L}_{\text{total}} = \sum_{l=1}^{L} \lambda_l \cdot \mathcal{L}(D_l, y_{\text{downsampled}})
$$

Это улучшает сходимость на глубоких сетях и работает как регуляризатор.

> [!note] Когда использовать
> Deep supervision полезен для:
> - очень глубоких декодеров (5+ уровней);
> - ускорения сходимости на ранних этапах;
> - регуляризации при малом датасете.
>
> Для стандартного 4-уровневого U-Net обычно не нужен.

---

## Типы современных декодеров

### U-Net-подобные (зеркальные)

| Архитектура | Особенности |
|---|---|
| **U-Net** | Конкатенация skip, симметричный декодер |
| **U-Net++** | Вложенные skip-connections |
| **Attention U-Net** | Attention gates на skip |
| **U-Net 3+** | Полномасштабные skip-connections со всех уровней на все |
| **Self-Regularized UNet** (Zhu, 2024) | Саморегуляризация для медицинской сегментации |
| **Enhanced Encoder-Decoder** (2024) | Улучшенная передача информации через skip |

### Пирамидальные

| Архитектура | Особенности |
|---|---|
| **FPN** | Пирамида признаков для детекции/сегментации |
| **UPerNet** | Унифицированный декодер с пирамидой и PPM |
| **PSPNet** | Pyramid Pooling Module на bottleneck |

### С атрус-свёртками

| Архитектура | Особенности |
|---|---|
| **DeepLabv3+** | ASPP + лёгкий декодер |
| **DeepLabv3** | Только ASPP без декодера |

### Трансформерные

| Архитектура | Особенности |
|---|---|
| **SETR** | Naive/Progressive upsampling |
| **MaskFormer** | DETR-подобный декодер с query tokens |
| **Mask2Former** | Masked attention |
| **OneFormer** | Единый декодер для всех типов сегментации |
| **SegFormer** | Лёгкий MLP-декодер |
| **Multi-scale Transformer Decoder** (2024) | Мультимасштабное внимание |

### Лёгкие (для мобильных устройств)

| Архитектура | Особенности |
|---|---|
| **BiSeNet** | Spatial + context path |
| **Fast-SCNN** | Shared bottleneck, быстрый декодер |
| **LightSeg** | Лёгкие модули вместо тяжёлых блоков |

---

## Современные трансформерные и универсальные декодеры

### MaskFormer и Mask2Former

**MaskFormer** (Cheng et al., 2021) предложил переосмыслить сегментацию как задачу **масочной классификации**:

$$
\hat{y} = \{(m_k, c_k)\}_{k=1}^{N}
$$

где $m_k$ — бинарная маска, $c_k$ — её класс. Декодер — это трансформер с $N$ query-токенами, каждый из которых «отвечает» за один объект.

**Mask2Former** (Cheng et al., 2022) добавил **masked attention** — внимание, ограниченное областью маски предыдущего шага, что ускоряет сходимость и улучшает качество.

### OneFormer: один декодер для всех типов сегментации

**OneFormer** (Jain et al., 2023) — единый декодер, работающий одновременно для semantic, instance и panoptic сегментации. Использует **task-guided queries** — запросы, направляемые текстовым промптом («semantic», «instance», «panoptic»).

> [!note] Значимость
> OneFormer показал, что единая архитектура может решать все три типа сегментации без специализации. Это открывает путь к универсальным декодерам.

### SEEM: Segment Everything Everywhere All at Once

**SEEM** (Zou et al., 2023) — расширение идеи promptable сегментации: декодер умеет работать с разными типами промптов (точки, боксы, текст, другие маски) и сегментировать «всё сразу».

---

## Promptable декодеры (SAM и SAM 2)

**Segment Anything Model (SAM)** (Kirillov et al., 2023) — это foundation-модель для сегментации, состоящая из трёх компонентов:
1. Image encoder (ViT-Huge).
2. Prompt encoder (точки, боксы, маски).
3. **Mask decoder** — лёгкий декодер на двух трансформерных слоях.

### Архитектура mask decoder SAM

Mask decoder в SAM:
- Получает image embeddings и prompt embeddings.
- Использует два слоя трансформера с cross-attention.
- Выдаёт $N$ предсказанных масок + токен уверенности (IoU prediction).
- Очень быстрый: ~30 мс на изображение.

**SAM 2** (Ravi et al., 2024) расширил идею на видео: декодер теперь работает с memory tokens, позволяя сегментировать объекты через временные кадры.

### Что это значит для сегментации солнечных пятен

> [!tip] Возможности
> - SAM можно использовать для **автоматической полуразметки** солнечных изображений.
> - Mask decoder SAM — пример того, каким может быть лёгкий и эффективный декодер.
> - Promptable подход позволяет сегментировать пятна интерактивно, задавая промпты на границах.
>
> Однако прямое применение SAM к солнечным изображениям требует адаптации: модель обучена на обычных фотографиях, а не на астрономических данных.

---

## Выбор декодера под ограничения железа

### Сравнение декодеров

| Декодер | Параметры (доп.) | Качество | Память | Пригодность для 4 ГБ |
|---|---|---|---|---|
| **U-Net (простой)** | ~15M | Хорошее | Средняя | Отлично |
| **U-Net++** | ~20M | Лучше | Выше | Хорошо |
| **Attention U-Net** | ~20M | Хорошо | Выше | Хорошо |
| **DeepLabv3+** | ~10M | Отличное | Средняя | Хорошо |
| **SegFormer MLP** | ~0.5M | Хорошее | Низкая | Отлично |
| **MaskFormer** | ~50M | Отличное | Высокая | Плохо |
| **FPN** | ~5M | Хорошее | Средняя | Отлично |

### Рекомендации для вашей конфигурации

> [!success] Практические рекомендации
> 1. **Старт:** классический **U-Net** с ResNet-18/34 энкодером — надёжный и хорошо работающий вариант.
> 2. **Если нужны лучшие границы:** Attention U-Net с attention gates на skip-connections.
> 3. **Если нужна компактность:** SegFormer с MLP-декодером — минимум параметров, быстрая работа.
> 4. **Если нужен лучший результат:** DeepLabv3+ с ASPP — особенно хорошо для крупных групп пятен.
> 5. **Трансформерные декодеры** (MaskFormer, Mask2Former) требуют больших данных и памяти — не для первой итерации на слабом ноутбуке.
> 6. **SAM / SAM 2** — для полуавтоматической разметки датасета, не для основного обучения.

### Апсемплинг: выбор

> [!success] Рекомендация
> Используйте **bilinear upsample + 3×3 conv** для всех уровней декодера. Это:
> - свободен от checkerboard-артефактов;
> - стабилен при обучении;
> - эффективен по памяти и скорости;
> - стандарт в современных архитектурах.

---

## Специфика солнечных изображений

### Особенности задачи для декодера

- **Мультимасштабные объекты:** от мелких пор (несколько пикселей) до крупных групп пятен (сотни пикселей) — декодер должен обрабатывать признаки всех масштабов.
- **Чёткие границы:** для точного вычисления площадей пятен важны границы — внимание к качеству skip-connections и апсемплинга.
- **Limb darkening:** градиент яркости к краю диска — декодер должен его компенсировать, а не путать с пятнами.
- **Разреженные объекты:** пятна занимают <5% пикселей — attention gates могут быть полезны для фокусировки на пятнах.

### Что используют в работах по солнечной сегментации

- U-Net и его модификации — стандарт в работах по сегментации солнечных пятен (Mourato et al., 2024; Sayez et al., 2023).
- DeepLabv3+ применяется для крупных структур (корональные дыры, филаменты).
- Трансформерные декодеры в солнечной физике встречаются редко из-за малых датасетов, но появляются гибридные подходы (MBP-TransCNN, Yang et al., 2023).
- Для активных областей Солнца применяются CNN-декодеры с мультимасштабной обработкой (Quan et al., 2021; Zhang et al., 2024).

### Мультимасштабность для солнечных пятен

Для солнечных изображений особенно важна мультимасштабность декодера:

- **Уровень $H/4$:** видит мелкие поры, тонкие границы полутени.
- **Уровень $H/8$:** малые пятна.
- **Уровень $H/16$:** средние пятна.
- **Уровень $H/32$:** крупные группы, общий контекст.

Все эти уровни должны быть интегрированы в декодере. U-Net с 4 уровнями skip-connections — хороший баланс.

---

## Практика в PyTorch

### Подключение готовых декодеров (segmentation_models.pytorch)

```python
import segmentation_models_pytorch as smp

# U-Net с ResNet-18 энкодером
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    decoder_channels=(256, 128, 64, 32, 16),  # размеры каналов декодера
)

# DeepLabv3+ с ResNet-34 энкодером
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)

# SegFormer с MiT-B0 энкодером
model = smp.Segformer(
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
```

> [!note]+ Построчный разбор
> 1. `smp.Unet` / `smp.DeepLabV3Plus` / `smp.Segformer` — разные архитектуры с готовыми декодерами.
> 2. `decoder_channels` — позволяет контролировать ширину декодера (важно для баланса качества и памяти).
> 3. `encoder_weights="imagenet"` — загружает предобученный энкодер, декодер обучается с нуля.

### Собственный декодер (упрощённый U-Net)

```python
import torch
import torch.nn as nn

class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels,
                                out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        # Приведение к размеру skip (если нужно)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode='bilinear',
                align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        return self.conv(x)
```

> [!note]+ Построчный разбор
> 1. `nn.Upsample(scale_factor=2, mode='bilinear')` — повышает разрешение вдвое билинейной интерполяцией (без артефактов).
> 2. `torch.cat([x, skip], dim=1)` — конкатенация с skip-признаками по каналам (U-Net стиль).
> 3. `nn.GroupNorm` вместо BatchNorm — важно при малых батчах.
> 4. Два свёрточных блока 3×3 — стандартный паттерн U-Net.

### Attention Gate в декодере

```python
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.GroupNorm(8, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """g — из декодера (апсемплированный), x — skip из энкодера."""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
```

> [!note]+ Построчный разбор
> 1. `W_g` и `W_x` — проекции признаков декодера и энкодера в общее пространство.
> 2. `g1 + x1` — сложение двух проекций: позволяет выучить соответствие.
> 3. `psi` — sigmoid-активация, выдающая attention weights в диапазоне [0, 1].
> 4. `x * psi` — поэлементное умножение: «включает» только релевантные области skip-признаков.

### Deep Supervision

```python
class DecoderWithDeepSupervision(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ... блоки декодера ...
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(ch, num_classes, 1) for ch in aux_channels
        ])
        self.final_head = nn.Conv2d(final_channels, num_classes, 1)

    def forward(self, encoder_features):
        outputs = []
        x = encoder_features[-1]

        for i, (block, skip) in enumerate(
            zip(self.decoder_blocks, reversed(encoder_features[:-1]))
        ):
            x = block(x, skip)
            if self.training:  # только во время обучения
                outputs.append(self.aux_heads[i](x))

        outputs.append(self.final_head(x))
        return outputs  # последняя — основное предсказание
```

---

## Масштабирование и перспектива

### Если появится больше вычислительных ресурсов

- **Больше VRAM** → можно использовать MaskFormer, Mask2Former или OneFormer.
- **Больше данных** → можно обучать трансформерные декодеры с нуля.
- **Многокарточное обучение** → можно использовать promptable модели (SAM 2) для интерактивной разметки.

### Если появится больше данных

- Можно отказаться от классического U-Net и перейти на Mask2Former/OneFormer.
- Можно использовать SAM 2 для полуавтоматической разметки.
- Можно применять deep supervision и более сложные attention-механизмы.

### Перспективные направления

> [!note]+ На будущее
> - **Universal decoders:** OneFormer-подобные декодеры для одновременной semantic/instance/panoptic сегментации солнечных структур.
> - **Promptable segmentation:** использование SAM 2 для интерактивной разметки солнечных изображений.
> - **Foundation models для астрономии:** специализированные модели, предобученные на больших астрономических датасетах.
> - **Temporal decoders:** декодеры, работающие с последовательностями солнечных изображений для трекинга эволюции пятен.

---

## Частые ошибки

> [!danger]- 1. Transposed convolution вместо upsample + conv
> Checkerboard-артефакты в масках, особенно при высоких скоростях обучения.
> **Решение:** использовать bilinear upsample + 3×3 conv.

> [!danger]- 2. Слишком тяжёлый декодер на малом датасете
> Переобучение: декодер запоминает детали разметки.
> **Решение:** использовать более узкие `decoder_channels` или MLP-декодер.

> [!danger]- 3. Несовпадение размеров skip-признаков
> Ошибка при конкатенации: размеры не совпадают.
> **Решение:** использовать `F.interpolate` для точного приведения к размеру skip.

> [!danger]- 4. Отсутствие нормализации в декодере
> Нестабильное обучение, особенно при малых батчах.
> **Решение:** использовать GroupNorm в декодере.

> [!danger]- 5. Сложение вместо конкатенации без проекции
> Если числа каналов не совпадают, сложение невозможно; если совпадают — теряется информация.
> **Решение:** использовать конкатенацию (U-Net стиль) или проекцию перед сложением.

> [!danger]- 6. Слишком много skip-connections
> Избыточность параметров, риск переобучения на малом датасете.
> **Решение:** использовать 4 уровня skip-connections (стандарт U-Net).

> [!danger]- 7. Игнорирование bottleneck-обработки
> Простой bottleneck теряет мультимасштабную информацию.
> **Решение:** добавить ASPP или pyramid pooling в bottleneck.

---

## Литература и источники

### Основы декодеров

1. **Long, Shelhamer, Darrell (2015).** *Fully Convolutional Networks for Semantic Segmentation.* [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
   Первая работа по декодерам для сегментации.

2. **Ronneberger, Fischer, Brox (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
   Канонический U-Net с зеркальным декодером.

3. **Badrinarayanan, Kendall, Cipolla (2015).** *SegNet: A Deep Convolutional Encoder-Decoder Architecture.* [arXiv:1511.00561](https://arxiv.org/abs/1511.00561)
   Декодер с передачей индексов max-pooling.

### Улучшенные U-Net декодеры

4. **Zhou et al. (2018).** *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* [arXiv:1807.10165](https://arxiv.org/abs/1807.10165)
   Вложенные skip-connections.

5. **Oktay et al. (2018).** *Attention U-Net: Learning Where to Look for the Pancreas.* [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)
   Attention gates на skip-connections.

6. **Huang et al. (2020).** *UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.* [arXiv:2004.08790](https://arxiv.org/abs/2004.08790)
   Полномасштабные skip-connections.

7. **Jiangtao (2025).** *A Comprehensive Review of U-Net and Its Variants.* [arXiv:2502.06895](https://arxiv.org/abs/2502.06895)
   Свежий обзор всех вариантов U-Net.

8. **Zhu et al. (2024).** *Self-Regularized UNet for Medical Image Segmentation.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12408486/)
   Саморегуляризация в декодере.

9. **Enhanced Encoder-Decoder Network for Reducing Information Loss (2024).** [arXiv:2406.01605](https://arxiv.org/abs/2406.01605)
   Улучшенная передача информации через skip-connections.

### Пирамидальные декодеры

10. **Lin et al. (2017).** *Feature Pyramid Networks for Object Detection.* [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
    Пирамида признаков.

11. **Zhao et al. (2017).** *Pyramid Scene Parsing Network (PSPNet).* [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
    Pyramid Pooling Module.

12. **Xiao et al. (2018).** *Unified Perceptual Parsing for Scene Understanding (UPerNet).* [arXiv:1807.10221](https://arxiv.org/abs/1807.10221)
    Унифицированный декодер.

### Атруc-декодеры

13. **Chen et al. (2018).** *Encoder-Decoder with Atrous Separable Convolution (DeepLabv3+).* [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
    ASPP + лёгкий декодер.

### Трансформерные и универсальные декодеры

14. **Zheng et al. (2021).** *Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (SETR).* [arXiv:2012.15840](https://arxiv.org/abs/2012.15840)

15. **Cheng, Schwing, Kirillov (2021).** *Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer).* [arXiv:2107.06278](https://arxiv.org/abs/2107.06278)

16. **Cheng et al. (2022).** *Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former).* [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)

17. **Jain et al. (2023).** *OneFormer: One Transformer to Understand Universal Image Segmentation.* [arXiv:2211.06220](https://arxiv.org/abs/2211.06220)

18. **Zou et al. (2023).** *Segment Everything Everywhere All at Once (SEEM).* [arXiv:2304.06718](https://arxiv.org/abs/2304.06718)

19. **Multi-scale Transformer-based Decoder for Semantic Segmentation (2024).** [arXiv:2211.13928](https://arxiv.org/abs/2211.13928)

### Promptable декодеры

20. **Kirillov et al. (2023).** *Segment Anything (SAM).* [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
    Foundation-модель для сегментации с лёгким mask decoder.

21. **Ravi et al. (2024).** *SAM 2: Segment Anything in Images and Videos.* [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
    Расширение SAM на видео с memory tokens.

### Лёгкие декодеры

22. **Xie et al. (2021).** *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
    Лёгкий MLP-декодер.

23. **Yu et al. (2018).** *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation.* [arXiv:1808.00897](https://arxiv.org/abs/1808.00897)

24. **Poudel et al. (2019).** *Fast-SCNN: Fast Semantic Segmentation Network.* [arXiv:1902.04502](https://arxiv.org/abs/1902.04502)

### Апсемплинг и артефакты

25. **Odena, Dumoulin, Olah (2016).** *Deconvolution and Checkerboard Artifacts.* [Distill Pub](https://distill.pub/2016/deconv-checkerboard/)
    Анализ checkerboard-артефактов transposed conv.

26. **Shi et al. (2016).** *Real-Time Single Image and Video Super-Resolution (PixelShuffle).* [arXiv:1609.05158](https://arxiv.org/abs/1609.05158)

27. **Wang et al. (2019).** *CARAFE: Content-Aware ReAssembly of FEatures.* [arXiv:1905.02188](https://arxiv.org/abs/1905.02188)

### Deep Supervision

28. **Lee et al. (2015).** *Deeply-Supervised Nets.* [arXiv:1409.5185](https://arxiv.org/abs/1409.5185)

### Обзоры

29. **Transformer-Based Visual Segmentation: A Survey.** [arXiv:2304.09854](https://arxiv.org/abs/2304.09854)
    Обзор трансформерных подходов к сегментации (PAMI 2024).

30. **Rafi et al. (2024).** *Domain generalization for semantic segmentation: a survey.* [Springer](https://link.springer.com/article/10.1007/s10462-024-10817-z)

31. **A Survey on Deep Learning-based Architectures for Semantic Segmentation.** [arXiv:1912.10230](https://arxiv.org/abs/1912.10230)

### Солнечная физика

32. **Mourato et al. (2024).** *Automatic sunspot detection through semantic and instance segmentation.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197623018201)

33. **Sayez et al. (2023).** *SunSCC: Segmenting, Grouping and Classifying Sunspots.* [AGU](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JA031548)

34. **Quan et al. (2021).** *Solar Active Region Detection Using Deep Learning.* [MDPI Electronics](https://www.mdpi.com/2079-9292/10/18/2284)

35. **Zhang et al. (2024).** *Statistical Analyses of Solar Prominences and Active Region Filaments.* [ApJS](https://iopscience.iop.org/article/10.3847/1538-4365/ad3039)

36. **Yang et al. (2023).** *MBP-TransCNN: Automated segmentation of magnetic bright points.* [A&A](https://www.aanda.org/articles/aa/full_html/2023/09/aa46914-23/aa46914-23.html)

---

## Вывод

> [!success] Итого
> **Декодер** — восстанавливающая часть архитектуры, преобразующая bottleneck-признаки в маску сегментации через серию апсемплингов.
>
> Ключевые принципы:
>
> - **Апсемплинг:** bilinear + 3×3 conv — современный стандарт, без checkerboard-артефактов.
> - **Skip-connections:** конкатенация (U-Net стиль) или сложение (ResNet/FPN стиль).
> - **Attention gates** — улучшают фокус на целевых объектах, особенно полезны при дисбалансе.
> - **Bottleneck** можно усилить через ASPP, PPM или трансформерные блоки.
> - **Типы декодеров:**
>   - U-Net-подобные (классика, надёжно);
>   - Пирамидальные (FPN, UPerNet);
>   - С атрус-свёртками (DeepLabv3+);
>   - Трансформерные (MaskFormer, OneFormer);
>   - Promptable (SAM, SAM 2);
>   - Лёгкие (BiSeNet, SegFormer MLP).
> - **Память:** декодер часто тяжелее энкодера; контроль `decoder_channels` важен для слабого GPU.
> - **Современный тренд:** универсальные декодеры (OneFormer) и promptable декодеры (SAM 2).
>
> Для задачи сегментации солнечных пятен:
>
> ```text
> Декодер: U-Net с 4 уровнями skip-connections
> Апсемплинг: bilinear upsample + 3×3 conv (на каждом уровне)
> Skip-connections: конкатенация (U-Net стиль)
> Нормализация декодера: GroupNorm (при батчах 2–4)
> Опционально: Attention gates для борьбы с дисбалансом
> Головка: 1×1 conv → sigmoid (бинарная) или softmax (многоклассовая)
> Энкодер: ResNet-18 или EfficientNet-B0 (предобучение ImageNet)
> ```

---

## Связанные термины

- [[Энкодер (Encoder)|Энкодер]] — парный компонент, выдающий признаки для декодера
- [[Сегментация (Segmentation)|Сегментация]] — задача, которую решает декодер
- [[Функция потерь (Loss Function)|Функция потерь]] — определяет, как обучать декодер
- [[Нормализация (Normalization)|Нормализация]] — GroupNorm в декодере при малых батчах
- [[Маска (Mask)|Маска]] — финальный выход декодера
- [[Аугментация (Data Augmentation)|Аугментация]] — увеличивает разнообразие признаков для декодера
- [[Оптимизаторы (Optimizers)|Оптимизаторы]] — оптимизация параметров декодера
- [[Дамп (Dump)|Дамп]] — дамп активаций декодера для диагностики
- [[Батч (Batch)|Батч]] — размер батча влияет на потребление памяти декодера
- [[Унитарное кодирование (One-hot)|Унитарное кодирование]] — для многоклассовых выходов декодера