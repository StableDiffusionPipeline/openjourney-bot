# openjourney-bot
Telegram бот OpenJourney

# Что нужно для запуска
Видеокарта которая поддерживает NVIDIA CUDA (У меня Gigabyte RTX 3060Ti)

Не менее 4 ГБ видеопамяти

Запускать на GTX 1660 и ниже не рекомендуется

# Как запустить
Скачиваем [Python](https://www.python.org/downloads/)

Идём на сайт NVIDIA Developer скачивать CUDA: https://developer.nvidia.com/cuda-downloads

Далее идём на сайт [PyTorch](https://pytorch.org/), скачиваем его

На сайте PyTorch мне подходят такие параметры:

![изображение](https://user-images.githubusercontent.com/64083584/232308097-15695ba6-bf3d-4553-8fcb-12a3fb0b60da.png)

Бёрем команду, вводим в терминал и ждём (у меня скачивание заняло 1 час)

Выбираем CUDA 11.8 обязательно

Когда всё скачали, можно поставить Python библиотеки: ``pip install pyTelegramBotAPI diffusers`` (вроде ещё нужен random ``pip install random``)

А теперь настраиваем бота (зайдите в файл ``openjourney_bot.py``, и измените значения строк 19 и 20)

Когда всё настроили, можно запускать!

``python3 openjourney_bot.py`` (или просто ``python``)

Скачивание модели будет автоматическое, ждём (у меня заняло 6 часов)

Потом заходим в своего бота в тг, вводим что мы хотим сгенерировать (только на английском), и ждём

В консоли будет что-то типо:
```
Запускаем таск...
Начинаем генерацию...
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:26<00:00,  3.75it/s]
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:22<00:00,  4.44it/s]
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:21<00:00,  4.73it/s]
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.98it/s]
Изображение готово
```

```
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:20<00:00,  4.98it/s]
```

100% - Процент готовности изображения

100/100 - первая 100 это количество пройденных итераций, вторая 100 это сколько всего итераций

00:20<00:00 - 00:20 это сколько времени прошло, 00:00 - сколько времени осталось

4.98it/s - итераций в секунду
