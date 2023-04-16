from telebot import TeleBot
#from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from PIL import Image

from diffusers import StableDiffusionPipeline
import torch
#from super_image import EdsrModel, ImageLoader

import random
import string

import json

from multiprocessing.dummy import Pool

import os

token = "ВАШ:ТОКЕН_БОТА"
videocard_name = "Название вашой видеокарты. Например RTX 3060Ti"

bot = TeleBot(token)
pool = Pool(1)

text2image_model_id = "prompthero/openjourney"
#upscale_model_id = "eugenesiow/edsr-base"

taskRunningGen = 0

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Параметры torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def gen_random_str(str_len):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=str_len))

# def callback_task(step, timestep, latents):
#     #print(f'step: {step} | timestep: {timestep}')
#     if step % 5 == 0:
#         bot.edit_message_text(chat_id = currentChatID, message_id = currentMessageID, text = f'Итерация: {step}/100')
#         #pool.apply_async(bot.edit_message_text, [], {'chat_id': currentChatID, 'message_id': currentMessageID, 'text': f'Итерация: {step}/100'})
#     #bot.edit_message_text(chat_id = currentChatID, message_id = currentMessageID, text = f'Итерация: {step}/100')

print(f'[StableDiffusionPipeline] Инициализируем модель...')
print(f'[StableDiffusionPipeline] model_id: {text2image_model_id}')
pipe = StableDiffusionPipeline.from_pretrained(
    text2image_model_id,
    torch_dtype = torch.float16,
    #subfolder = f"./models/{model_id.replace('/', '_')}",
    cache_dir = f"./models_cache/{text2image_model_id.replace('/', '_')}"
)
print(f'[StableDiffusionPipeline] Инициализируем NVIDIA CUDA')
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
print(f'[StableDiffusionPipeline] Инициализация прошла')
print(f'[StableDiffusionPipeline] Генерируем изображение для теста')
pipe("nature")
print(f'[StableDiffusionPipeline] Сгенерировано')
# print(f'[EdsrModel] Инициализируем модель для повышения разрешения')
# upscale_model = EdsrModel.from_pretrained(upscale_model_id, scale=2)
# print(f'[EdsrModel] Инициализация прошла')

def launch_gen_task(
        chat_id,
        user_id,
        content: str,
        negative_prompt: str,
        image_id: int,
        seed: int,
        height: int = 512,
        width: int = 512
    ):
    global taskRunningGen

    taskRunningGen = 1

    content = "mdjrny-v4 style " + content

    print('Запускаем таск...')

    prompt = [content]
    negative_prompt = [negative_prompt]
    #generator = [torch.Generator(device="cuda").manual_seed(seed)]

    # if (height < 512 + 1) and (width < 512 + 1):
    #     num_inference_steps = 250
    # else:
    #     num_inference_steps = 160

    print('Начинаем генерацию...')
    openedImages = []
    for x in range(1, 4 + 1):
        images = pipe(
            prompt,
            negative_prompt = negative_prompt,
            #generator = generator,
            num_inference_steps = 100,
            height = height,
            width = width,
            #callback = callback_task
        ).images
        images[0].save(f"./outputs/{user_id}_{image_id}_{x}.png")
        # inputs = ImageLoader.load_image(
        #     Image.open(f"./outputs/{user_id}_{image_id}_{x}.png")
        # )
        # preds = upscale_model(inputs)
        # ImageLoader.save_image(preds, f"./outputs/{user_id}_{image_id}_{x}.png")
        openedImages.append(
            Image.open(f"./outputs/{user_id}_{image_id}_{x}.png")
        )
    
    grid = image_grid(openedImages, rows=2, cols=2)
    grid.save(f'./outputs_grid/{user_id}_{image_id}.png')
    print("Изображение готово")

    # bot.send_message(chat_id, "Изображение готово! Мы повышаем разрешение в 2 раза, пожалуйста подождите...")

    # image = Image.open(open(f'./outputs/{image_id}.png', 'rb'))

    # inputs = ImageLoader.load_image(image)
    # preds = model(inputs)
    # ImageLoader.save_image(preds, f'./outputs/{image_id}_U2.png')

    taskRunningGen = 0

    bot.send_chat_action(chat_id, 'upload_photo')

    img = open(f'./outputs_grid/{user_id}_{image_id}.png', 'rb')
    bot.send_photo(chat_id, img)
    #bot.send_message(chat_id, "Параметры", reply_markup=buttons_gen(image_id, content, negative_prompt, seed, height, width, chat_id))

@bot.message_handler(commands=['login'])
def login_message(message):
    user_id = message.from_user.id
    return bot.send_message(message.chat.id,
        f'https://openjourney.loca.lt/login/{user_id}'
    )

@bot.message_handler(commands=['source_image'])
def source_image(message):
    content = message.text
    contentx = content.split()
    user_id = message.from_user.id
    try:
        image_id = contentx[1]
        image_var = contentx[2]
    except:
        return bot.send_message(message.chat.id, f'Укажите id изображения, и его вариант. Например: /source_image X8m3OIoXRh 1')

    try:
        img = open(f'./outputs/{user_id}_{image_id}_{image_var}.png', 'rb')
    except:
        return bot.send_message(message.chat.id, f'Что-то пошло не так во время получения изображения')
    return bot.send_photo(message.chat.id, img)

@bot.message_handler(commands=['delete_image'])
def delete_image(message):
    content = message.text
    contentx = content.split()
    user_id = message.from_user.id
    try:
        image_id = contentx[1]
        image_var = contentx[2]
    except:
        return bot.reply_to(message, f'Укажите id изображения и его вариант. Например: /delete_image X8m3OIoXRh 3')

    try:
        os.remove(f'./outputs/{user_id}_{image_id}_{image_var}.png')
    except:
        return bot.reply_to(message, f'Что-то пошло не так во время удаления изображения')
    return bot.reply_to(message, f'Изображение удалено')

@bot.message_handler(commands=['start'])
def start_message(message):
    return bot.reply_to(message,
        f'''Привет!
        
        Я бот от DragonFire Community, который может превращать текст в изображения

        Основано на модели {text2image_model_id}
        Рендерится на {videocard_name}
        
        Чтобы сгенерировать изображение, просто напиши запрос, который ты хочешь сгенерировать

        Для профессионалов, используйте JSON
        
        ВНИМАНИЕ: Бот в бета тестирований, всё может работать нестабильно'''
    )

def join_query_gen(
    chat_id,
    user_id,
    content: str,
    negative_prompt: str,
    image_id: int,
    seed: int,
    height: int = 512,
    width: int = 512
):
    global taskRunningGen
    bot.send_message(chat_id, f'Ожидание запуска...')
    while True:
        if taskRunningGen == 1: pass
        else:
            taskRunningGen = 1
            bot.send_message(chat_id, f'Генерируем... Займёт 1 минуту, может меньше...\nID изображения: {image_id}\nСид: {seed}')
            try:
                launch_gen_task(chat_id, user_id, content, negative_prompt, image_id, seed, height, width)
                os.remove(f'./outputs_grid/{user_id}_{image_id}.png')
                taskRunningGen = 0
                break
            except Exception as e:
                print(e)
                if str(e).startswith('CUDA out of memory.'):
                    taskRunningGen = 0
                    bot.send_message(chat_id, f'У нас наблюдаются проблемы с памятью на видеокарте. Изображение не сгенерируется')
                    #bot.send_message(chat_id, f'У нас наблюдаются проблемы с памятью на видеокарте. Изображение не сгенерируется')
                    break
                else:
                    launch_gen_task(chat_id, user_id, content, "", image_id, seed)
                    taskRunningGen = 0
                    break

@bot.message_handler(content_types=['text'])
def message_handler(message):
    content = message.text
    global taskRunningGen

    user_id = message.from_user.id

    # if taskRunning == 1:
    #     return bot.send_message(message.chat.id, f'Задача уже выполняется!')
    # else:
    #     pass

    if content.startswith('/'):
        return bot.send_message(message.chat.id, f'Команда не найдена')
    else:
        image_id = gen_random_str(10)
        bot.send_message(message.chat.id, f'Ожидайте, вы в очереди')
        try:
            json_content = json.loads(content)

            prompt = str(json_content['prompt'])

            try: seed = int(json_content['seed'])
            except: seed = random.randint(1, 2147483647)

            try: negative_prompt = str(json_content['no'])
            except: negative_prompt = ""

            try: height, width = int(json_content['height']), int(json_content['width'])
            except: height, width = 512, 512

            #return join_query()

            #bot.send_message(message.chat.id, f'Генерируем... Займёт 1 минуту, может меньше...\nID изображения: {image_id}\nСид: {seed}\nШирина/Высота: {width}x{height}')
            #taskRunning = 1
            #return launch_gen_task(message.chat.id, content, negative_prompt, image_id, seed, height, width)
            pool.apply_async(join_query_gen, args=[message.chat.id, user_id, prompt, negative_prompt, image_id, seed, height, width])
        except Exception as e:
            print(e)
            if str(e).startswith('CUDA out of memory.'):
                return bot.send_message(message.chat.id, f'У нас наблюдаются проблемы с памятью на видеокарте. Изображение не сгенерируется')
            else:
                seed = random.randint(1, 2147483647)
                #print(f'image_id: {image_id}')
                #bot.send_message(message.chat.id, f'Генерируем... Займёт 1 минуту, может меньше...\nID изображения: {image_id}\nСид: {seed}')
                #taskRunning = 1
                #return launch_gen_task(message.chat.id, content, "", image_id, seed)
                pool.apply_async(join_query_gen, args=[message.chat.id, user_id, content, "", image_id, seed])

print('[telebot] Бот запущен')
bot.infinity_polling()
