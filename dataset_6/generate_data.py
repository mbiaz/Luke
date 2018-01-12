from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from random import randint
import random
import json
import sys
import os
import shutil


def str_generator(size=6, chars="azertyuiopqsdfghjjkllmwxcvbn"):
    return ''.join(random.choice(chars) for _ in range(size))


TEST_DIR = "test/"
TRAIN_DIR = "train/"

try:
    shutil.rmtree(TEST_DIR)
except:
    pass

try:
    shutil.rmtree(TRAIN_DIR)
except:
    pass

try:
    os.makedirs("train/images")
except:
    raise

try:
    os.makedirs("test/images")
except:
    raise

array_devises = [
    "€",
    "$",
    "eur"
]
colors_array = [
    "#000000"
]
base_array = [
    "blank.jpg"
]
array_vat = [
    "5.5% : ",
    "10% : ",
    "20% : ",
    "2,1% : ",
    "5.5% : ",
    "10% : ",
    "20% : ",
    "2,1%",
    "(5.5%) : ",
    "(10%) : ",
    "(20%) : ",
    "(2,1%) :"
]
possiblechars = "0123456789.,%€$"
labels = {}
labels2 = {}
CHARS_NUMBER = 7



for i in range(0, int(sys.argv[1])):
    img = Image.open(base_array[randint(0, len(base_array)-1)])
    rand_font = "0"

    first_length = randint(1, 4)

    randomStr = str_generator(first_length, possiblechars)
    randomStr += " "
    randomStr += str_generator(CHARS_NUMBER-first_length-1, possiblechars)

    font = ImageFont.truetype("fonts/0.ttf", randint(18, 26))
    draw = ImageDraw.Draw(img)
    draw.text((randint(10,20), 15), randomStr, font=font, fill=colors_array[randint(0, len(colors_array)-1)])
    # draw.text((400, 10), randomPrice, (0, 0, 0), font=font)
    img.save("train/images/" + str(i) + '.jpg')
    labels[str(i)] = randomStr

with open('train/labels.json', 'w') as outfile:
    json.dump(labels, outfile, indent=4)

for i in range(0, int(sys.argv[2])):
    img = Image.open(base_array[randint(0, len(base_array)-1)])
    rand_font = str(randint(0, 11))

    first_length = randint(1, 4)

    randomStr = str_generator(first_length, possiblechars)
    randomStr += " "
    randomStr += str_generator(CHARS_NUMBER - first_length - 1, possiblechars)

    font = ImageFont.truetype("fonts/0.ttf", randint(14, 26))
    draw = ImageDraw.Draw(img)
    draw.text((randint(10,20), 15), randomStr, font=font, fill=colors_array[randint(0, len(colors_array)-1)])
    # draw.text((400, 10), randomPrice, (0, 0, 0), font=font)
    img.save("test/images/" + str(i) + '.jpg')
    labels2[str(i)] = randomStr

with open('test/labels.json', 'w') as outfile:
    json.dump(labels2, outfile, indent=4)
