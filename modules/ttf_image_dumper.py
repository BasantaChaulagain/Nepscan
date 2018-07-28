from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

from values import *
from config import TTF_FONT_DIRECTORY, TEST_DATA_DIRECTORY, TRAIN_DATA_DIRECTORY


def main():
    count = -1
    test_directories = []
    train_directories = []
    for dir in directoryNames:
        TEST_DIRECTORY = TEST_DATA_DIRECTORY + dir + "/"
        test_directories.append(TEST_DIRECTORY)
        TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + dir + "/"
        train_directories.append(TRAIN_DIRECTORY)

        if not os.path.exists(TEST_DIRECTORY):
            os.makedirs(TEST_DIRECTORY)
        if not os.path.exists(TRAIN_DIRECTORY):
            os.makedirs(TRAIN_DIRECTORY)

        count = count + 1
        for key, value in fontFiles.items():
            print key
            custom_font = ImageFont.truetype(TTF_FONT_DIRECTORY + key, font_size)

            #draw image for each characters of each font
            if value[count] != '':
                im = Image.new("RGB", (width, height), back_ground_color)
                draw = ImageDraw.Draw(im)
                draw.text((10, 10), value[count], font=custom_font, fill=font_color)
                im.save(test_directories[count]+key.lower().strip('.ttf')+".png", 'PNG')
                im.save(train_directories[count]+key.lower().strip('.ttf')+".png", 'PNG')


if __name__== "__main__":
    main()