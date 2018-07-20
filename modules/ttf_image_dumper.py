from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

from values import *
from config import TTF_FONT_DIRECTORY, TEST_DATA_DIRECTORY, TRAIN_DATA_DIRECTORY


def main():
    for key, value in fontFiles.items():
        custom_font = ImageFont.truetype(TTF_FONT_DIRECTORY+key, font_size)
        TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + key.lower().strip('.ttf')+"/"
        TEST_DIRECTORY = TEST_DATA_DIRECTORY + key.lower().strip('.ttf')+"/"

        if not os.path.exists(TEST_DIRECTORY):
            os.makedirs(TEST_DIRECTORY)

        if not os.path.exists(TRAIN_DIRECTORY):
            os.makedirs(TRAIN_DIRECTORY)

        #draw image for each characters of each font
        count = 0
        for ch in value:
            ch = ch.strip()
            count += 1
            if (ch==''):
                continue
            im  =  Image.new ( "RGB", (width,height), back_ground_color )
            draw  =  ImageDraw.Draw ( im )
            draw.text ( (10,10), ch, font=custom_font, fill=font_color )
            im.save(TRAIN_DIRECTORY+fileNames[count-1]+".png",'PNG')
            im.save(TEST_DIRECTORY+fileNames[count-1]+".png",'PNG')


if __name__== "__main__":
    main()