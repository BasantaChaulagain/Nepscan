from PIL import Image, ImageDraw, ImageFont
import os
from values import *
from config import *

def main():
    cntr =1
    op_directory = []
    for dir in char_dirs:
        char_folder = DUMPED_IMAGES + cntr.__str__() +"_" + dir + "/"
        op_directory.append(char_folder)
        if not os.path.exists(char_folder):
            os.makedirs(char_folder)
        cntr+=1

    for key, value in fontFiles.items():
        print ("Generating Font Images for : " + key)
        print (TTF_FONT_DIRECTORY + key)
        custom_font = ImageFont.truetype(TTF_FONT_DIRECTORY + key, font_size)
        print custom_font
        #draw image for each characters of each font
        count = 0
        for ch in value:
            ch = ch.strip()
            count += 1
            if (ch==''):
                continue
            im  =  Image.new ("RGB", (width,height), back_ground_color )
            draw  =  ImageDraw.Draw ( im )
            draw.text ( (offsetx,offsety), ch, font=custom_font, fill=font_color )
            im_resized = im.resize(resized_dimen, Image.ANTIALIAS)
            im_resized.save(op_directory[count-1]+key.lower().strip(".ttf")+".png",'PNG')

if __name__== "__main__":
    main()
