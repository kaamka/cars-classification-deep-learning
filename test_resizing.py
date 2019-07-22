from PIL import Image, ImageOps
import os

#im_pth = '../test_imgs/skoda2.jpg'

#/Desktop/InteliGate/CLASSIFICATION/VMMR/google_imgs/downloads
def resize_black(desired_size, im_pth, overwrite = False, print_oldsize=True):
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
    if print_oldsize:
        print(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
    if overwrite:
        new_im.save(im_pth)
    return new_im



def resize_white(im_pth, width, height, overwrite = False, print_oldsize=True):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    image_pil = Image.open(im_pth)
    if print_oldsize:
        print(image_pil.size)
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    new_im = background.convert('RGB')
    if overwrite:
        new_im.save(im_pth)
    return new_im

# if __name__ == "__main__":
#     resize('../test_imgs/forg.png', 299, 299, overwrite = True)

if __name__ == "__main__":
    if (os.getcwd() == '/home/kalkami/translearn' or os.getcwd() == '/home/kalkami/translearn_cpu'):
        #lhcpgpu1
        TRAIN_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/train'
        TEST_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/test'
    else:
        #local
        TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/train'
        TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/test'
        #folder = '../google_imgs/downloads'
    folders = [TRAIN_DIR, TEST_DIR]
    width = 299
    height = 299
    for folder in folders:
        for subfol in os.scandir(folder):
            for img in os.scandir(subfol):
                if os.path.isfile(img):
                    print(img.name)
                    resize_white(os.path.abspath(img), width, height, overwrite = True)
                    #resize_to_square(desired_size, os.path.abspath(img), overwrite=True)
                #resize_to_square(desired_size, im_pth)