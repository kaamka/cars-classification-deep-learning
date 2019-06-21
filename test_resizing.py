from PIL import Image, ImageOps
im_pth = '../test_imgs/skoda2.jpg'
desired_size = 299

def resize_to_square(desired_size, im_pth):
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
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
    new_im.save(im_pth)
    new_im.show()

if __name__ == "__main__":
    resize_to_square(desired_size, im_pth)