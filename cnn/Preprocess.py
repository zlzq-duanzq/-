from glob import glob
import cv2
import os


def my_copy(src, dst):
    img = cv2.imread(src)
    img = cv2.resize(img, (128,128))
    cv2.imwrite(dst, img)

def partition(input_path, tr_num, te_num, va_num):
    file_path = glob(input_path)
    cur_path = os.getcwd()
    tr_path = cur_path+'/images/train'
    te_path = cur_path+'/images/test'
    va_path = cur_path+'/images/val'
    if not os.path.exists(tr_path):
        os.mkdir(tr_path)
    if not os.path.exists(te_path):
        os.mkdir(te_path)
    if not os.path.exists(va_path):
        os.mkdir(va_path)

    for i, img_path in enumerate(file_path):
        #print(i, img_path)
        if i < tr_num:
            my_copy(img_path, tr_path+'/'+str(i)+'.jpg')
        if i < tr_num+te_num and i >= tr_num:
            my_copy(img_path, te_path+'/'+str(i-tr_num)+'.jpg')
        if i >= tr_num+te_num and i < tr_num+te_num+va_num:
            my_copy(img_path, va_path+'/'+str(i-tr_num-te_num)+'.jpg')

partition('images/val2017/*.jpg', 1000, 50, 200)