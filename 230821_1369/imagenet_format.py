import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm

def split_class(train = True):
    if train:
        sp = 'train'
    else:
        sp = 'valid'
    f = open(f'annotations/annotation_{sp}.json')
    orion_anno = json.load(f)
    
    images = {}
    for img in orion_anno['images']:
        images[img['id']] = img['file_name']
    
    categories = {}
    for cat in orion_anno['categories']:
        categories[cat['id']] = cat['name']
    
    ex_img_id = ''
    for anno in tqdm(orion_anno['annotations']):
        os.makedirs(f'{sp}/{str(categories[anno["category_id"]])}/', exist_ok=True)
        try:
            if ex_img_id != anno['image_id']:
                img = Image.open(f'/workspace/230821_1369/images/{images[anno["image_id"]]}')
                img = ImageOps.exif_transpose(img)
                ex_img_id = anno['image_id']
            crop_img = img.crop((anno['bbox'][0],anno['bbox'][1],anno['bbox'][0] + anno['bbox'][2],anno['bbox'][1] + anno['bbox'][3]))
            crop_img.save(f'{sp}/{str(categories[anno["category_id"]])}/{str(anno["id"])}_{images[anno["image_id"]]}')

        except:
            print(f'images/{images[anno["image_id"]]}')
if __name__ == '__main__':
    split_class(train = True)
    split_class(train = False)