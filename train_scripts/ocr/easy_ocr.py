import os
import glob

import cv2
import easyocr

import pandas as pd


reader = easyocr.Reader(['en', 'id'])  # need to run only once to load model into memory

image_paths = '/home/ksenia/progas/kaggle/shopee/dataset/train_images/'
df = pd.read_csv('../image/reliable_validation_tm.csv')
df = df[df['label_group'] == 1141798720]

for image_name in df['image']:
    image_path = os.path.join(image_paths, image_name)
    result = reader.readtext(image_path)

    image = cv2.imread(image_path)
    for bbox, text, prob in sorted(result, key=lambda x: x[2], reverse=True):
        x1 = int(min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
        y1 = int(min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
        x2 = int(max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
        y2 = int(max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        result_text = '{} {}'.format(round(prob, 2), text)
        cv2.putText(image, result_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('1', image)
    if cv2.waitKey(0) == 27:
        break
    print(image_path, '\n')

# for image_path in glob.iglob(os.path.join(image_paths, '*')):
#     result = reader.readtext(image_path)
#
#     image = cv2.imread(image_path)
#     for bbox, text, prob in sorted(result, key=lambda x: x[2], reverse=True):
#         x1 = int(min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
#         y1 = int(min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
#         x2 = int(max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]))
#         y2 = int(max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]))
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
#         result_text = '{} {}'.format(round(prob, 2), text)
#         cv2.putText(image, result_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (255, 0, 0), 1, cv2.LINE_AA)
#
#     cv2.imshow('1', image)
#     if cv2.waitKey(0) == 27:
#         break
#     print(image_path, '\n')
