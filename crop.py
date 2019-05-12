import numpy as np
import cv2, os
import pdb
image_path = 'JPEGImages/Full-Resolution'
anno_path = 'Annotations/Full-Resolution'
output_path = 'Cropped'
sequences = os.listdir(image_path)
for sequence in sequences:
    video_path = os.path.join(image_path, sequence)
    img_ids = os.listdir(video_path)
    for k in img_ids:
        img = cv2.imread(os.path.join(video_path, k))
        mask_file = cv2.imread(os.path.join(anno_path, sequence, k[:-3]+'png'))
        id = 0
        mask_file = mask_file.astype(int)
        mask_file = mask_file[:,:,0] + 255 * mask_file[:,:,1] + 255 *255 * mask_file[:,:,2]
        for j in np.unique(mask_file):
            if j:
                mask=(mask_file==j)*255
                mask = mask.astype(np.uint8)
                rect = cv2.boundingRect(mask)               # function that computes the rectangle of interest
                print(rect)
                left = max(rect[0]-50, 0)
                right = min(rect[0]+rect[2]+50, img.shape[1])
                top = max(rect[1]-50, 0)
                bottom = min(rect[1]+rect[3]+50, img.shape[0])
                cropped_img = img[top:bottom, left: right]  # crop the image to the desired rectangle
                cropped_mask = mask[top:bottom, left: right]
                cropped_img = cv2.resize(cropped_img, (28,28),interpolation=cv2.INTER_CUBIC)
                cropped_mask = cv2.resize(cropped_mask, (28,28),interpolation=cv2.INTER_NEAREST)
                id += 1
                if not os.path.exists(os.path.join(output_path, sequence, 'JPEGImage')):
                    os.makedirs(os.path.join(output_path, sequence, 'JPEGImage'))
                if not os.path.exists(os.path.join(output_path, sequence, 'Annotations')):
                    os.makedirs(os.path.join(output_path, sequence, 'Annotations'))
                cv2.imwrite(os.path.join(output_path, sequence, 'JPEGImage', k[:-4]+'_'+str(id)+'.jpg' ),cropped_img)
                cv2.imwrite(os.path.join(output_path, sequence, 'Annotations', k[:-4]+'_'+str(id)+'.png' ),cropped_mask)
