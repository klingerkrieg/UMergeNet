#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#jupyter nbconvert --to script DatasetAugmentation.ipynb
#Version 1.0
import os
import cv2
import random
import shutil
import albumentations as A
from tqdm import tqdm
from PIL import Image
import numpy as np

images_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

# -----------------------------
# Auxiliary functions
# -----------------------------

def copy_and_fix(img_src_dir, mask_src_dir, img_out_dir, mask_out_dir, selected_files=None, 
                 function_to_apply_to_masks=None, mask_suffix=''):
    """
    Copies images and masks from img_src_dir/mask_src_dir to img_out_dir/mask_out_dir.
    If selected_files is None, copies all files from the source folder.
    Accepts masks with any extension (.png, .jpg, .jpeg .bmp).
    """
    if not os.path.exists(img_src_dir) or not os.path.exists(mask_src_dir):
        return 0  # nothing to copy

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    if selected_files is None:
        files = sorted([f for f in os.listdir(img_src_dir) if f.lower().endswith(images_extensions)])
    else:
        files = selected_files

    count = 0
    for f in tqdm(files, desc=f"Copiando {os.path.basename(img_out_dir)}"):
        img_src = os.path.join(img_src_dir, f)
        base = os.path.splitext(f)[0]

        # search for the mask with any extension
        possible_mask_paths = [
            os.path.join(mask_src_dir, base + mask_suffix + ext)
            for ext in images_extensions
        ]
        mask_src = next((p for p in possible_mask_paths if os.path.exists(p)), None)

        if mask_src and os.path.exists(img_src):
            # Copy image
            shutil.copy(img_src, os.path.join(img_out_dir, base + ".png"))
            # Copy mask, convert to PNG
            mask = np.array(Image.open(mask_src).convert("L"))
            if function_to_apply_to_masks is not None:
                #Apply the correction function
                mask = function_to_apply_to_masks(mask)
            Image.fromarray(mask).save(os.path.join(mask_out_dir, base + ".png"))
            count += 1
    return count



# -----------------------------
# Augmentation for training
# -----------------------------
def augment_train_images(image_list, image_dir, mask_dir, output_image_dir, output_mask_dir, transforms, N, 
                         function_to_apply_to_masks=None, mask_suffix=''):
    for img_name in tqdm(image_list, desc="Enlarging workout images"):
        # Base name without extension
        base_name = os.path.splitext(img_name)[0]

        # Possible paths (.png, .jpg, .jpeg)
        possible_img_paths = [
            os.path.join(image_dir, base_name + ext) for ext in images_extensions
        ]
        possible_mask_paths = [
            os.path.join(mask_dir, base_name + mask_suffix + ext) for ext in images_extensions
        ]

        # Choose the first file that exists
        img_path = next((p for p in possible_img_paths if os.path.exists(p)), None)
        mask_path = next((p for p in possible_mask_paths if os.path.exists(p)), None)

        if img_path is None or mask_path is None:
            print(f"[WARNING] File not found for {base_name}. Jumping.")
            print(possible_mask_paths)
            continue

        # Reads image and mask
        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if function_to_apply_to_masks is not None:
            mask  = function_to_apply_to_masks(mask)


        if image is None or mask is None:
            print(f"[WARNING] Failed to read {base_name}. Jumping.")
            continue

        # Transform and save original version as PNG
        orig = transforms(image=image, mask=mask)
        cv2.imwrite(os.path.join(output_image_dir, f"{base_name}_orig.png"), orig['image'])
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_orig.png"), orig['mask'])

        # Generates augmentations
        for i in range(N):
            aug = transforms(image=image, mask=mask)
            cv2.imwrite(os.path.join(output_image_dir, f"{base_name}_aug{i}.png"), aug['image'])
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_aug{i}.png"), aug['mask'])



def augment_dataset(N, num_to_valid, num_to_test,  
                    orig_train_img_dir, orig_train_mask_dir,
                    orig_valid_img_dir, orig_valid_mask_dir,
                    orig_test_img_dir, orig_test_mask_dir,
                    output_base,
                    transforms,
                    function_to_apply_to_masks=None,
                    mask_suffix=''):

    output_dirs = {
        'train_images': os.path.join(output_base, 'images/train'),
        'train_labels': os.path.join(output_base, 'labels/train'),
        'valid_images': os.path.join(output_base, 'images/valid'),
        'valid_labels': os.path.join(output_base, 'labels/valid'),
        'test_images':  os.path.join(output_base, 'images/test'),
        'test_labels':  os.path.join(output_base, 'labels/test'),
    }

    # -----------------------------
    # Select images for splits
    # -----------------------------
    all_images = sorted([f for f in os.listdir(orig_train_img_dir) if f.lower().endswith(images_extensions)])
    total_imgs = len(all_images)

    if num_to_valid + num_to_test >= total_imgs:
        print("num_to_valid:",num_to_valid, "num_to_test:", num_to_test, "total_imgs:",total_imgs)
        raise ValueError("Number of images for valid+test is greater than or equal to the total available.")

    # Random samples (without modifying the original directory)
    selected_test = set(random.sample(all_images, num_to_test))
    remaining = [f for f in all_images if f not in selected_test]

    selected_valid = set(random.sample(remaining, num_to_valid))
    remaining = [f for f in remaining if f not in selected_valid]

    train_images = remaining

    # -----------------------------
    # Total estimated images generated
    # -----------------------------
    train_total = len(train_images)
    total_output = train_total * (N + 1)

    print(f"Total images in the original dataset: {total_imgs}")
    print(f"→ Training: {len(train_images)}")
    print(f"→ Validation (of training): {len(selected_valid)}")
    print(f"→ Test (training): {len(selected_test)}")
    print(f"\nWith N={N}, total images generated in training will be: {total_output}")

    choice = input("Do you want to proceed? (y/n):").strip().lower()
    if choice not in ('y', 's'):
        print("\n\nProcess cancelled.")
        raise SystemExit

    # -----------------------------
    # Creating folders
    # -----------------------------
    if any(os.path.exists(d) for d in output_dirs.values()):
        print("\n\n**The output directory already exists. Aborting to avoid overwriting.**")
        raise SystemExit
    else:
        for d in output_dirs.values():
            os.makedirs(d, exist_ok=True)





    # -----------------------------
    # Copy existing valid/test folders
    # -----------------------------
    count_valid_existing = copy_and_fix(orig_valid_img_dir, orig_valid_mask_dir,
                                        output_dirs['valid_images'], output_dirs['valid_labels'],
                                        function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
    count_test_existing  = copy_and_fix(orig_test_img_dir, orig_test_mask_dir,
                                        output_dirs['test_images'], output_dirs['test_labels'],
                                        function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)

    print(f"→ {count_valid_existing} images copied from the original valid folder.")
    print(f"→ {count_test_existing} images copied from the original test folder.")

    # -----------------------------
    # Mover imagens do train para valid/test se num_to_valid/num_to_test > 0
    # -----------------------------
    # Valid
    if num_to_valid > 0:
        selected_valid = list(selected_valid)
        copied = copy_and_fix(orig_train_img_dir, orig_train_mask_dir,
                            output_dirs['valid_images'], output_dirs['valid_labels'], selected_valid,
                            function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
        print(f"→ {copied} images copied from the train for validation.")

    # Test
    if num_to_test > 0:
        selected_test = list(selected_test)
        copied = copy_and_fix(orig_train_img_dir, orig_train_mask_dir,
                            output_dirs['test_images'], output_dirs['test_labels'], selected_test,
                            function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
        print(f"→ {copied} images copied from the train for testing.")



    # -----------------------------
    # Process the training set
    # -----------------------------
    augment_train_images(train_images, orig_train_img_dir, orig_train_mask_dir,
                    output_dirs['train_images'], output_dirs['train_labels'],
                    transforms, N,
                    function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)

    # -----------------------------
    # Final report
    # -----------------------------
    def count_images_in_dir(directory):
        return len([f for f in os.listdir(directory) if f.lower().endswith(images_extensions)])

    print("\nFinal summary:")
    for key, path in output_dirs.items():
        print(f"{key}: {count_images_in_dir(path)} files")

