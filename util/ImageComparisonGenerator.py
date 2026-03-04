#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from scipy.ndimage import binary_erosion
from enum import Enum

#jupyter nbconvert --to script ImageComparisonGenerator.ipynb
#Version 1.5
# DiffMode, Drawing rects, arrows 
class DiffMode(Enum):
    DIAGONAL_DOTS_WITH_BORDER   = 1
    DIAGONAL_DOTS               = 2
    HORIZONTAL_DOTS_WITH_BORDER = 3
    HORIZONTAL_DOTS             = 4
    SPARSE_DOTS_WITH_BORDER     = 5
    SPARSE_DOTS                 = 6
    THIN_DOTS_WITH_BORDER       = 7
    THIN_DOTS                   = 8
    APPLY_RED                   = 9
    APPLY_RED_WITH_ALPHA        = 10


class ImageComparisonGenerator:

    color_map = {0:[0.9, 0.9, 0.0],
                 1:[0.3, 0.3, 1.0 ],
                 2:[0.7, 0.7, 1.0]}
    rectangle_color = (0.2, 0.8, 0.2)

    font_size = 35

    def __init__(self, model, model_name1="Prediction", model2=None, model_name2="Prediction", model3=None, model_name3="Prediction", color_map=None):
        self.model  = model
        self.model_name1 = model_name1
        self.model2 = model2
        self.model_name2 = model_name2
        self.model3 = model3
        self.model_name3 = model_name3

        if color_map == 'binary':
            self.set_binary_color_map()
        elif color_map is not None:
            self.color_map = color_map

    def set_binary_color_map(self):
        self.color_map = {0:[0.0, 0.0, 0.0], 1:[1.0, 1.0, 1.0]}

    def set_color_map(self, color_map):
        self.color_map = color_map
    
    def set_rectangle_color(self, color):
        self.rectangle_color = color

    def get_model_output(self,images, model=None):
        if model is None:
            return self.model(images)
        else:
            return model(images)
    
    # Helper function within the class
    def _get_sample_by_index(self, dataloader, idx):
        count = 0
        for imgs, masks in dataloader:
            batch_size = imgs.shape[0]
            if idx < count + batch_size:
                local_idx = idx - count
                return imgs[local_idx:local_idx+1], masks[local_idx]
            count += batch_size
        raise IndexError(f"Index {idx} out of range for dataset of length {len(dataloader.dataset)}")


    # Auxiliary functions
    def _prepare_mask_vis(self, mask, num_classes=1, ignore_val=255):
        ignore_mask = (mask == ignore_val)
        mask_vis = mask.copy()
        
        if num_classes > 1:
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            for class_val, color in self.color_map.items():
                mask_rgb[mask == class_val] = color
            mask_rgb[ignore_mask] = [0.8,0.8,0.8]
            mask_vis = mask_rgb
        else:
            mask_vis[ignore_mask] = -1
        return mask_vis, ignore_mask

    def _prepare_prediction_vis(self, output, mask, num_classes=1, do_diff=True, invert_diff_colors=False, ignore_mask=None, diff_mode=DiffMode.DIAGONAL_DOTS_WITH_BORDER):
        if num_classes == 1:
            out_sigmoid = torch.sigmoid(output[0])
            pred = (out_sigmoid > 0.5).float().cpu().squeeze().numpy()
            num_classes = 2
        else:
            out_softmax = torch.softmax(output[0], dim=0)
            pred = torch.argmax(out_softmax, dim=0).cpu().numpy()

        if do_diff:
            if len(mask.shape) == 2:
                h, w = mask.shape
            else:
                h, w, _ = mask.shape

            diff_img = np.zeros((h, w, 3), dtype=np.float32)
            
            # First, paint everything with the expected class.
            for class_id in range(num_classes):
                pred_mask_cls = (pred == class_id)
                diff_img[pred_mask_cls] = self.color_map.get(class_id,[1,1,1])

            # Then paint only real errors in red (ignoring 255)
            # Wherever the value 255 is, it will not be painted red, preserving
            # what was predicted by the model
            mismatches = (mask != pred) & (~ignore_mask)
            rows, cols = np.indices((h, w))
            if diff_mode in [DiffMode.DIAGONAL_DOTS_WITH_BORDER, DiffMode.DIAGONAL_DOTS]:
                checker_pattern = (rows - cols) % 4 == 0 # diagonal
            elif diff_mode in [DiffMode.HORIZONTAL_DOTS_WITH_BORDER, DiffMode.HORIZONTAL_DOTS]:
                checker_pattern = rows % 2 == 0 # horizontal
            elif diff_mode in [DiffMode.SPARSE_DOTS_WITH_BORDER, DiffMode.SPARSE_DOTS]:
                checker_pattern = (rows % 3 == 0) & (cols % 3 == 0) # sparse
            elif diff_mode in [DiffMode.THIN_DOTS_WITH_BORDER, DiffMode.THIN_DOTS]:
                checker_pattern = (rows + cols) % 2 == 0 # thin              
            
            # Old mode
            if diff_mode == DiffMode.APPLY_RED:
                # if its binary
                if num_classes == 2:
                    tp = (pred == 1) & (mask == 1)
                    fn = (pred == 0) & (mask == 1)
                    fp = (pred == 1) & (mask == 0)
                    diff_img[tp] = [1,1,1]
                    diff_img[fp] = [1,0.5,0]
                    diff_img[fn] = [1,0,0]
                else:
                    diff_img[mismatches] = [1,0,0]
            # Red with alpha
            elif diff_mode == DiffMode.APPLY_RED_WITH_ALPHA:
                diff_img[mismatches] = 0.5 * diff_img[mismatches] + 0.5 * np.array([1,0,0])
            else:
            # Dotted patterns
                dotted_errors = mismatches & checker_pattern
                diff_img[dotted_errors] = [1, 0, 0]

            # Border
            if diff_mode in [DiffMode.DIAGONAL_DOTS_WITH_BORDER, DiffMode.HORIZONTAL_DOTS_WITH_BORDER, DiffMode.SPARSE_DOTS_WITH_BORDER, DiffMode.THIN_DOTS_WITH_BORDER]:
                eroded = binary_erosion(mismatches)
                border = mismatches & (~eroded)
                diff_img[border] = [1, 0, 0]
                
            return diff_img
        else:
            if num_classes > 1:
                pred_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
                for class_val, color in self.color_map.items():
                    pred_vis[pred == class_val] = color
                return pred_vis
            else:
                return pred

    def _prepare_image_disp(self, img):
        img_disp = img.cpu().squeeze()
        if img_disp.ndim == 3 and img_disp.shape[0] == 3:
            img_disp = img_disp.permute(1,2,0)
        elif img_disp.ndim == 3 and img_disp.shape[0] == 1:
            img_disp = img_disp.squeeze(0)
        img_disp = img_disp*0.5 + 0.5
        return img_disp
    

    # print images in list
    def save_output_row(self, sample_loader, samples=[0],
                        num_classes=1, do_diff=True, invert_diff_colors=False,
                        do_save=False, figsize=(None,None), 
                        diff_mode=DiffMode.DIAGONAL_DOTS_WITH_BORDER,
                        rectangles=None, arrows=None,):
        if self.model is None:
            raise Exception("The model is not loaded.")

        device = next(self.model.parameters()).device
        num_rows = len(samples)

        self.model.eval()
        fig_width = 11
        fig_height = 3*num_rows
        if self.model2 is not None:
            self.model2.eval()
            fig_width = 16
            fig_height = 3*num_rows
        if self.model3 is not None:
            self.model3.eval()
            fig_width = 22
            fig_height = 4*num_rows

        if figsize == (None, None):
            figsize = (fig_width, fig_height)
            
        fig = plt.figure(figsize=figsize)

        with torch.no_grad():
            for idx, sample_idx in enumerate(samples):
                img, mask = self._get_sample_by_index(sample_loader, sample_idx)
                img = img.to(device)
                mask = mask.cpu().squeeze().numpy()

                mask_vis, ignore_mask = self._prepare_mask_vis(mask, num_classes)
                pred_vis = self._prepare_prediction_vis(self.get_model_output(img), mask, num_classes, do_diff, invert_diff_colors, ignore_mask, diff_mode=diff_mode)
                img_disp = self._prepare_image_disp(img)
                

                # Model 2 prediction, if it exists
                if self.model2 is not None:
                    img2 = img.to(device)
                    pred2_vis = self._prepare_prediction_vis(self.get_model_output(img2, model=self.model2), mask, num_classes, do_diff, invert_diff_colors, ignore_mask, diff_mode=diff_mode)
                    num_cols = 4


                    # Model 3 prediction, if it exists
                    if self.model3 is not None:
                        img3 = img.to(device)
                        pred3_vis = self._prepare_prediction_vis(self.get_model_output(img3, model=self.model3), mask, num_classes, do_diff, invert_diff_colors, ignore_mask, diff_mode=diff_mode)
                        num_cols = 5
                        enum_list = [img_disp, mask_vis, pred_vis, pred2_vis, pred3_vis]
                    else:
                        pred3_vis = None
                        enum_list = [img_disp, mask_vis, pred_vis, pred2_vis]
                else:
                    pred2_vis = None
                    pred3_vis = None
                    num_cols = 3
                    enum_list = [img_disp, mask_vis, pred_vis]


                for col, im in enumerate(enum_list):
                    ax = fig.add_axes([col/num_cols, 1-(idx+1)/num_rows, 1/num_cols, 1/num_rows])
                    ax.imshow(im, cmap='gray' if num_classes==1 else None, aspect='auto')
                    ax.set_xticks([]); ax.set_yticks([])

                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)

                    if idx==0:
                        if col==0: ax.set_title("Image", fontsize=self.font_size, y=1.0)
                        elif col==1: ax.set_title("Ground Truth", fontsize=self.font_size, y=1.0)
                        elif col == 2:
                            ax.set_title(self.model_name1, fontsize=self.font_size, y=1.0)
                        elif col == 3:
                            ax.set_title(self.model_name2, fontsize=self.font_size, y=1.0)
                        elif col == 4:
                            ax.set_title(self.model_name3, fontsize=self.font_size, y=1.0)

        # draw rects and arrows
        if rectangles is not None:
            self._draw_rectangles(fig, rectangles)
        if arrows is not None:
            self._draw_arrows(fig, arrows)

        #plt.tight_layout(rect=[0,0,1,0.96])
        if do_save: fig.savefig(do_save, format='eps', bbox_inches='tight', pad_inches=0.1)
        else: plt.show()


    def save_output_quad(self, sample_loader, samples=[0],
                        num_classes=1, do_diff=True, invert_diff_colors=False,
                        do_save=False,
                        figsize=(10,10),
                        vertical_gap=0.13,
                        before_plot=None, diff_mode=DiffMode.DIAGONAL_DOTS_WITH_BORDER,
                        rectangles=None, arrows=None):
        if self.model is None:
            raise Exception("The model is not loaded.")

        device = next(self.model.parameters()).device

        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()

        fig = plt.figure(figsize=figsize)  # greater width for 2 models

        with torch.no_grad():
            for idx, sample_idx in enumerate(samples):
                img, mask = self._get_sample_by_index(sample_loader, sample_idx)
                img = img.to(device)
                mask_np = mask.cpu().squeeze().numpy()

                mask_vis, ignore_mask = self._prepare_mask_vis(mask_np, num_classes)

                # Model 1 Prediction
                pred1_vis = self._prepare_prediction_vis(
                    self.get_model_output(img, model=self.model), 
                    mask_np, num_classes, do_diff, invert_diff_colors, ignore_mask, diff_mode=diff_mode
                )
                img_disp = self._prepare_image_disp(img)

                # Model 2 prediction, if it exists
                if self.model2 is not None:
                    img2 = img.to(device)
                    pred2_vis = self._prepare_prediction_vis(
                        self.get_model_output(img2, model=self.model2),
                        mask_np, num_classes, do_diff, invert_diff_colors, ignore_mask, diff_mode=diff_mode
                    )
                else:
                    pred2_vis = None

                height = (1 - vertical_gap) / 2

                # Position structure: (row, column, image, title)
                positions = [
                    (0,0,img_disp,"Image"),
                    (0,1,mask_vis,"Ground Truth"),
                    (1,0,pred1_vis,self.model_name1),
                    (1,1,pred2_vis,self.model_name2 if self.model2 is not None else "Prediction")
                ]

                for row, col, im, title in positions:
                    bottom = 1 - (row + 1)*(height + vertical_gap/2)
                    ax = fig.add_axes([col/2, bottom, 1/2, height])

                    if im is not None:
                        if before_plot is not None:
                            im = before_plot(im)
                        ax.imshow(im, cmap='gray' if num_classes==1 else None, aspect='auto')
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)
                    
                    ax.set_title(title, fontsize=self.font_size, y=1.0)


        # draw rects and arrows
        if rectangles is not None:
            self._draw_rectangles(fig, rectangles)
        if arrows is not None:
            self._draw_arrows(fig, arrows)

        
        if do_save:
            fig.savefig(do_save, format='eps', bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()


    rect_thickness = 4
    def _draw_rectangles(self,fig,rectangles):
        for rect in rectangles:
            (x, y), (w, h) = rect
            y_corr = 1 - y - h
            rectangle = patches.Rectangle(
                (x, y_corr), w, h,
                linewidth=self.rect_thickness,
                edgecolor=self.rectangle_color,
                facecolor='none',
                transform=fig.transFigure,
                clip_on=False
            )

            fig.add_artist(rectangle)


    arrow_thickness = 4
    arrow_head_size = 30

    def _draw_arrows(self, fig, arrows):
        for (x1, y1), (x2, y2) in arrows:
            y1_mpl = 1 - y1
            y2_mpl = 1 - y2
            arrow = patches.FancyArrowPatch(
                (x1, y1_mpl),
                (x2, y2_mpl),
                arrowstyle='->',
                mutation_scale=self.arrow_head_size,
                color=self.rectangle_color,
                linewidth=self.arrow_thickness,
                transform=fig.transFigure,
                zorder=1000
            )

            fig.patches.append(arrow)


# In[ ]:


def load_model(model, model_file_name):
    checkpoint = torch.load(model_file_name, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # Filters only existing keys in the model
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    model.load_state_dict(filtered_state_dict, strict=False)
    return model


# In[ ]:


from PIL import Image
def combine_images(img1_path, img2_path, output):
    # Abrir imagens EPS
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Converter para RGB (evita problemas)
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    # Criar nova imagem com largura combinada
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Colocar imagens lado a lado
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    # Salvar como EPS novamente
    combined.save(output, format="EPS")

