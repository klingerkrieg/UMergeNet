#Version 1.3

import torch
import time
import os
import re
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from torch.profiler import record_function, ProfilerActivity
from torch.profiler import profile as torch_profiler
from thop import profile
from ultralytics import YOLO


def show_dataset_prev(train_loader, test_loader, val_loader=None, num_images=3, num_classes=1):
    images_shown = 0

    # Creates an iterator for val_loader if provided
    val_iter = iter(val_loader) if val_loader is not None else None

    for (images_train, masks_train), (images_test, masks_test) in zip(train_loader, test_loader):
        if val_iter:
            try:
                images_val, masks_val = next(val_iter)
            except StopIteration:
                break  # Terminate if val_loader runs out

        for i in range(images_train.size(0)):
            if images_shown >= num_images:
                break

            def process_image_mask(img_tensor, mask_tensor):
                img_tensor = img_tensor.cpu()
                img = img_tensor.permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.squeeze(0).numpy()
                img = img * 0.5 + 0.5
                mask = mask_tensor.cpu().squeeze().numpy()
                return img, mask

            img_train, mask_train = process_image_mask(images_train[i], masks_train[i])
            img_test, mask_test = process_image_mask(images_test[i], masks_test[i])
            
            if val_iter:
                img_val, mask_val = process_image_mask(images_val[i], masks_val[i])

            # Sets number of columns based on val_loader
            n_cols = 6 if val_iter else 4
            fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 4))

            axs[0].imshow(img_train)
            axs[0].set_title("Training Image")
            axs[1].imshow(mask_train, cmap='viridis', vmin=0, vmax=num_classes)
            axs[1].set_title("Training Mask")

            axs[2].imshow(img_test)
            axs[2].set_title("Test Image")
            axs[3].imshow(mask_test, cmap='viridis', vmin=0, vmax=num_classes)
            axs[3].set_title("Test Mask")

            if val_iter:
                axs[4].imshow(img_val)
                axs[4].set_title("Val Image")
                axs[5].imshow(mask_val, cmap='viridis', vmin=0, vmax=num_classes)
                axs[5].set_title("Val Mask")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            images_shown += 1

        if images_shown >= num_images:
            break

    
def verificar_mascara_multiclasse(mascara, num_classes):
    if len(mascara.shape) != 2:
        print("Invalid format: the mask must be [H, W]")
    else:
        print("Format ok")

    valores = np.unique(mascara)
    if not np.all(np.equal(valores, valores.astype(int))):
        print("Mask contains non-integer values")
    else:
        print(f"Unique values: {valores}")

    if mascara.dtype not in [np.uint8, np.int32, np.int64]:
        print(f"Invalid type: {mascara.dtype}")
    else:
        print("Data type ok")

    if valores.min() < 0 or valores.max() >= num_classes:
        print(f"Values outside the expected range [0, {num_classes - 1}]")
    else:
        print("Value range is correct")


def beep():
    os.system('powershell.exe -Command "[console]::beep(500,400); [console]::beep(500,400)"')


def count_trainable_parameters(model, format=False):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if format:
        return f"{total:,}".replace(",", ".")
    return total


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def measure_inference_speed(model, test_loader, measure_cpu_speed=True, measure_gpu_speed=True, steps=100):
    model.eval()

    is_yolo = isinstance(model, YOLO)

    # Get only the first batch from test_loader
    inputs, _ = next(iter(test_loader))

    results = {}
    devices = []
    if measure_gpu_speed:
        devices.append('cuda')
    if measure_cpu_speed:
        devices.append('cpu')
    if len(devices) == 0:
        raise ValueError('A least one device must be True, measure_cpu_speed or measure_gpu_speed')

    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            results['gpu'] = (None, None)
            continue


        if is_yolo == False:
            model.to(device)
        inputs_device = inputs.to(device)

        # Heating
        with torch.no_grad():
            for _ in range(5):
                if is_yolo:
                    _ = model(inputs_device, verbose=False, stream=False)
                else:
                    _ = model(inputs_device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        # Measures pure routing time (100 executions of the same batch)
        with torch.no_grad():
            for _ in range(steps):
                if is_yolo:
                    _ = model(inputs_device, verbose=False, stream=False)
                else:
                    _ = model(inputs_device)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_image = total_time / (steps * inputs_device.size(0))
        time_per_image = f"{avg_time_per_image * 1000:.3f} ms"
        fps = 1.0 / avg_time_per_image

        results['gpu' if device == 'cuda' else 'cpu'] = (fps, time_per_image)

    fps_gpu, time_per_image_gpu = results.get('gpu', (None, None))
    fps_cpu, time_per_image_cpu = results.get('cpu', (None, None))

    if len(devices) == 2:
        return fps_gpu, time_per_image_gpu, fps_cpu, time_per_image_cpu
    elif measure_cpu_speed:
        return fps_cpu, time_per_image_cpu
    elif measure_gpu_speed:
        return fps_gpu, time_per_image_gpu




def measure_glops_fps(model, val_loader, model_filename=None, resolution=224):
    model.eval()
    torch.set_grad_enabled(False)

    if model_filename is not None:
        state_dict = torch.load(model_filename, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'], strict=False)

    # Measure FLOPs
    if isinstance(model, YOLO):
        # It's yolo
        model.info(verbose=True)
        params_m = 0
        gflops = 0
    else:
        device = next(model.parameters()).device
        inp = torch.randn(1, 3, resolution, resolution).to(device)
        macs, params = profile(model, inputs=(inp,))

        # Convert MACs (Multiply-Accumulate operations) to GFLOPs. 
        # We multiply by 2 because 1 MAC = 2 FLOPs.
        gflops = macs / 1e9
        params_m = params

    # Measure FPS
    gpu_fps, gpu_time_per_image, cpu_fps, cpu_time_per_image = measure_inference_speed(model, val_loader, steps=100)

    return {
        'params': params_m,
        'gflops': gflops,
        'gpu_fps': gpu_fps,
        'cpu_fps': cpu_fps
    }



def compile_xls_best_results(input_dir, output_file="result.xlsx"):
    linhas = []
    modelo_atual = None
    lines_block = []

    files = os.listdir(input_dir)
    files.sort()

    def add_block(linhas, bloco_linhas, modelo):
        if not bloco_linhas:
            return
        # Convert block to DataFrame
        df_bloco = pd.DataFrame(bloco_linhas)

        # Calculates averages (only for Data and FPS columns if they exist)
        avg_row = {"file": f"{modelo}-AVERAGE"}
        if "dice" in df_bloco.columns:
            avg_row["dice"] = df_bloco["dice"].mean()
        if "FPS" in df_bloco.columns:
            avg_row["FPS"] = df_bloco["FPS"].mean()

        # Add block + midline + blank line
        linhas.extend(df_bloco.to_dict("records"))
        linhas.append(avg_row)   #1st blank line with average
        linhas.append({})          # 2nd blank line

    for file in files:
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_dir, file)

            try:
                # Template base name (everything before the last "-number")
                match = re.match(r"(.+)-\d+$", file.replace("-epochs300.xlsx", ""))
                if match:
                    modelo = match.group(1)
                else:
                    modelo = file.replace(".xlsx", "")

                # But val_history tab
                val_history = pd.read_excel(file_path, sheet_name="val_history")

                # Gets the row with the highest value in the "data" column
                best_row = val_history.loc[val_history["dice"].idxmax()].copy()

                # But model_info tab
                model_info = pd.read_excel(file_path, sheet_name="model_info")

                # Look for the FPS column
                fps_value = model_info["FPS"].iloc[0] if "FPS" in model_info.columns else None

                # Converts the line to Series and adds FPS
                best_row["FPS"] = fps_value
                best_row = pd.Series({"file": file.replace(".xlsx", ""), **best_row.to_dict()})

                # If you change models, close the previous block
                if modelo_atual is not None and modelo != modelo_atual:
                    add_block(linhas, lines_block, modelo_atual)
                    lines_block = []

                # Update current model
                modelo_atual = modelo

                #Add line to block
                lines_block.append(best_row)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Finish last block
    if lines_block:
        add_block(linhas, lines_block, modelo_atual)

    # Joins all rows into a DataFrame
    df_final = pd.DataFrame(linhas)

    # Saves to Excel
    df_final.to_excel(output_file, index=False)
    print(f"File saved in: {output_file}")



def get_next_run_dir(base_dir='./tb_profiler', prefix='run_'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    existing_ids = []
    for d in existing:
        try:
            existing_ids.append(int(d.replace(prefix, '')))
        except ValueError:
            continue
    next_id = max(existing_ids, default=0) + 1
    return os.path.join(base_dir, f'{prefix}{next_id}')

def run_profiler(model, data_loader, model_name='model', num_steps=1):
    device = 'cuda'
    logdir = get_next_run_dir('./tb_profiler_streams', prefix=f'{model_name}_run_')

    with torch_profiler(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for i, (images, masks) in enumerate(data_loader):
                if i >= num_steps:
                    break
                images = images.to(device)
                masks  = masks.to(device)
                model.to(device)

                with record_function("model_inference"):
                    _ = model(images)
                #Torch.cuda.synchronize()

    print(f"Profiling results saved to: {logdir}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def get_flops_gflops(model, input_size=(1, 3, 224, 224), device='cuda'):
    dummy_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    gflops = flops / 1e9  # converts to GFLOPs
    return gflops