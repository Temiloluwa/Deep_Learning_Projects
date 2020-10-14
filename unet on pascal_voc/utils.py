import numpy as np

# read text file and return a list 
def read_file(file_path):
    with open(file_path, "r") as f:
        file_list = f.readlines()
    file_list = [l.strip("\n") for l in file_list]
    return file_list


def save_npy(np_array, filename):
    """Saves a .npy file to disk"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0\
        else filename
    with open(filename, "wb") as f:
        return np.save(f, np_array)


def load_npy(filename):
    """Reads a npy file"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0\
        else filename
    with open(filename, "rb") as f:
        return np.load(f)


def tensor_to_numpy(pytensor):
    """Converts pytorch tensor to numpy"""
    if pytensor.is_cuda:
        return pytensor.cpu().detach().numpy()
    else:
        return pytensor.detach().numpy()


# TO-DO - implement multi-class iou calculator
def calculate_iou(pred_x, targets):
    """Calculates iou"""
    iou_list = []
    for i in range(len(pred_x)):
        pred = np.argmax(tensor_to_numpy(pred_x[i]), 0).astype(int)
        target = tensor_to_numpy(targets[i]).astype(int)
        #iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    return mean_iou


def concat_img(img, skip_img, crop_only=False):
    _, _, h_i, w_i = img.shape
    if crop_only:
        _, h_s, w_s = skip_img.shape
    else:
        _, _, h_s, w_s = skip_img.shape
    h_idx = int((h_s - h_i)/2)
    w_idx = int((w_s - w_i)/2)

    if crop_only:
        skip_img = skip_img[:, h_idx:h_s - h_idx, w_idx: w_s - w_idx]
        return skip_img  

    skip_img = skip_img[:, :, h_idx:h_s - h_idx, w_idx: w_s - w_idx]
    output = torch.cat([img, skip_img], dim=1)
    return output


def calc_encoder_dims(x):
    """Calculates dimensions of unet encoder feature maps"""
    for i in range(5):
        print(f"layer: {5-i}, conv2: {x}, conv1: {x*2}, previous_layer_input: {x*2 + 4}")
        x = x*2 + 4