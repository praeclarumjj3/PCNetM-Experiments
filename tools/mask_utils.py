import matplotlib.pyplot as plt
import os
import numpy as np


def tensor_to_list(maps):
    
    masks = []
    maps = maps.squeeze(0)

    for i in range(maps.shape[0]):
        masks.append(maps[i])

    return masks


def visualize(x,name):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    plt.rcParams.update({'font.size': 3})
    dim = int(x.shape[1])
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(int(dim**0.5)*int(dim**0.5)):
        r = int(j/int(dim**0.5))
        c = int(j%int(dim**0.5))
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(name))

def visualize_single_map(mapi, name):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    x = mapi[0]
    x = np.uint8(x)
    plt.imsave('visualizations/{}.jpg'.format(name), np.squeeze(x))

def save_image(img, name):
    if not os.path.exists('visualizations/imgs'):
            os.makedirs('visualizations/imgs')
    plt.imsave('visualizations/imgs/'+name+'.jpg', img)

def visualize_run(img, modal, amodal_pred, i):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    plt.rcParams.update({'font.size': 10})

    modal = modal[i] 
    modal = np.uint8(modal)

    amodal_pred = amodal_pred[i]
    amodal_pred = np.uint8(amodal_pred)

    f, (ax1,ax2,ax3) = plt.subplots(1,3)

    ax1.imshow(img)
    ax1.set_title("Image")
    ax1.axis('off')

    ax2.imshow(modal)
    ax2.set_title("Modal Mask")
    ax2.axis('off')

    ax3.imshow(amodal_pred)
    ax3.set_title("Amodal Mask")
    ax3.axis('off')

    f.savefig('visualizations/preds_{}.jpg'.format(i))