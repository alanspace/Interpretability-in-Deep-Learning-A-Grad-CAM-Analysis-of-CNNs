# import matplotlib.pyplot as plt 
# import numpy as np 
# import torch


# def show_image(image,mask,pred_image = None):
    
#     if pred_image == None:
        
#         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
#         ax1.set_title('IMAGE')
#         ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
#         ax2.set_title('GROUND TRUTH')
#         ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
#     elif pred_image != None :
        
#         f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
#         ax1.set_title('IMAGE')
#         ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
#         ax2.set_title('GROUND TRUTH')
#         ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
#         ax3.set_title('MODEL OUTPUT')
#         ax3.imshow(pred_image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        

# In Road_seg_dataset/helper.py

import matplotlib.pyplot as plt
import torch
import numpy as np

def show_image(image, mask, pred_mask=None, save_path=None):
    """
    Displays the original image, ground truth mask, and optional predicted mask.
    Optionally saves the figure to a file if a save_path is provided.

    Args:
        image (torch.Tensor or np.ndarray): The original image.
        mask (torch.Tensor or np.ndarray): The ground truth mask.
        pred_mask (torch.Tensor or np.ndarray, optional): The predicted mask. Defaults to None.
        save_path (str, optional): The file path to save the figure. Defaults to None.
    """
    # Convert PyTorch tensors to NumPy arrays suitable for plotting
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    if pred_mask is not None and isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy()

    # Determine the number of subplots needed
    num_plots = 2 if pred_mask is None else 3
    plt.figure(figsize=(18, 6))

    # Plot Original Image
    plt.subplot(1, num_plots, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot Ground Truth Mask
    plt.subplot(1, num_plots, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    # Plot Predicted Mask if available
    if pred_mask is not None:
        plt.subplot(1, num_plots, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

    plt.tight_layout()

    # --- THIS IS THE CRUCIAL ADDITION ---
    # Save the figure to the specified path before showing it
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Figure saved to: {save_path}")
    
    # Display the plot
    plt.show()