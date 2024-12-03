
import torch
import numpy as np
from MultiViewModel import AttentionMultiViewFusionNet
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from GradCAM_utils import GradCAM, show_cam_on_image, center_crop_img


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.model.patch_embed.img_size
        patch_size = model.model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        print('ReshapeTr中，x的尺寸：', x.shape)
        result = x[:, 1:, :].reshape(2,
                                     7,
                                     7,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        # print(result.shape)
        return result


def main():
    model = AttentionMultiViewFusionNet(arch="vit_tiny_r_s16_p8_224", num_classes=2, n=2)
    model.load_state_dict(torch.load("/home/wcy/Smoke_dt/multi-view-hybrid-main/output_final/exp_2/best.pth", map_location='cpu')['model'])
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.model.blocks[-1].norm1]

    data_transform = transforms.Compose([ 
                                        transforms.ToTensor()])

    # load image
    # img_path = "both.png"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)
    # [C, H, W]
    # img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    # input_tensor = torch.unsqueeze(img_tensor, dim=0)
    '''
    pos_1: 56
    pos_2: 37
    pos_3: 28
    pos_4: 50
    pos_5: 55
    '''
    img_v = Image.open('/data1/wcy/data/smoke/pos_1_1/pair56/image_vis_56.png').convert('RGB')
    img_i = Image.open('/data1/wcy/data/smoke/pos_1_1/pair56/image_ir_56.png').convert('RGB')
    img_v = np.array(img_v, dtype=np.uint8)
    img_i = np.array(img_i, dtype=np.uint8)
    img_v = center_crop_img(img_v, 224)
    img_i = center_crop_img(img_i, 224)
    img_v_tensor = data_transform(img_v)
    img_i_tensor = data_transform(img_i)
    img_tensor = torch.stack([img_v_tensor, img_i_tensor], dim=0)
    input_tensor = img_tensor.unsqueeze(0)
    
    # input_tensor = torch.randn(1, 2, 3, 224, 224)
    # img = np.random.randint(0, 256, size=[224, 224, 3], dtype=np.uint8)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization_v = show_cam_on_image(img_v / 255., grayscale_cam, use_rgb=True)
    visualization_i = show_cam_on_image(img_i / 255., grayscale_cam, use_rgb=True)
    
    plt.imsave('/home/wcy/Smoke_dt/multi-view-hybrid-main/gradcam_visual/None/s/cam_v_p1_1.png', visualization_v)
    plt.imsave('/home/wcy/Smoke_dt/multi-view-hybrid-main/gradcam_visual/None/s/cam_i_p1_1.png', visualization_i)


if __name__ == '__main__':
    main()
