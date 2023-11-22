import cv2
import os
import numpy as np
import torch
import xml.etree.ElementTree as ET
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from utils.fusion_strategy import addition_fusion, attention_fusion_weight
plt.style.use('ggplot')
sns.set_style("darkgrid")

class Utils():

    @staticmethod
    def stack_images(imgs, device='cpu', ablation_flag=False):
        if ablation_flag:
            return torch.stack(list(image.to(device, dtype=torch.float) for image in imgs))    
        return torch.stack(list(image.to(device) for image in imgs))
    

    @staticmethod
    def batch_images_for_encoder_decoder(imgs, device='cpu', ablation_flag=False):
        return Utils.stack_images(imgs=imgs, device=device, ablation_flag=ablation_flag)


    @staticmethod
    def list_targets(targets, device='cpu', detach=False, detector_name='ssd'):
        
        if 'fcos' in detector_name: # ps: tested on gpu.

            if detach:
                return [{k: (v.float().detach().to(device) if not isinstance(v, str) else v) if k == 'boxes' else
                        (v.detach().to(device) if not isinstance(v, str) else v)
                     for k, v in t.items()
                     } for t in targets]

            return [{k: (v.float().to(device) if not isinstance(v, str) else v) if k == 'boxes' else
                        (v.to(device) if not isinstance(v, str) else v)
                     for k, v in t.items()
                     } for t in targets]
                
        if detach:
            return [{k: (v.detach().to(device) if not isinstance(v, str) else v)  for k, v in t.items()} for t in targets]
        return [{k: (v.to(device) if not isinstance(v, str) else v) for k, v in t.items()} for t in targets]


    @staticmethod
    def batch_targets_for_detector(targets, device='cpu', detach=False, detector_name='ssd'):
        return Utils.list_targets(targets=targets, device=device, detach=detach, detector_name=detector_name)


    @staticmethod
    def remove_small_boxes(boxes, min_size):
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        return boxes[keep]


    @staticmethod
    def filter_pseudo_label(targets, images, device):
        _, _, height, width = images.shape
        
        return [{k: (v.detach().to(device) if k != 'boxes' else 
                            Utils.remove_small_boxes(
                                torchvision.ops.clip_boxes_to_image(v.detach().to(device), size=(height, width)),
                                min_size=10.0
                                )
                            )
                            for k, v in t.items()} for t in targets]


    @staticmethod
    def create_pseudo_labels(detector, imgs, device, threshold=0.5):

        detector.eval()
        output_pseudo_label = detector(imgs)

        final_result = []
        for idx, img in enumerate(imgs):

            output_pseudo_label_th = {}
            output_pseudo_label_th['boxes'] = output_pseudo_label[idx]['boxes'][output_pseudo_label[idx]['scores'] > threshold].to(device)
            output_pseudo_label_th['scores'] = output_pseudo_label[idx]['scores'][output_pseudo_label[idx]['scores'] > threshold].to(device)
            output_pseudo_label_th['labels'] = output_pseudo_label[idx]['labels'][output_pseudo_label[idx]['scores'] > threshold].to(device)
        
            if output_pseudo_label_th is None or output_pseudo_label_th['boxes'].numel() == 0:
                output_pseudo_label_th['boxes'] = torch.zeros((1,4)).to(device)
                output_pseudo_label_th['scores'] = torch.zeros((1)).to(device)
                output_pseudo_label_th['labels'] = torch.zeros((1), dtype=torch.int64).to(device)

            final_result.append(output_pseudo_label_th)
        
        detector.train()
        return final_result


    @staticmethod
    def expand_one_channel_to_output_channels(imgs, output_channels=3):
        return imgs.repeat(1, output_channels, 1, 1)


    @staticmethod
    def concat_modalities(img_rgb, img_ir):
        return torch.cat([img_rgb, img_ir], dim=0)


    @staticmethod
    def sum_per_batch(input_tensor):
        sum_batch = torch.zeros(input_tensor[0].size(), device=input_tensor[0].device)
        for x in input_tensor:
            sum_batch += x
        return sum_batch/len(input_tensor)


    @staticmethod
    def plot_histogram(hist, color='forestgreen', cont=0, clear=False):

        # hist = torch.histc(hist, bins=256, min=0.0, max=1.0, out=None).detach().cpu().numpy()
        # plt.bar(range(256), hist, align='center', color=[color])
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        # plt.show()

        p = sns.distplot(hist*256, hist = False, kde = True,
                        kde_kws = {'shade': True, 'linewidth': 3},
                        color=[color] 
                        )

        p.set(title='Density Estimation on Kaist Validation Dataset')
        p.set_xlabel("Pixel Intensity", fontsize=10)
        p.set_ylabel("Density", fontsize=10)

        legend_params_labels = ['Red Channel (RGB)', 
                        'Green Channel (RGB)', 
                        'Blue Channel (RGB)', 
                        'Infrared (IR)',]
        
        p.legend(loc='center left', labels=legend_params_labels, bbox_to_anchor=(1.01, 1),
                 borderaxespad=0)

        
        # cont > 1, skip first validation of pytorch lightning
        if(clear == True):
            if(cont > 1):
                plt.savefig('distplot_' + str(cont) + '_' + str(color) , dpi=300, bbox_inches='tight')
                plt.savefig('distplot_' + str(cont) + '_' + str(color) + '.pdf', dpi=300, bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()


    @staticmethod
    def convert_bbox_xyxy_xywh(bboxes=None):
        # Convert x1,y1,x2,y2 -> x1,y1,w,h
        bboxes = bboxes.clone()
        if bboxes is None:
            return []
        elif(len(bboxes.shape) <= 1):
            bboxes = bboxes.unsqueeze(0)

        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        return bboxes


    @staticmethod
    def convert_bbox_xywh_xyxy(bboxes=None):
        # Convert x1,y1,w,h -> x1,y1,x2,y2
        bboxes = bboxes.clone()
        
        if bboxes is None:
            return []
        elif(len(bboxes.shape) <= 1):
            bboxes = bboxes.unsqueeze(0)

        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        return bboxes


    @staticmethod
    def normalize_bboxes(bboxes=None, w=640, h=512):
        # Convert  x1,y1,x2,y2
        bboxes = bboxes.clone()
        
        if bboxes is None:
            return []
        elif(len(bboxes.shape) <= 1):
            bboxes = bboxes.unsqueeze(0)

        bboxes[:, 0] /= w
        bboxes[:, 1] /= h
        bboxes[:, 2] /= w
        bboxes[:, 3] /= h

        return bboxes


    @staticmethod
    def unnormalize_bboxes(bboxes=None, w=640, h=512):
        # Convert  x1,y1,x2,y2
        bboxes = bboxes.clone()
        
        if bboxes is None:
            return []
        elif(len(bboxes.shape) <= 1):
            bboxes = bboxes.unsqueeze(0)

        bboxes[:, 0] *= w
        bboxes[:, 1] *= h
        bboxes[:, 2] *= w
        bboxes[:, 3] *= h

        return bboxes


    @staticmethod
    def show_bbox(img, bboxes, color=None, label=None, score=None, position=None, decode_dict={'0' : 'Person'}, format='xyxy'): 
        assert position in [
            None,
            "top",
            "left",
        ], "Position must be either top or left"

        if format == 'xywh':
            bboxes = Utils()._convert_bbox_xywh_xyxy(bboxes)

        if len(bboxes.shape) > 2:
            bboxes.squeeze_(0)

        for index, bbox in enumerate(bboxes):
            
            tl = int(round(0.001 * max(img.shape[0:2])))  # line thickness
            tl = tl + 1 if tl == 0 else tl
            color = color or (0, 122, 204)
                
            c1, c2 = (int(bbox[0]), int(bbox[1])), (
                int(bbox[2]),
                int(bbox[3]),
            )

            cv2.rectangle(img, c1, c2, color, thickness=tl)

            if label: #and bbox[-1] >= 0:
                tlabel = label #str(int(bbox[-1])) if self.labels is None else label #decode_dict[str(int(bbox[-1]))]
                if isinstance(label, list):
                    tlabel = label[index]
                if score is not None:
                    if type(score) is np.ndarray:
                        score = score.tolist()
                    if isinstance(score, list):
                        tlabel = "{}: {:1.3f}".format(tlabel, score[index])
                    else:
                        tlabel = "{}: {:1.3f}".format(tlabel, score)

                tf = max(tl - 2, 1)  # font thickness
                t_size = cv2.getTextSize(
                    tlabel, 0, fontScale=float(tl) / 3, thickness=tf
                )[0]

                offset = 0
                if position == "left":
                    img = np.ascontiguousarray(np.rot90(img))
                    s_size = img.shape[:2]
                    c1 = (c1[1], s_size[0] - c1[0])
                    c2 = (c2[1], s_size[0] - c2[0])
                    offset = 10

                c2 = (
                    c1[0] + t_size[0] + 15,
                    c1[1] - t_size[1] - 3 + offset,
                )
                c1 = (c1[0], c1[1] + offset)
                cv2.rectangle(img, c1, c2, color, -1)  # filled
                cv2.putText(
                    img,
                    "{}".format(tlabel),
                    (c1[0], c1[1] - 2),
                    0,
                    float(tl) / 3,
                    [0, 0, 0],
                    thickness=tf,
                    lineType=cv2.FONT_HERSHEY_SIMPLEX,
                )

                if position == "left":
                    img = np.ascontiguousarray(np.rot90(img, -1))

        return img
    

    @staticmethod
    def open_txt_file(file_name, path_images):

        with open(file_name, 'r') as f:
            paths = f.readlines()

        imgs_path = [os.path.join(path_images, x.strip()) for x in paths]
        
        return imgs_path


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


    @staticmethod
    def split_dataset(train_dataset, split_ratio=0.8, seed=123):
        train_size = int(split_ratio * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, 
                                                                    [train_size, valid_size], 
                                                                    generator=torch.Generator().manual_seed(seed))
        return train_dataset, valid_dataset

    @staticmethod
    def normalize_image(image):

        mins = torch.as_tensor([image[idx].min() for idx in range(3)])
        maxs = torch.as_tensor([image[idx].max() for idx in range(3)])
  
        for idx in range(0, 3):
            if (maxs[idx]-mins[idx] != 0.0):
                image[idx] = (image[idx] - mins[idx]) / (maxs[idx]-mins[idx])
            else:    
                image[idx] = 0.0

        return image
    
    @staticmethod # Not vectorized
    def normalize_batch_images(images):
        for idx, img in enumerate(images):
            images[idx] = Utils.normalize_image(img)
        return images    

    @staticmethod
    def plot_each_image(image, output, target, threshold=0.5):

        image = image.cpu().detach()

        boxes_th = output['boxes'][output['scores'] > threshold]
        labels_th = output['labels'][output['scores'] > threshold]

        image = Utils.normalize_image(image)

        image = (image.numpy().transpose(1,2,0) * 255).astype("uint8").copy()

        targets_plt = target['boxes'].cpu().numpy()

        decode_dict = {0:'background', 1: 'person', 2: 'bicycle', 3: 'car'}

        image = Utils().show_bbox(
                        image,
                        targets_plt,
                        color=(255, 255, 0),
                        label="ground_truth",
                        position="left",
                    )
        
        if((len(boxes_th) > 0)):
            
            for annot_squeeze in labels_th:

                image = Utils().show_bbox(
                image, boxes_th.type(torch.int), color=(255,0,0), 
                    label=decode_dict[int(annot_squeeze.cpu().numpy())], # ('person'), 
                    position="top", 
                )
                
        ## Unnormalize # Dont need to undo because wandb is going to plot it like image (between 0-255)
        # for idx in range(0, 3):
        #     if (maxs[idx]-mins[idx] != 0.0):
        #         image[idx] = image[idx]*(maxs[idx].item()-mins[idx].item()) + mins[idx].item()
        #     else:    
        #         image[idx] = 0.0

        return image.transpose(2, 0, 1) / 255.0

    @staticmethod
    def reduce_dict(input_dict, average=True):
        """
        Args:
            input_dict (dict): all the values will be reduced
            average (bool): whether to do average or sum
        Reduce the values in the dictionary from all processes so that all processes
        have the averaged results. Returns a dict with the same fields as
        input_dict, after reduction.
        """
        with torch.no_grad():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    @staticmethod
    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


    @staticmethod
    def filter_dictionary(input_dict, filter_keys):
        output_dict = {}
        for key, value in input_dict.items():
            if key in filter_keys:
                output_dict[key] = value
        return output_dict
        

    @staticmethod
    def get_bbox(filename, dataset='kaist', train=False):
        """
        Utility function for getting the bbox given a ``xml`` annotation file.
        The function returns a list of bounding boxes each of them with the format
        ``[xmin, ymin, xmax, ymax]`` where ``(xmin, ymin)`` corresponds with the top
        left point and ``(xmax, ymax)`` is the bottom right point.
        """
        # parsing xml annotation file
        if dataset == 'kaist':
            print("") # Need to implement soon
        elif dataset == 'llvip':
            filename = os.path.join(filename[:filename.index('LLVIP')], 'LLVIP' , 'Annotations', filename.split('/')[-1]) 

        elif dataset == 'flir':
            filename = os.path.join(filename.split('/JPEGImages/')[0], 'Annotations', 
                            filename.split('/JPEGImages/')[-1]).replace('RGB', 'PreviewData')
        else:
            raise Exception("Dataset not supported")
        
        root = ET.parse(filename).getroot()

        # bounding boxes list
        bboxes = list()
        labels = list()
        parser_list = ["x", "y", "w", "h"] if dataset == 'kaist' else (
                    ["xmin", "ymin", "xmax", "ymax"] # for llvip and flir
        )

        # for each object within the annotation file
        for obj in root.findall("object"):
            # find the tag bndbox
            bbox_obj = obj.find("bndbox")

            # convert x,y,w,h to xmin,ymin,xmax,ymax
            # convert it to a list of int in the format [xmin, ymin, xmax, ymax]
            bbox = [
                int(val)
                for val in [
                    bbox_obj.find(parser_list[0]).text,
                    bbox_obj.find(parser_list[1]).text,
                    bbox_obj.find(parser_list[2]).text,
                    bbox_obj.find(parser_list[3]).text,
                ]
            ] 
            
            if(dataset == 'kaist'):
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

            # Just to make sure that the min is min and max is max
            xmin = min(bbox[0], bbox[2])
            ymin = min(bbox[1], bbox[3])
            xmax = max(bbox[0], bbox[2])
            ymax = max(bbox[1], bbox[3])

            bbox = [xmin, ymin, xmax, ymax]
            
            # person, people, cyclist, or person? (Kaist)
            # select only person label (others are ignored)
            # ignore bbox that are too small (there are problems with bbox zero area and some of the albumentations transf.)

            if(dataset == 'flir'):
                if(train and abs(bbox[2]-bbox[0])*abs(bbox[3]-bbox[1]) > 10.0):
                    if(obj.find("name").text == "person"):
                        bboxes.append(bbox)
                        labels.append([1])

                    # if(obj.find("name").text == "bicycle"):
                    #     bboxes.append(bbox)
                    #     labels.append([2])

                    # if(obj.find("name").text == "car"):
                    #     bboxes.append(bbox)
                    #     labels.append([3])

                # https://github.com/mrkieumy/YOLOv3_PyTorch/blob/45230ba75014b8eb77dc1b7f2b8dd9b71cb9af56/LAMR_AP.py#L72
                # These seems to came from other evaluation code on this dataset, so we kept the same for comparison
                elif(not train and abs(ymax-ymin) > 50.0): # Filter like the above link, get bbox greater than 50 height for test
                    if(obj.find("name").text == "person"):
                        bboxes.append(bbox)
                        labels.append([1])

                    # if(obj.find("name").text == "bicycle"):
                    #     bboxes.append(bbox)
                    #     labels.append([2])

                    # if(obj.find("name").text == "car"):
                    #     bboxes.append(bbox)
                    #     labels.append([3])
                

            elif(abs(bbox[2]-bbox[0])*abs(bbox[3]-bbox[1]) > 5.0):
                if(obj.find("name").text == "person"):
                    bboxes.append(bbox)
                    labels.append([1])

        return {"bboxes" : np.array(bboxes).astype("float"), "labels" : np.array(labels).astype("int")}

    @staticmethod
    def fusion_data(imgs_hallucinated, img_ir, fuse_data="none"):

        if(fuse_data == "cross"):
            imgs_hallucinated = torch.clip(img_ir * imgs_hallucinated, min=0.0, max=1.0)
        elif(fuse_data == "addition"):
            imgs_hallucinated = addition_fusion(img_ir, imgs_hallucinated)
        elif(fuse_data == "attention"):
            imgs_hallucinated = attention_fusion_weight(img_ir, imgs_hallucinated)

        return imgs_hallucinated



if __name__ == "__main__":

    print("debug")
