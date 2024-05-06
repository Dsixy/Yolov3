import torch
import torch.nn as nn

anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]


class Decoder():
    def __init__(self, anchors, input_size):
        self.anchors = anchors
        self.input_size = input_size

    def decode(self, x):
        output = []
        for i, input in enumerate(x):
            input_size = input.size()
            batch_size = input_size[0]
            input_width = input_size[2]
            input_height = input_size[3]

            scale_factor = self.input_size / input_width
            prediction = input.view(batch_size, 3, -1, input_width, input_width).permute(0, 1, 3, 4, 2).contiguous()

            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])
            class_prob = torch.sigmoid(prediction[..., 5:])

            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * 3, 1, 1).view(x.shape).type(torch.FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * 3, 1, 1).view(y.shape).type(torch.FloatTensor)

            scale_anchors = [[anchor_w // scale_factor, anchor_h // scale_factor]
                             for anchor_w, anchor_h in self.anchors[i]]

            anchor_w = torch.FloatTensor(scale_anchors).index_select(1, torch.LongTensor([0]))
            anchor_h = torch.FloatTensor(scale_anchors).index_select(1, torch.LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #  pred_box: (batch_size, 3, width, height, channel)
            #  pred_box: (4, 3, 13, 13, 25)
            pred_box = torch.FloatTensor(prediction[..., :4].size())
            pred_box[..., 0] = x.data + grid_x
            pred_box[..., 1] = y.data + grid_y
            pred_box[..., 2] = torch.exp(w) * anchor_w
            pred_box[..., 3] = torch.exp(h) * anchor_h

            pred_box = pred_box * scale_factor
            output.append(torch.cat((pred_box, conf.unsqueeze(-1), class_prob), dim=-1))
        return output


class YoloLoss(nn.Module):
    def __init__(self, box=1, obj=1, noobj=1, classes=1):
        super(YoloLoss, self).__init__()
        self.box_rate = box
        self.obj_rate = obj
        self.noobj_rate = noobj
        self.class_rate = classes
        self.decoder = Decoder(anchors, 416)

    def forward(self, batch_x, batch_y):
        loss = 0
        decoded_x = self.decoder.decode(batch_x)
        batch_size = batch_x[0].size(0)
        img_dim = batch_x[0].size(1) // 3
        width1, width2, width3 = batch_x[0].size(-1), batch_x[1].size(-1), batch_x[2].size(-1)

        # 将输入的[output1, output2, output3整合
        # 例如，将[batch_size, 3, w1, h1, 25], [batch_size, 3, w2, h2, 25], [batch_size, 3, w3, h3, 25]
        # 变成[batch_size, 3 * w1 * h1 + 3 * w2 * h2 + 3 * w3 * h3, 25]
        batch_x = torch.cat([batch_x[i].view(batch_size, 3, -1, width1, width1)
                            .permute(0, 1, 3, 4, 2)
                            .contiguous().view(batch_size, -1, img_dim)
                             for i in range(3)], dim=1)

        decoded_x = torch.cat([decoded_x[i].view(batch_size, -1, img_dim)
                               for i in range(3)], dim=1)

        for d_x, x, y in zip(decoded_x, batch_x, batch_y):
            for ground_truth in y:
                iou = self.calculate_batch_iou(d_x[..., :5], ground_truth)

                pos_values, pos_indices = torch.max(iou, dim=0)
                neg_indices = torch.where(iou < 0.5)[0]

                loss += self.box_rate * nn.MSELoss()(x[pos_indices, :4], ground_truth[:4])\
                        - self.obj_rate * torch.mean(torch.log(d_x[pos_indices, 4] + 1e-9))\
                        - self.noobj_rate * torch.mean(torch.log(1 - d_x[neg_indices, 4] + 1e-9))\
                        - self.class_rate * nn.CrossEntropyLoss()(d_x[pos_indices, 5:], ground_truth[4].long())
        return loss

    def calculate_batch_iou(self, pred_boxes, ground_truth):
        true_box = ground_truth.unsqueeze(0)
        true_boxes = true_box.expand_as(pred_boxes)

        x1_a, y1_a, w1_a, h1_a = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
        x1_b, y1_b, w2_b, h2_b = true_boxes[..., 0], true_boxes[..., 1], true_boxes[..., 2], true_boxes[..., 3]

        x_left = torch.maximum(x1_a, x1_b)
        y_top = torch.maximum(y1_a, y1_b)
        x_right = torch.minimum(x1_a + w1_a, x1_b + w2_b)
        y_bottom = torch.minimum(y1_a + h1_a, y1_b + h2_b)

        intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        box1_area = w1_a * h1_a
        box2_area = w2_b * h2_b

        iou = intersection_area / (box1_area + box2_area - intersection_area)
        return iou