import torch.nn as nn
import torchvision.models as models

def create_pretrained_model(model_name='resnet50', num_classes=10, pretrained=True, freeze_layers=False, freeze_strategy='last_layer'):
    # Load the pre-trained model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=pretrained)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(pretrained=pretrained)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    # Modify the final classification layer
    if model_name == 'inception_v3':
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, num_classes)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'resnet' in model_name or 'vgg' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'efficientnet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif 'vit' in model_name:
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    elif 'googlenet' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Freeze layers based on the specified strategy
    if freeze_layers:
        if freeze_strategy == 'all_except_last':
            if model_name == 'inception_v3':
                for name, param in model.named_parameters():
                    if 'AuxLogits.fc' not in name and 'fc' not in name:
                        param.requires_grad = False
            elif 'resnet' in model_name or 'vgg' in model_name:
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
            elif 'efficientnet' in model_name:
                for name, param in model.named_parameters():
                    if 'classifier.1' not in name:
                        param.requires_grad = False
            elif 'vit' in model_name:
                for name, param in model.named_parameters():
                    if 'heads.head' not in name:
                        param.requires_grad = False
            elif 'googlenet' in model_name:
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
        elif freeze_strategy == 'first_k':
            k = 5  # Example value, adjust as needed
            for i, (name, param) in enumerate(model.named_parameters()):
                if i < k:
                    param.requires_grad = False
        else:
            raise ValueError(f'Unsupported freeze strategy: {freeze_strategy}')

    return model
