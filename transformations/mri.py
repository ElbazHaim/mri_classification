from torchvision import transforms

mri_jpg_preprocessing = transforms.Compose(
    [
        transforms.Resize((208, 176)),
        transforms.CenterCrop((188, 156)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)
