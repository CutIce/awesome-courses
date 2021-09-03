import torch

device_num = torch.cuda.device_count()
print(device_num)
infos = [torch.cuda.get_device_properties(i) for i in range(device_num)]

print(infos)