import torch

model = torch.load('saved_models/DHTR0401/DHTR_29_loss_0.69.pth')
print(model.row_nums)
print(model.col_nums)
print(model.cell_nums)

torch.save(model.state_dict(), 'selected_model/DHTR0426_loss_0.69.pth')
