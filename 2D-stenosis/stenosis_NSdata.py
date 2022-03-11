import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
import vtk
from vtk.util import numpy_support as VN
import random

# torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


batchsize = 256
epochs = 5500
# epochs = 1
learning_rate = 5e-4 #starting learning rate
step_epoch = 1200 #1000
decay_rate = 0.1 # 0.1
n_hid = [2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3]
device = torch.device("cuda")

Lambda_BC = 20.
Lambda_data = 1.

path = "Results/"
Directory = "/home/hwlui/development/PINN-wss/Data/2D-stenosis/"
mesh_file = Directory + "sten_mesh000000.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"
File_data = Directory + "velocity_sten_steady.vtu"
fieldname = 'f_5-0' #The velocity field name in the vtk file (see from ParaView)

Diff = 0.001
rho = 1.
X_scale = 2.0 #The length of the  domain (need longer length for separation region)
Y_scale = 1.0
U_scale = 1.0


# MSE LOSS
mse = nn.MSELoss()


def load_data():
    print('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of the mesh:', n_points)
    xy_f = np.zeros((n_points, 2))
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        xy_f[i, 0] = pt_iso[0]
        xy_f[i, 1] = pt_iso[1]

    print('Loading', bc_file_wall)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(bc_file_wall)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_pointsw = data_vtk.GetNumberOfPoints()
    print('n_points of at wall', n_pointsw)
    xy_b = np.zeros((n_pointsw, 2))
    for i in range(n_pointsw):
        pt_iso = data_vtk.GetPoint(i)
        xy_b[i, 0] = pt_iso[0]
        xy_b[i, 1] = pt_iso[1]

    ##### Read ground truth data here#########
    print('Loading', File_data)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(File_data)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of the data file read:', n_points)
    xy_val = np.zeros((n_points, 2))
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        xy_val[i, 0] = pt_iso[0] / X_scale
        xy_val[i, 1] = pt_iso[1] / Y_scale
    all_array = data_vtk.GetPointData().GetArray(fieldname)
    uv_sol = VN.vtk_to_numpy(all_array) / U_scale

    # !!specify pts location here:
    xd = np.asarray([1., 1.2, 1.22, 1.31, 1.39])
    yd = np.asarray([0.15, 0.07, 0.22, 0.036, 0.26])
    zd = np.asarray([0., 0., 0., 0., 0.])
    VTKpoints = vtk.vtkPoints()
    for i in range(len(xd)):
        VTKpoints.InsertPoint(i, xd[i], yd[i], zd[i])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname)
    uv_d = (VN.vtk_to_numpy(array) / U_scale)
    uv_d = uv_d[:, :2]
    xd = xd / X_scale
    yd = yd / Y_scale
    xy_d = np.stack([xd, yd], 1)
    return xy_f, xy_b, xy_d, uv_d, xy_val, uv_sol


def plot_results(xy_val, uv_sol, uv_pred):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].scatter(xy_val[:, 0], xy_val[:, 1], c=uv_sol[:, 0], cmap='rainbow')
    axs[0, 0].set_title('GT U')

    axs[0, 1].scatter(xy_val[:, 0], xy_val[:, 1], c=uv_pred[:, 0], cmap='rainbow')
    axs[0, 1].set_title('Predict U')

    axs[1, 0].scatter(xy_val[:, 0], xy_val[:, 1], c=uv_sol[:, 1], cmap='rainbow')
    axs[1, 0].set_title('GT U')

    axs[1, 1].scatter(xy_val[:, 0], xy_val[:, 1], c=uv_pred[:, 1], cmap='rainbow')
    axs[1, 1].set_title('Predict U')

    plt.savefig('debug_main_3_nets.png', dpi=500)
    plt.clf()
    plt.close()


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


def create_unified_model():
    layers = []
    for i in range(len(n_hid) - 1):
        if i > 0:
            layers.append(Swish())
        layer = nn.Linear(n_hid[i], n_hid[i + 1])
        nn.init.kaiming_normal_(layer.weight)
        layers.append(layer)
    pinn = nn.Sequential(*layers)
    pinn.to(device)
    return pinn


def create_model(n_hid):
    class Net(nn.Module):

        # The __init__ function stack the layers of the
        # network Sequentially
        def __init__(self):
            super().__init__()
            layers = []
            for i in range(len(n_hid) - 1):
                if i > 0:
                    layers.append(Swish())
                layer = nn.Linear(n_hid[i], n_hid[i + 1])
                nn.init.kaiming_normal_(layer.weight)
                layers.append(layer)

            self.main = nn.Sequential(*layers)

        # This function defines the forward rule of
        # output respect to input.
        def forward(self, x):
            output = self.main(x)
            return output
    return Net()


# def residual(pinn, xy):
#     x = xy[:, [0]]
#     y = xy[:, [1]]
#     x.requires_grad = True
#     y.requires_grad = True
#     outputs = pinn(torch.cat([x, y], 1))
#     u = outputs[:, [0]]
#     v = outputs[:, [1]]
#     p = outputs[:, [2]]
#
#     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#     u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#     u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
#     u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
#     v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#     v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#     v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
#     v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
#     p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#     p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
#
#     XX_scale = U_scale * (X_scale**2)
#     YY_scale = U_scale * (Y_scale**2)
#     UU_scale = U_scale ** 2
#
#     loss_1 = u*v_x / X_scale + v*v_y / Y_scale - Diff*(v_xx/XX_scale + v_yy / YY_scale) + 1/rho * (p_y / (Y_scale * UU_scale)) #Y-dir
#     loss_2 = u*u_x / X_scale + v*u_y / Y_scale - Diff*(u_xx/XX_scale + u_yy / YY_scale) + 1/rho * (p_x / (X_scale * UU_scale))  #X-dir
#     loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity
#     return (loss_1 ** 2).mean() + (loss_2 ** 2).mean() + (loss_3 ** 2).mean()


def residual(net_u, net_v, net_p, xy):
    x = xy[:, [0]]
    y = xy[:, [1]]
    x.requires_grad = True
    y.requires_grad = True

    x.requires_grad = True
    y.requires_grad = True
    net_in = torch.cat((x, y), 1)
    u = net_u(net_in)
    u = u.view(len(u), -1)
    v = net_v(net_in)
    v = v.view(len(v), -1)
    p = net_p(net_in)
    p = p.view(len(p), -1)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    XX_scale = U_scale * (X_scale**2)
    YY_scale = U_scale * (Y_scale**2)
    UU_scale = U_scale ** 2

    loss_1 = u*v_x / X_scale + v*v_y / Y_scale - Diff*(v_xx/XX_scale + v_yy / YY_scale) + 1/rho * (p_y / (Y_scale * UU_scale)) #Y-dir
    loss_2 = u*u_x / X_scale + v*u_y / Y_scale - Diff*(u_xx/XX_scale + u_yy / YY_scale) + 1/rho * (p_x / (X_scale * UU_scale))  #X-dir
    loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity
    return (loss_1 ** 2).mean() + (loss_2 ** 2).mean() + (loss_3 ** 2).mean()


# def boundary(pinn, xy_b):
#     outputs = pinn(xy_b)
#     u = outputs[:, [0]]
#     v = outputs[:, [1]]
#     return (u**2).mean() + (v**2).mean()


def boundary(net_u, net_v, xy_b):
    u = net_u(xy_b)
    v = net_v(xy_b)
    return (u**2).mean() + (v**2).mean()


# def regression(pinn, xy_d, uv_d):
#     outputs = pinn(xy_d)
#     u = outputs[:, [0]]
#     v = outputs[:, [1]]
#     ud = uv_d[:, [0]]
#     vd = uv_d[:, [0]]
#     return mse(u, ud) + mse(v, vd)


def regression(net_u, net_v, xy_d, uv_d):
    u = net_u(xy_d)
    v = net_v(xy_d)
    ud = uv_d[:, [0]]
    vd = uv_d[:, [1]]
    return mse(u, ud) + mse(v, vd)


def geo_train():
    xy_f, xy_b, xy_d, uv_d, xy_val, uv_sol = load_data()

    xy_f = torch.Tensor(xy_f).to(device)
    xy_b = torch.Tensor(xy_b).to(device)
    xy_d = torch.Tensor(xy_d).to(device)
    uv_d = torch.Tensor(uv_d).to(device)
    xy_val = torch.Tensor(xy_val).to(device)
    uv_sol = torch.Tensor(uv_sol).to(device)

    dataset = TensorDataset(xy_f)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)
    # pinn = create_unified_model().to(device)
    # opt = optim.Adam(pinn.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_epoch, gamma=decay_rate)

    net_u = create_model([2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]).to(device)
    net_v = create_model([2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]).to(device)
    net_p = create_model([2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]).to(device)

    net_u.load_state_dict(torch.load(path + "sten_data_u" + ".pt"))
    net_v.load_state_dict(torch.load(path + "sten_data_v" + ".pt"))
    net_p.load_state_dict(torch.load(path + "sten_data_p" + ".pt"))

    ############################################################################
    optimizer_u = optim.Adam(net_u.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
    optimizer_v = optim.Adam(net_v.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
    optimizer_p = optim.Adam(net_p.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
    scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

    # Main loop
    tic = time.time()
    for epoch in range(epochs):
        residual_loss_tot = 0.
        mse_b_tot = 0.
        mse_0_tot = 0.
        for batch_idx, (x_y_batch, ) in enumerate(dataloader):
            # pinn.zero_grad()
            net_u.zero_grad()
            net_v.zero_grad()
            net_p.zero_grad()
            # residual_loss = residual(pinn, x_y_batch)
            # mse_b = boundary(pinn, xy_b)
            # mse_0 = regression(pinn, xy_d, uv_d)
            residual_loss = residual(net_u, net_v, net_p, x_y_batch)
            mse_b = boundary(net_u, net_v, xy_b)
            mse_0 = regression(net_u, net_v, xy_d, uv_d)
            loss = residual_loss + Lambda_BC * mse_b + Lambda_data*mse_0
            loss.backward()
            # opt.step()
            optimizer_u.step()
            optimizer_v.step()
            optimizer_p.step()
            residual_loss_tot += residual_loss
            mse_b_tot += mse_b
            mse_0_tot += mse_0
            if batch_idx % 40 == 0:
                print('Train Epoch: %i [%i/%i (%.0f %%)] - residual_loss %.10f mse_b %.8f mse_0 %.8f' %
                      (epoch, batch_idx * len(x_y_batch), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), residual_loss.item(), mse_b.item(), mse_0.item()))
        # scheduler.step()
        scheduler_u.step()
        scheduler_v.step()
        scheduler_p.step()
        residual_loss_tot /= len(dataloader)
        mse_b_tot /= len(dataloader)
        mse_0_tot /= len(dataloader)
        print('*****Total avg Loss : residual_loss %.10f mse_b %.8f mse_0 %.8f ****' %
              (residual_loss_tot, mse_b_tot, mse_0_tot))
        print('learning rate is: %.8f' % optimizer_u.param_groups[0]['lr'])

    print("elapse time in parallel = %i" % (time.time() - tic))
    # uv_pred = pinn(xy_val)  #evaluate model (runs out of memory for large GPU problems!)
    u_pred = net_u(xy_val)
    v_pred = net_v(xy_val)
    uv_pred = torch.cat([u_pred, v_pred], 1)
    val_loss_u = mse(uv_pred[:, 0], uv_sol[:, 0])
    val_loss_v = mse(uv_pred[:, 1], uv_sol[:, 1])
    print('*****Validation loss: val_loss_u %.8f val_loss_v %.8f*****' %
          (val_loss_u.item(), val_loss_v.item()))
    plot_results(xy_val.detach().cpu().numpy(), uv_sol.detach().cpu().numpy(), uv_pred.detach().cpu().numpy())
    return


if __name__ == "__main__":
    geo_train()
