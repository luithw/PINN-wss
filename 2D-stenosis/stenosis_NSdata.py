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


device = torch.device("cuda")
Lambda_BC = 20.
Lambda_data = 1.

Directory = "/home/hwlui/development/PINN-wss/Data/2D-stenosis/"
mesh_file = Directory + "sten_mesh000000.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"
File_data = Directory + "velocity_sten_steady.vtu"
fieldname = 'f_5-0' #The velocity field name in the vtk file (see from ParaView)

batchsize = 256
# epochs = 5500
epochs = 20
Diff = 0.001
rho = 1.
T = 0.5 #total duraction
Flag_x_length = True #if True scales the eqn such that the length of the domain is = X_scale
X_scale = 2.0 #The length of the  domain (need longer length for separation region)
Y_scale = 1.0
U_scale = 1.0
U_BC_in = 0.5
Lambda_div = 1.  #penalty factor for continuity eqn (Makes it worse!?)
Lambda_v = 1.  #penalty factor for y-momentum equation
nPt = 130
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0
delta_circ = 0.2
h_n = 128  # Width for u,v,p
input_n = 2  # this is what our answer is a function of (x,y)
n_hid = [input_n, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3]
learning_rate = 5e-4 #starting learning rate
step_epoch = 1200 #1000
decay_rate = 0.1 # 0.1
path = "Results/"

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
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))

    t = np.linspace(0., T, nPt * nPt)
    t = t.reshape(-1, 1)
    print('shape of x', x.shape)
    print('shape of y', y.shape)
    # print('shape of t',t.shape)

    ## Define boundary points
    print('Loading', bc_file_in)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(bc_file_in)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of at inlet', n_points)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    print('Loading', bc_file_wall)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(bc_file_wall)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_pointsw = data_vtk.GetNumberOfPoints()
    print('n_points of at wall', n_pointsw)
    x_vtk_mesh = np.zeros((n_pointsw, 1))
    y_vtk_mesh = np.zeros((n_pointsw, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_pointsw):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    xb_wall = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
    yb_wall = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))
    u_wall_BC = np.linspace(0., 0., n_pointsw)
    v_wall_BC = np.linspace(0., 0., n_pointsw)
    xb = xb_wall
    yb = yb_wall
    ub = u_wall_BC
    vb = v_wall_BC

    xb = xb.reshape(-1, 1)  # need to reshape to get 2D array
    yb = yb.reshape(-1, 1)  # need to reshape to get 2D array
    ub = ub.reshape(-1, 1)  # need to reshape to get 2D array
    vb = vb.reshape(-1, 1)  # need to reshape to get 2D array
    print('shape of xb', xb.shape)
    print('shape of yb', yb.shape)
    print('shape of ub', ub.shape)

    ##### Read data here#########
    # !!specify pts location here:
    x_data = [1., 1.2, 1.22, 1.31, 1.39]
    y_data = [0.15, 0.07, 0.22, 0.036, 0.26]
    z_data = [0., 0., 0., 0., 0.]

    x_data = np.asarray(x_data)  # convert to numpy
    y_data = np.asarray(y_data)  # convert to numpy

    print('Loading', File_data)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(File_data)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of the data file read:', n_points)

    VTKpoints = vtk.vtkPoints()
    for i in range(len(x_data)):
        VTKpoints.InsertPoint(i, x_data[i], y_data[i], z_data[i])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    probe = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname)
    data_vel = VN.vtk_to_numpy(array)

    data_vel_u = data_vel[:, 0] / U_scale
    data_vel_v = data_vel[:, 1] / U_scale
    x_data = x_data / X_scale
    y_data = y_data / Y_scale

    print('Using input data pts: pts: ', x_data, y_data)
    print('Using input data pts: vel u: ', data_vel_u)
    print('Using input data pts: vel v: ', data_vel_v)
    xd = x_data.reshape(-1, 1)  # need to reshape to get 2D array
    yd = y_data.reshape(-1, 1)  # need to reshape to get 2D array
    ud = data_vel_u.reshape(-1, 1)  # need to reshape to get 2D array
    vd = data_vel_v.reshape(-1, 1)  # need to reshape to get 2D array
    return x, y, xb, yb, ub, vb, xd, yd, ud, vd


def plot_results(x, y, output_u, output_v):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, c=output_u, cmap='rainbow')
    plt.title('NN results, u')
    plt.colorbar()
    plt.savefig('results_u.png', dpi=500)
    plt.clf()
    plt.close()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, c=output_v, cmap='rainbow')
    plt.title('NN results, v')
    plt.colorbar()
    plt.savefig('results_v.png', dpi=500)
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


def create_model():
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


def residual(pinn, x, y):
    x.requires_grad = True
    y.requires_grad = True
    outputs = pinn(torch.cat((x, y), 1))
    u = outputs[:, [0]]
    v = outputs[:, [1]]
    p = outputs[:, [2]]

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


def boundary(pinn, xb, yb):
    outputs = pinn(torch.cat((xb, yb), 1))
    u = outputs[:, [0]]
    v = outputs[:, [1]]
    return (u**2).mean() + (v**2).mean()


def regression(pinn, xd, yd, ud, vd):
    outputs = pinn(torch.cat((xd, yd), 1))
    u = outputs[:, [0]]
    v = outputs[:, [1]]
    return mse(u, ud) + mse(v, vd)

def geo_train():
    x_in, y_in, xb, yb, ub, vb, xd, yd, ud, vd = load_data()
    x = torch.Tensor(x_in).to(device)
    y = torch.Tensor(y_in).to(device)
    xb = torch.Tensor(xb).to(device)
    yb = torch.Tensor(yb).to(device)
    xd = torch.Tensor(xd).to(device)
    yd = torch.Tensor(yd).to(device)
    ud = torch.Tensor(ud).to(device)
    vd = torch.Tensor(vd).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)
    pinn = create_model().to(device)
    opt = optim.Adam(pinn.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=10**-15)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_epoch, gamma=decay_rate)

    # Main loop
    tic = time.time()
    for epoch in range(epochs):
        residual_loss_tot = 0.
        mse_b_tot = 0.
        mse_0_tot = 0.
        n = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            pinn.zero_grad()
            residual_loss = residual(pinn, x_batch, y_batch)
            mse_b = boundary(pinn, xb, yb)
            mse_0 = regression(pinn, xd, yd, ud, vd)
            loss = residual_loss + Lambda_BC * mse_b + Lambda_data*mse_0
            loss.backward()
            opt.step()
            residual_loss_tot += residual_loss
            mse_b_tot += mse_b
            mse_0_tot += mse_0
            n += 1
            if batch_idx % 40 == 0:
                print('Train Epoch: %i [%i/%i (%.0f %%)] - residual_loss %.10f mse_b %.8f mse_0 %.8f' %
                      (epoch, batch_idx * len(x_batch), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), residual_loss.item(), mse_b.item(), mse_0.item()))
            scheduler.step()
        residual_loss_tot /= n
        mse_b_tot /= n
        mse_0_tot /= n
        print('*****Total avg Loss : residual_loss %.10f mse_b %.8f mse_0 %.8f ****' %
              (residual_loss_tot, mse_b_tot, mse_0_tot))
        print('learning rate is: %.8f' % opt.param_groups[0]['lr'])

    print("elapse time in parallel = %i" % (time.time() - tic))
    net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    outputs = pinn(net_in).cpu().data.numpy()  #evaluate model (runs out of memory for large GPU problems!)
    u = outputs[:, [0]]
    v = outputs[:, [1]]
    plot_results(x_in, y_in, u, v)
    return


if __name__ == "__main__":
    geo_train()
