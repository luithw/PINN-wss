import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
import vtk
from vtk.util import numpy_support as VN


device = torch.device("cuda")
Flag_batch = True
Flag_BC_exact = False
Lambda_BC  = 20.
Lambda_data = 1.

#Directory = "/home/aa3878/Data/ML/Amir/stenosis/"
Directory = "/home/hwlui/development/PINN-wss/Data/2D-stenosis/"
mesh_file = Directory + "sten_mesh000000.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"
File_data = Directory + "velocity_sten_steady.vtu"
fieldname = 'f_5-0' #The velocity field name in the vtk file (see from ParaView)

batchsize = 256
epochs  = 5500
Diff = 0.001
rho = 1.
T = 0.5 #total duraction
#nPt_time = 50 #number of time-steps

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
n_hid = [input_n, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]
learning_rate = 5e-4 #starting learning rate
step_epoch = 1200 #1000
decay_rate = 0.1 # 0.1
path = "Results/"
N_Layers = 9

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
    xb_in = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))
    yb_in = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))

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
    plt.scatter(x.detach().numpy(), y.detach().numpy(), c=output_u, cmap='rainbow')
    plt.title('NN results, u')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(x.detach().numpy(), y.detach().numpy(), c=output_v, cmap='rainbow')
    plt.title('NN results, v')
    plt.colorbar()
    plt.show()

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
    net_u = create_model().to(device)
    net_v = create_model().to(device)
    net_p = create_model().to(device)

    def criterion(x, y):
        x.requires_grad = True
        y.requires_grad = True
        net_in = torch.cat((x, y), 1)
        u = net_u(net_in)
        u = u.view(len(u), -1)
        v = net_v(net_in)
        v = v.view(len(v), -1)
        P = net_p(net_in)
        P = P.view(len(P), -1)

        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

        P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

        XX_scale = U_scale * (X_scale**2)
        YY_scale = U_scale * (Y_scale**2)
        UU_scale  = U_scale ** 2

        loss_2 = u*u_x / X_scale + v*u_y / Y_scale - Diff*( u_xx/XX_scale  + u_yy /YY_scale  )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
        loss_1 = u*v_x / X_scale + v*v_y / Y_scale - Diff*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
        loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity

        # MSE LOSS
        loss_f = nn.MSELoss()

        #Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))
        return loss

    def Loss_BC(xb, yb):
        net_in1 = torch.cat((xb, yb), 1)
        out1_u = net_u(net_in1)
        out1_v = net_v(net_in1)
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)
        loss_f = nn.MSELoss()
        loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v))
        return loss_noslip

    def Loss_data(xd, yd, ud, vd):
        net_in1 = torch.cat((xd, yd), 1)
        out1_u = net_u(net_in1)
        out1_v = net_v(net_in1)
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)
        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd)
        return loss_d

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
        loss_eqn_tot = 0.
        loss_bc_tot = 0.
        loss_data_tot = 0.
        n = 0
        for batch_idx, (x_in,y_in) in enumerate(dataloader):
            net_u.zero_grad()
            net_v.zero_grad()
            net_p.zero_grad()
            loss_eqn = criterion(x_in, y_in)
            loss_bc = Loss_BC(xb, yb)
            loss_data = Loss_data(xd, yd, ud, vd)
            loss = loss_eqn + Lambda_BC * loss_bc + Lambda_data*loss_data
            loss.backward()
            optimizer_u.step()
            optimizer_v.step()
            optimizer_p.step()
            loss_eqn_tot += loss_eqn
            loss_bc_tot += loss_bc
            loss_data_tot += loss_data
            n += 1
            if batch_idx % 40 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
                    epoch, batch_idx * len(x_in), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(),loss_data.item()))
            scheduler_u.step()
            scheduler_v.step()
            scheduler_p.step()
        loss_eqn_tot /= n
        loss_bc_tot /= n
        loss_data_tot /= n
        print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot,loss_data_tot) )
        print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])

    print("elapse time in parallel = ", time.time() - tic)
    net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    output_u = net_u(net_in).cpu().data.numpy()  #evaluate model (runs out of memory for large GPU problems!)
    output_v = net_v(net_in).cpu().data.numpy()  #evaluate model
    plot_results(x_in, y_in, output_u, output_v)
    return


if __name__ == "__main__":
    geo_train()
