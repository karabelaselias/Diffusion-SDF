import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch_scatter import scatter_mean, scatter_max, scatter_add

# Add this to ConvPointnet
@torch.jit.script
def fast_coordinate2index(x: torch.Tensor, reso: int) -> torch.Tensor:
    x = (x * reso).long()
    x = torch.clamp(x, 0, reso - 1)
    index = x[:, :, 0] + reso * x[:, :, 1]
    return index.unsqueeze(1)

class ConvPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=512, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=True, unet_kwargs={"depth": 4, "merge_mode": "concat", "start_filts": 32}, 
                 plane_resolution=64, plane_type=['xz', 'xy', 'yz'], padding=0.15, n_blocks=5,
                 inject_noise=False, use_dropout=True):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean

        self.use_dropout = use_dropout

    def generate_plane_features(self, p, c, plane='xz'):
        xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = self.coordinate2index(xy, self.reso_plane)
        
        batch_size = p.size(0)
        c = c.permute(0, 2, 1)

        # Get both mean and max features
        fea_plane_mean = scatter_mean(c, index, dim=2, dim_size=self.reso_plane**2)
        fea_plane_max, _ = scatter_max(c, index, dim=2, dim_size=self.reso_plane**2)

        # Count points per cell
        ones = torch.ones(batch_size, 1, c.shape[2], device=c.device)
        count = scatter_add(ones, index, dim=2, dim_size=self.reso_plane**2)

        # Reshape to 2D
        fea_plane_mean = fea_plane_mean.reshape(batch_size, self.c_dim, self.reso_plane, self.reso_plane)
        fea_plane_max = fea_plane_max.reshape(batch_size, self.c_dim, self.reso_plane, self.reso_plane)
        count_grid = count.reshape(batch_size, 1, self.reso_plane, self.reso_plane)

        # Start with max features (they're non-zero even for single-point cells)
        fea_plane = fea_plane_max.clone()

        # Progressively dilate to fill voids
        kernel_size = 3
        for i in range(8):  # 8 iterations fills up to 16-pixel voids
            # Find current empty cells
            empty_mask = (count_grid == 0).float()    
            if empty_mask.sum() == 0:
                break  # All cells filled
            # Dilate features into empty regions
            dilated = F.max_pool2d(fea_plane, kernel_size, stride=1, padding=kernel_size//2)
            # Only fill empty cells, preserve original occupied cells
            fea_plane = fea_plane * (1 - empty_mask) + dilated * empty_mask
            # Update count for next iteration
            count_dilated = F.max_pool2d(count_grid, kernel_size, stride=1, padding=kernel_size//2)
            count_grid = count_grid * (1 - empty_mask) + count_dilated * empty_mask
        
        # Blend with mean features where we have multiple points
        # Use sigmoid to create smooth transition
        blend_weight = torch.sigmoid((count_grid - 2.0) / 1.0)  # Transitions around 2 points/cell
        fea_plane = fea_plane * (1 - blend_weight) + fea_plane_mean * blend_weight
            
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        
        return fea_plane
    
    # takes in "p": point cloud and "query": sdf_xyz 
    # sample plane features for unlabeled_query as well 
    def forward(self, p, query):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        
        fea = {}
        plane_feat_sum = 0
        #denoise_loss = 0
        
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            #plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            #plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
            #plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        # REPLACE simple summation with weighted aggregation
        plane_features = []
        plane_weights = []

        if 'xz' in self.plane_type:
            feat_xz = self.sample_plane_feature(query, fea['xz'], 'xz')
            # Weight based on distance from y-axis (y=0 is problematic for xz plane)
            weight_xz = torch.abs(query[:, :, 1:2]).transpose(2, 1) + 0.1  # Add 0.1 to avoid zero
            plane_features.append(feat_xz)
            plane_weights.append(weight_xz)

        if 'xy' in self.plane_type:
            feat_xy = self.sample_plane_feature(query, fea['xy'], 'xy')
            # Weight based on distance from z-axis (z=0 is problematic for xy plane)
            weight_xy = torch.abs(query[:, :, 2:3]).transpose(2, 1) + 0.1
            plane_features.append(feat_xy)
            plane_weights.append(weight_xy)
    
        if 'yz' in self.plane_type:
            feat_yz = self.sample_plane_feature(query, fea['yz'], 'yz')
            # Weight based on distance from x-axis (x=0 is problematic for yz plane)
            weight_yz = torch.abs(query[:, :, 0:1]).transpose(2, 1) + 0.1
            plane_features.append(feat_yz)
            plane_weights.append(weight_yz)
        
        # Stack and normalize weights
        plane_features = torch.stack(plane_features, dim=0)  # (3, B, C, N)
        plane_weights = torch.stack(plane_weights, dim=0)     # (3, B, 1, N)
        plane_weights = plane_weights / plane_weights.sum(dim=0, keepdim=True)  # Normalize
    
        # Weighted sum
        plane_feat_sum = (plane_features * plane_weights).sum(dim=0)

        return plane_feat_sum.transpose(2,1)

    # given plane features with dimensions (3*dim, 64, 64)
    # first reshape into the three planes, then generate query features from it 
    def forward_with_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        idx = int(plane_features.shape[1] / 3)
        fea = {}
        fea['xz'], fea['xy'], fea['yz'] = plane_features[:,0:idx,...], plane_features[:,idx:idx*2,...], plane_features[:,idx*2:,...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])

        # USE WEIGHTED AGGREGATION (same as forward())
        plane_features_list = []
        plane_weights = []

        # XZ plane
        feat_xz = self.sample_plane_feature(query, fea['xz'], 'xz')
        weight_xz = torch.abs(query[:, :, 1:2]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_xz)
        plane_weights.append(weight_xz)

        # XY plane  
        feat_xy = self.sample_plane_feature(query, fea['xy'], 'xy')
        weight_xy = torch.abs(query[:, :, 2:3]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_xy)
        plane_weights.append(weight_xy)

        # YZ plane
        feat_yz = self.sample_plane_feature(query, fea['yz'], 'yz')
        weight_yz = torch.abs(query[:, :, 0:1]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_yz)
        plane_weights.append(weight_yz)

        # Stack, normalize, and weight
        plane_features_tensor = torch.stack(plane_features_list, dim=0)
        plane_weights = torch.stack(plane_weights, dim=0)
        plane_weights = plane_weights / plane_weights.sum(dim=0, keepdim=True)
    
        plane_feat_sum = (plane_features_tensor * plane_weights).sum(dim=0)
        
        #plane_feat_sum = 0
        #plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
        #plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        #plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)

    # c is point cloud features
    # p is point cloud (coordinates)
    def forward_with_pc_features(self, c, p, query):

        #print("c, p shapes:", c.shape, p.shape)

        fea = {}
        fea['xz'] = self.generate_plane_features(p, c, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
        fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        # USE WEIGHTED AGGREGATION (same as forward())
        plane_features_list = []
        plane_weights = []

        # XZ plane
        feat_xz = self.sample_plane_feature(query, fea['xz'], 'xz')
        weight_xz = torch.abs(query[:, :, 1:2]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_xz)
        plane_weights.append(weight_xz)

        # XY plane  
        feat_xy = self.sample_plane_feature(query, fea['xy'], 'xy')
        weight_xy = torch.abs(query[:, :, 2:3]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_xy)
        plane_weights.append(weight_xy)

        # YZ plane
        feat_yz = self.sample_plane_feature(query, fea['yz'], 'yz')
        weight_yz = torch.abs(query[:, :, 0:1]).transpose(2, 1) + 0.1
        plane_features_list.append(feat_yz)
        plane_weights.append(weight_yz)

        # Stack, normalize, and weight
        plane_features_tensor = torch.stack(plane_features_list, dim=0)
        plane_weights = torch.stack(plane_weights, dim=0)
        plane_weights = plane_weights / plane_weights.sum(dim=0, keepdim=True)
    
        plane_feat_sum = (plane_features_tensor * plane_weights).sum(dim=0)
        
        #plane_feat_sum = 0
        #plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
        #plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        #plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def get_point_cloud_features(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        return c

    def get_plane_features(self, p):

        c = self.get_point_cloud_features(p)
        fea = {}
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea['xz'], fea['xy'], fea['yz']


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane =='xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
        xy_new = xy_new + 0.5 # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new


    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            x (tensor): coordinate
            reso (int): defined resolution
            coord_type (str): coordinate type
        '''
        return fast_coordinate2index(x, reso)
        #x = (x * reso).long()
        # Ensure indices are within bounds
        #x = torch.clamp(x, 0, reso - 1)  # ADD THIS

        #index = x[:, :, 0] + reso * x[:, :, 1]
        #index = index[:, None, :]
        
        #return index


    # xy is the normalized coordinates of the point cloud of each plane 
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input 
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    
    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, 
                                     vgrid, 
                                     padding_mode='zeros', 
                                     align_corners=False, 
                                     mode='bilinear').squeeze(-1)
        #sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat


def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=False, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1,
        bias=False
    )


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        # Use Sequential for better memory efficiency
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace saves memory
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2, 2) if pooling else nn.Identity()
        
        #self.conv1 = conv3x3(self.in_channels, self.out_channels)
        #self.conv2 = conv3x3(self.out_channels, self.out_channels)

        # ADD THESE TWO LINES
        #self.bn1 = nn.BatchNorm2d(self.out_channels)
        #self.bn2 = nn.BatchNorm2d(self.out_channels)

        #if self.pooling:
        #    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        return self.pool(x), x
        #x = F.relu(self.bn1(self.conv1(x)))  # Add bn1
        #x = F.relu(self.bn2(self.conv2(x)))  # Add bn2
        #before_pool = x
        #if self.pooling:
        #    x = self.pool(x)
        #return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        
        # Determine input channels after merge
        conv_in_channels = 2 * out_channels if merge_mode == 'concat' else out_channels

        # Fused conv block (same pattern as DownConv)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ADD THESE TWO LINES
        #self.bn1 = nn.BatchNorm2d(self.out_channels)
        #self.bn2 = nn.BatchNorm2d(self.out_channels)
        
        #if self.merge_mode == 'concat':
        #    self.conv1 = conv3x3(
        #        2*self.out_channels, self.out_channels)
        #else:
        #    # num of input channels to conv2 is same
        #    self.conv1 = conv3x3(self.out_channels, self.out_channels)
        #self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1) if self.merge_mode == 'concat' else from_up + from_down
        #x = F.relu(self.bn1(self.conv1(x)))  # Add bn1
        #x = F.relu(self.bn2(self.conv2(x)))  # Add bn2
        # Single call to sequential block
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', same_channels=False,
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i) if not same_channels else self.in_channels
            pooling = True if i < depth-1 else False
            #print("down ins, outs: ", ins, outs)  # [latent dim, 32], [32, 64]...[128, 256]

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2 if not same_channels else ins 
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)
            #print("up ins, outs: ", ins, outs)# [256, 128]...[64, 32]; final 32 to latent is done through self.conv_final 

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    #@staticmethod
    #def weight_init(m):
    #    if isinstance(m, nn.Conv2d):
    #        init.xavier_normal_(m.weight)
    #        init.constant_(m.bias, 0)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            #print("down {} x1: ".format(i), x.shape) # increasing channels but decreasing resolution (64x64 -> 8x8)
            x, before_pool = module(x)
            #print("down {} x2: ".format(i), x.shape)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            #print("up {} x1: ".format(i), x.shape)
            x = module(before_pool, x)
            #print("up {} x2: ".format(i), x.shape)
        #exit()
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

    def generate(self, x):
        return self(x)

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, use_bn=True):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h, bias=not use_bn)
        self.fc_1 = nn.Linear(size_h, size_out, bias=not use_bn)
        
        # Use LayerNorm for stability (faster than BatchNorm1d transpose)
        self.ln0 = nn.LayerNorm(size_h) if use_bn else nn.Identity()
        self.ln1 = nn.LayerNorm(size_out) if use_bn else nn.Identity()
        
        self.actvn = nn.ReLU(inplace=True)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        
        net = self.ln0(self.fc_0(self.actvn(x)))
        dx = self.ln1(self.fc_1(self.actvn(net)))
        
        #net = self.fc_0(self.actvn(x))
        #dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx