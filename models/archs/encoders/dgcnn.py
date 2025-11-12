import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import knn_graph

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  
                                       # -> (batch_size*num_points, num_dims) 
                                       #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature

def get_graph_feature_fast(x, k=20):
    """
    Optimized version using torch-cluster
    3-5x faster KNN computation
    """
    batch_size, num_dims, num_points = x.size()
    device = x.device
    
    # Transpose for torch-cluster (expects B*N, C format)
    x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
    x_flat = x_t.view(-1, num_dims)  # (B*N, C)
    
    # Create batch indices for torch-cluster
    batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_points)
    
    # Fast KNN using torch-cluster (much faster than matrix multiplication)
    edge_index = knn_cluster(x_flat, x_flat, k=k, batch_x=batch_idx, batch_y=batch_idx)
    
    # Reshape edge_index to match original format
    # edge_index is (2, E) where E = B*N*k
    _, col = edge_index
    idx = col.view(batch_size, num_points, k)
    
    # Gather features (same as original)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    feature = x_flat[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_repeat = x_t.unsqueeze(2).repeat(1, 1, k, 1)
    
    # Concatenate [neighbor - center, center]
    feature = torch.cat((feature - x_repeat, x_repeat), dim=3)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    
    return feature


class DGCNN(nn.Module):
    """
    DGCNN adapted for SDF learning with VAE
    Produces global features that can be used by VAE and SDF decoder
    """
    def __init__(
        self, 
        emb_dims=512,  # Match your latent_dim
        k=20,
        dropout=0.0,   # No dropout for SDF learning initially
    ):
        super().__init__()
        
        self.k = k
        self.emb_dims = emb_dims
        
        # Edge convolutions
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Global feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # For richer features: combine max and avg pooling
        self.global_feat_dim = emb_dims * 2  # Because we concatenate max and avg
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, N) point cloud
        Returns:
            global_features: (B, emb_dims*2)
            point_features: (B, emb_dims, N) - useful for attention later
        """
        batch_size = x.size(0)
        
        # First EdgeConv layer
        x = get_graph_feature(x, k=self.k)      # (B, 6, N, k)
        x = self.conv1(x)                       # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (B, 64, N)
        
        # Second EdgeConv layer
        x = get_graph_feature(x1, k=self.k)     # (B, 128, N, k)
        x = self.conv2(x)                       # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (B, 64, N)
        
        # Third EdgeConv layer
        x = get_graph_feature(x2, k=self.k)     # (B, 128, N, k)
        x = self.conv3(x)                       # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (B, 128, N)
        
        # Fourth EdgeConv layer
        x = get_graph_feature(x3, k=self.k)     # (B, 256, N, k)
        x = self.conv4(x)                       # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (B, 256, N)
        
        # Concatenate all features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # Get point-wise features
        point_features = self.conv5(x)          # (B, emb_dims, N)
        
        # Global features: both max and avg pooling
        x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1)  # (B, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1)  # (B, emb_dims)
        global_features = torch.cat((x1, x2), 1)  # (B, emb_dims*2)
        
        return global_features, point_features

class DGCNN_old(nn.Module):

    def __init__(
        self, 
        emb_dims=512,
        use_bn=False
    ):

        super().__init__()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm2d(emb_dims)

            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv2d(512, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv2d(512, emb_dims, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()                 # x:      batch x   3 x num of points
        x = get_graph_feature(x)                                    # x:      batch x   6 x num of points x 20

        x1     = self.conv1(x)                                      # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]                    # x1_max: batch x  64 x num of points x 1

        x2     = self.conv2(x1)                                     # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]                    # x2_max: batch x  64 x num of points x 1

        x3     = self.conv3(x2)                                     # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]                    # x3_max: batch x 128 x num of points x 1

        x4     = self.conv4(x3)                                     # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]                    # x4_max: batch x 256 x num of points x 1
 
        x_max  = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1) # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)        # point feat:  batch x 512 x num of points

        global_feat = point_feat.max(dim=2, keepdim=False)[0]       # global feat: batch x 512

        return global_feat