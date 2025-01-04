import numpy as np
import torch
from torch import nn

import warnings 
warnings.filterwarnings("ignore")

'''
---------------------------------------------------------------------------------------------------------------------
Generator
---------------------------------------------------------------------------------------------------------------------
'''

class ResBlock(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super(ResBlock, self).__init__()
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, padding="same")
    def forward(self, x):
        output = self.lrelu1(x)
        output = self.conv1(output)
        output = self.lrelu2(output)
        output = self.conv2(output)
        result = x + output
        return result

class EncoderNet(nn.Module):
    def __init__(self, data_dim, latent_dim, n_layers):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=data_dim, out_channels=latent_dim, kernel_size=6, padding="same")
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim*2, kernel_size=3, padding="same")
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        reslayer = []
        for i in range(n_layers):
            reslayer.append( ResBlock(input_dim=latent_dim*2, kernel_size=3) )
        self.reslayer = nn.Sequential(*reslayer)
        
        self.conv3 = nn.Conv1d(in_channels=latent_dim*2, out_channels=latent_dim*4, kernel_size=3, padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=latent_dim*4, out_channels=latent_dim*4, kernel_size=3, padding="same")
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv1d(in_channels=latent_dim*4, out_channels=latent_dim*4, kernel_size=3, padding="same")
        self.relu5 = nn.ReLU()
        
    def forward(self, x):
        x = self.maxpool1( self.relu1( self.conv1(x) ) )
        x = self.maxpool2( self.relu2( self.conv2(x) ) )
        x = self.reslayer(x)
        x = self.relu3( self.conv3(x) )
        x = self.relu4( self.conv4(x) )
        output = self.relu5( self.conv5(x) )
        return output


class DecoderNet(nn.Module):
    def __init__(self, z_dim, latent_dim, n_layers, data_seqlen):
        super(DecoderNet, self).__init__()
        
        self.latent_dim = latent_dim
        self.data_seqlen = data_seqlen
        
        self.linear1 = nn.Linear( in_features=z_dim, out_features=latent_dim * 4 * int(data_seqlen/2/2) )
        self.conv1 = nn.Conv1d(in_channels=latent_dim * 4, out_channels=latent_dim * 4, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=latent_dim * 4, out_channels=latent_dim * 4, kernel_size=3, padding="same")
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=latent_dim * 4, out_channels=latent_dim * 2, kernel_size=3, padding="same")
        self.relu3 = nn.ReLU()
        
        reslayer = []
        for i in range(n_layers):
            reslayer.append( ResBlock(input_dim=latent_dim*2, kernel_size=3) )
        self.reslayer = nn.Sequential(*reslayer)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest') # [64, 256, 82]
        self.conv4 = nn.Conv1d(in_channels=latent_dim * 2, out_channels=latent_dim, kernel_size=3, padding="same")
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv1d(in_channels=latent_dim, out_channels=4, kernel_size=6, padding="same")
        self.linear2 = nn.Linear( in_features=int(data_seqlen/2/2) * 16, out_features=data_seqlen * 4 )
        
    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1, self.latent_dim*4, int(self.data_seqlen/2/2))
        x = self.relu1( self.conv1(x) )
        x = self.relu2( self.conv2(x) )
        x = self.relu3( self.conv3(x) )
        x = self.reslayer(x) # [64, 256, 41]
        x = self.conv4( self.upsample1(x) ) # torch.Size([64, 128, 82])
        x = self.conv5( self.upsample2(x) ) # torch.Size([64, 4, 164])
        x = x.view(-1, int(self.data_seqlen/2/2) * 16) # [64, 656]
        x = self.linear2(x) # [64, 660]
        output = x.view(-1, self.data_seqlen, 4)
        return output
    



class ProjectionZ(nn.Module):
    def __init__(self, data_seqlen, latent_dim, z_dim):
        super(ProjectionZ, self).__init__()
        
        self.data_seqlen = data_seqlen
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        
        self.linear1 = nn.Linear(in_features= self.latent_dim * 4 * int(self.data_seqlen/2/2) , out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=self.z_dim)
    
    def forward(self, x):
        x = x.view( -1, self.latent_dim * 4 * int(self.data_seqlen/2/2) )
        x = self.linear1(x)
        output = self.linear2(x)
        return output

class ProjectionY(nn.Module):
    def __init__(self, data_seqlen, latent_dim, nbin):
        super(ProjectionY, self).__init__()

        self.data_seqlen = data_seqlen
        self.latent_dim = latent_dim
        
        self.dropout1 = nn.Dropout(0.2)        
        self.bilstm = nn.LSTM(input_size=self.latent_dim * 4, hidden_size=50, bidirectional=True, batch_first=True) 
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear( in_features=int(self.data_seqlen/2/2) * 100,  out_features=512)
        self.dropout2 = nn.Dropout(0.2)
        self.dense = nn.Linear( in_features=512, out_features=nbin * nbin )
        
    def forward(self, x):
        x = self.dropout1(x)
        x = x.permute(2,0,1) # [batch_size, data_dim, seq_length] --> (seq_length,batch_size,input_size)
        x, _ = self.bilstm(x) # (length, batch_size, hidden_dim) --> (length, batch_size, output_dim) 
        x = x.permute(1,2,0)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout2(x)
        output = self.dense(x)
        
        return output




class DiscriminatorZ(nn.Module):
    def __init__(self, z_dim):
        super(DiscriminatorZ, self).__init__()
        self.linear1 = nn.Linear( in_features=z_dim, out_features=z_dim*64 )
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear( in_features=z_dim*64, out_features=1 )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        output = self.linear2(x)
        # output = self.sigmoid(output)
        return output

class DiscriminatorY(nn.Module):
    def __init__(self, nbin):
        super(DiscriminatorY, self).__init__()
        self.linear1 = nn.Linear( in_features=nbin, out_features=1024 )
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear( in_features=1024, out_features=1 )

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        output = self.linear2(x)
        return output

'''
---------------------------------------------------------------------------------------------------------------------
Predictor
---------------------------------------------------------------------------------------------------------------------
'''

class BottleneckLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding="same")
    
    def forward(self,x):
        x = self.conv1( self.relu1( self.bn1(x) ) )
        x = self.conv2( self.relu2( self.bn2(x) ) )
        return x


class DenseBlock(nn.Module):
    def __init__(self, nb_layers=6, hidden_dim=128):
        super(DenseBlock, self).__init__()
        
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        
        self.bn = nn.ModuleList([BottleneckLayer(hidden_dim + 32 * i) for i in range(nb_layers)])

    def forward(self, x):
        layers_concat = []
        layers_concat.append(x)
        for i in range(self.nb_layers):
            x = torch.cat(layers_concat, dim=1)
            x = self.bn[i](x)
            layers_concat.append(x)
        x = torch.cat(layers_concat, dim=1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, input_dim):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=int(input_dim*0.5), kernel_size=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu1( self.bn1(x) )
        x = self.pool1( self.conv1(x) )
        return x
        
    
if __name__ == "__main__":
    inputs = torch.randn(64,4,165) # [batch_size, data_dim, seq_length]
    encoder = EncoderNet(4, 128, 8)
    emb = encoder(inputs) # emb.shape=[64, 512, 41]

    linear_z = ProjectionZ(data_seqlen=165, latent_dim=128, z_dim=64) 
    z = linear_z(emb) # z.shape = [64, 64]
    
    linear_y = ProjectionY(data_seqlen=165, latent_dim=128, nbin=3)
    y_ec, y_pa = linear_y(emb) # y_ec.shape=[64,3]
    
    decoder = DecoderNet(3, 128, 8, 165)
    tmp = decoder(y_ec) # [64, 165, 4]
    print(tmp.shape)
