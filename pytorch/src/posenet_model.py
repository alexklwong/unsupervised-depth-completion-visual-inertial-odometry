import torch
import networks


class PoseNetModel(object):
    '''
    Pose network for computing relative pose between a pair of images

    Arg(s):
        encoder_type : str
            posenet
        rotation_parameterization : str
            euler, axis
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='posenet',
                 rotation_parameterization='euler',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 device=torch.device('cuda')):

        self.device = device

        # Create pose encoder
        if encoder_type == 'posenet':
            self.encoder = networks.PoseEncoder(
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256, 256, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Create pose decoder
        if encoder_type == 'posenet':
            self.decoder = networks.PoseDecoder(
                rotation_parameterization=rotation_parameterization,
                weight_initializer=weight_initializer,
                input_channels=256)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Move to device
        self.to(self.device)

    def forward(self, image0, image1):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                image at time step 0
            image1 : torch.Tensor[float32]
                image at time step 1
        Returns:
            torch.Tensor[float32] : pose from time step 1 to 0
        '''

        # Forward through the network
        latent, _ = self.encoder(torch.cat([image0, image1], dim=1))
        output = self.decoder(latent)

        return output

    def parameters(self):
        '''
        Returns the set of parameters
        '''

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}
        # Save training state
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder weights
        if isinstance(self.encoder, torch.nn.DataParallel):
            checkpoint['encoder_state_dict'] = \
                self.encoder.module.state_dict()
        else:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()

        # Save depth decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            checkpoint['decoder_state_dict'] = \
                self.decoder.module.state_dict()
        else:
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore encoder weights
        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # Restore decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)
