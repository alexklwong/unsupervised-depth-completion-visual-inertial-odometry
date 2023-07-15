import torch, torchvision
import numpy as np
import data_utils, log_utils, losses, loss_utils, networks


EPSILON = 1e-8


class VOICEDModel(object):
    '''
    Depth Completion from Inertial Odometry and Vision (VOICeD)

    Arg(s):
        encoder_type : str
            encoder types: vggnet08, vggnet11
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        n_filters_encoder_image : list[int]
            number of filters to use in each block of image encoder
        n_filters_encoder_depth : list[int]
            number of filters to use in each block of depth encoder
        n_filters_decoder : list[int]
            number of filters to use in each block of decoder
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        min_predict_depth : float
            minimum depth prediction supported by model
        max_predict_depth : float
            maximum depth prediction supported by model
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type,
                 input_channels_image,
                 input_channels_depth,
                 n_filters_encoder_image,
                 n_filters_encoder_depth,
                 n_filters_decoder,
                 weight_initializer,
                 activation_func,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.encoder_type = encoder_type
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device

        n_filters_encoder = [
            i + z
            for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
        ]
        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        # Build network
        if 'vggnet08' in encoder_type:
            self.encoder_image = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=input_channels_image,
                n_filters=n_filters_encoder_image,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            self.encoder_depth = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=input_channels_depth,
                n_filters=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
        elif 'vggnet11' in encoder_type:
            self.encoder_image = networks.VGGNetEncoder(
                n_layer=11,
                input_channels=input_channels_image,
                n_filters=n_filters_encoder_image,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            self.encoder_depth = networks.VGGNetEncoder(
                n_layer=11,
                input_channels=input_channels_depth,
                n_filters=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        self.decoder = networks.VOICEDDecoder(
            input_channels=n_filters_encoder[-1],
            output_channels=1,
            n_filters=n_filters_decoder,
            n_skips=n_skips[:-1],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            output_func='sigmoid',
            use_batch_norm=False,
            deconv_type='up')

        # Move to device
        self.to(self.device)

    def forward(self, image, sparse_depth, validity_map):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        # Perform scaffolding
        scaffolding = [
            np.expand_dims(data_utils.interpolate_depth(np.squeeze(z), np.squeeze(v)), axis=0)
            for z, v in zip(sparse_depth.cpu().numpy(), validity_map.cpu().numpy())
        ]

        scaffolding = torch.from_numpy(np.stack(scaffolding, axis=0).astype(np.float32)).to(self.device)

        input_depth = torch.cat([
            scaffolding,
            validity_map], dim=1)

        # Forward through the network
        latent_image, skips_image = self.encoder_image(image)
        latent_depth, skips_depth = self.encoder_depth(input_depth)

        latent = torch.cat([latent_image, latent_depth], dim=1)

        skips = [
            torch.cat([skip_image, skip_depth], dim=1)
            for skip_image, skip_depth in zip(skips_image, skips_depth)
        ]

        output_depth0 = self.decoder(latent, skips, image.shape[-2:])[-1]

        # Convert inverse depth to depth
        output_depth0 = \
            self.min_predict_depth / (output_depth0 + self.min_predict_depth / self.max_predict_depth)

        return output_depth0

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     pose1to0=None,
                     pose2to0=None,
                     w_color=0.20,
                     w_structure=0.80,
                     w_sparse_depth=1.00,
                     w_smoothness=0.15,
                     w_pose=0.10):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{pc}l_{pc}

        Args:
            image0 : torch.Tensor[float32]
                image at time step t
            image1 : torch.Tensor[float32]
                image at time step t-1
            image2 : torch.Tensor[float32]
                image at time step t+1
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            intrinsics : torch.Tensor[float32]
                3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            pose1to0 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t-1 to t
            pose2to0 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t+1 to t
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
            w_pose : float
                weight of forward-backward pose consistency term
        Returns:
            float : loss
        '''

        validity_map_depth0 = validity_map0
        validity_map_image0 = 1.0 - validity_map_depth0
        shape = image0.shape

        # Backproject points to 3D camera coordinates
        points = loss_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy0to1 = loss_utils.project_to_pixel(points, pose0to1, intrinsics, shape)
        target_xy0to2 = loss_utils.project_to_pixel(points, pose0to2, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image1to0 = loss_utils.grid_sample(image1, target_xy0to1, shape)
        image2to0 = loss_utils.grid_sample(image2, target_xy0to2, shape)

        # Color consistency loss function
        loss_color1to0 = losses.color_consistency_loss_func(
            src=image1to0,
            tgt=image0,
            w=validity_map_image0)

        loss_color2to0 = losses.color_consistency_loss_func(
            src=image2to0,
            tgt=image0,
            w=validity_map_image0)

        loss_color = loss_color1to0 + loss_color2to0

        # Structural consistency loss function
        loss_structure1to0 = losses.structural_consistency_loss_func(
            src=image1to0,
            tgt=image0,
            w=validity_map_image0)

        loss_structure2to0 = losses.structural_consistency_loss_func(
            src=image2to0,
            tgt=image0,
            w=validity_map_image0)

        loss_structure = loss_structure1to0 + loss_structure2to0

        # Sparse depth consistency loss function
        loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
            src=output_depth0,
            tgt=sparse_depth0,
            w=validity_map_depth0)

        # Local smoothness loss function
        loss_smoothness = losses.smoothness_loss_func(
            predict=output_depth0,
            image=image0)

        # Pose consistency loss function
        if w_pose > 0.0 and pose1to0 is not None and pose2to0 is not None:
            loss_pose01 = losses.pose_consistency_loss_func(
                pose0=pose0to1,
                pose1=pose1to0)

            loss_pose02 = losses.pose_consistency_loss_func(
                pose0=pose0to2,
                pose1=pose2to0)

            loss_pose = loss_pose01 + loss_pose02
        else:
            loss_pose = 0.0

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{pc}l_{pc}
        loss = \
            w_color * loss_color + \
            w_structure * loss_structure + \
            w_sparse_depth * loss_sparse_depth + \
            w_smoothness * loss_smoothness + \
            w_pose * loss_pose

        loss_info = {
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            'loss_sparse_depth' : loss_sparse_depth,
            'loss_smoothness' : loss_smoothness,
            'loss' : loss,
            'image1to0' : image1to0,
            'image2to0' : image2to0
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        return list(self.encoder_image.parameters()) + \
            list(self.encoder_depth.parameters()) + \
            list(self.decoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder_image.train()
        self.encoder_depth.train()

        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder_image.eval()
        self.encoder_depth.eval()

        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder_image.to(device)
        self.encoder_depth.to(device)

        self.decoder.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder_image = torch.nn.DataParallel(self.encoder_image)
        self.encoder_depth = torch.nn.DataParallel(self.encoder_depth)
        self.decoder = torch.nn.DataParallel(self.decoder)

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
        if isinstance(self.encoder_image, torch.nn.DataParallel):
            checkpoint['encoder_image_state_dict'] = self.encoder_image.module.state_dict()
        else:
            checkpoint['encoder_image_state_dict'] = self.encoder_image.state_dict()

        if isinstance(self.encoder_depth, torch.nn.DataParallel):
            checkpoint['encoder_depth_state_dict'] = self.encoder_depth.module.state_dict()
        else:
            checkpoint['encoder_depth_state_dict'] = self.encoder_depth.state_dict()

        # Save depth decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            checkpoint['decoder_state_dict'] = self.decoder.module.state_dict()
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
        if isinstance(self.encoder_image, torch.nn.DataParallel):
            self.encoder_image.module.load_state_dict(checkpoint['encoder_image_state_dict'])
        else:
            self.encoder_image.load_state_dict(checkpoint['encoder_image_state_dict'])

        if isinstance(self.encoder_depth, torch.nn.DataParallel):
            self.encoder_depth.module.load_state_dict(checkpoint['encoder_depth_state_dict'])
        else:
            self.encoder_depth.load_state_dict(checkpoint['encoder_depth_state_dict'])

        # Restore depth decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    ground_truth0=None,
                    pose0to1=None,
                    pose0to2=None,
                    pose1to0=None,
                    pose2to0=None,
                    scalars={},
                    n_image_per_summary=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image at time step t
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth at time t
            validity_map0 : torch.Tensor[float32]
                validity map of sparse depth at time t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth at time t
            pose0to1 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            pose1to0 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t-1 to t
            pose2to0 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t+1 to t
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image0 is not None and image1to0 is not None:
                image1to0_summary = image1to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image1to0-error'

                # Compute reconstruction error w.r.t. image 0
                image1to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image1to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image1to0_error_summary = log_utils.colorize(
                    (image1to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image1to0_summary.cpu(),
                        image1to0_error_summary],
                        dim=3))

            if image0 is not None and image2to0 is not None:
                image2to0_summary = image2to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image2to0-error'

                # Compute reconstruction error w.r.t. image 0
                image2to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image2to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image2to0_error_summary = log_utils.colorize(
                    (image2to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image2to0_summary.cpu(),
                        image2to0_error_summary],
                        dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
                sparse_depth0_summary = sparse_depth0[0:n_image_per_summary, ...]
                validity_map0_summary = validity_map0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_sparse0-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth0_error_summary = \
                    torch.abs(output_depth0_summary - sparse_depth0_summary)

                sparse_depth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (sparse_depth0_error_summary + EPSILON) / (sparse_depth0_summary + EPSILON),
                    validity_map0_summary)

                # Add to list of images to log
                sparse_depth0_summary = log_utils.colorize(
                    (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth0_error_summary = log_utils.colorize(
                    (sparse_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth0_summary,
                        sparse_depth0_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:
                ground_truth0_summary = ground_truth0[0:n_image_per_summary, ...]

                n_channel = ground_truth0_summary.shape[1]

                if n_channel == 1:
                    validity_map0_summary = torch.where(
                        ground_truth0 > 0,
                        torch.ones_like(ground_truth0_summary),
                        torch.zeros_like(ground_truth0_summary))
                else:
                    validity_map0_summary = torch.unsqueeze(ground_truth0_summary[:, 1, :, :], dim=1)
                    ground_truth0_summary = torch.unsqueeze(ground_truth0_summary[:, 0, :, :], dim=1)

                display_summary_depth_text += '_groundtruth0-error'

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + EPSILON) / (ground_truth0_summary + EPSILON),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose0to1 is not None:
                # Log distribution of pose 0 to 1 translation vector
                summary_writer.add_histogram(tag + '_tx0to1_distro', pose0to1[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to1_distro', pose0to1[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to1_distro', pose0to1[:, 2, 3], global_step=step)

            if pose0to2 is not None:
                # Log distribution of pose 0 to 2 translation vector
                summary_writer.add_histogram(tag + '_tx0to2_distro', pose0to2[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to2_distro', pose0to2[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to2_distro', pose0to2[:, 2, 3], global_step=step)

            if pose1to0 is not None:
                # Log distribution of pose 1 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx1to0_distro', pose1to0[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty1to0_distro', pose1to0[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz1to0_distro', pose1to0[:, 2, 3], global_step=step)

            if pose2to0 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx2to0_distro', pose2to0[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty2to0_distro', pose2to0[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz2to0_distro', pose2to0[:, 2, 3], global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                global_step=step)
