import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torchvision import datasets, transforms
from stylesynth_unet import StyleSynth_UNet


class StyleSynth_DiffusionModel:
    '''
    The diffusion model used in the StyleSynth system
    '''
    def __init__(self, T, w, upper_beta, device, logger):
        self.logger = logger

        # set device
        self.device = device

        # set number of total timesteps
        self.T = T

        # set scaling factor when subtracting average noise from context noise
        self.w = w

        # set variance schedule (beta) and other variables based on it
        self.upper_beta = upper_beta
        self.beta = torch.concat([
            torch.Tensor([0]),
            torch.linspace(self.upper_beta / self.T, self.upper_beta, self.T)
        ], dim=-1).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.sqrt_alpha_inv = torch.sqrt(1 / self.alpha)
        self.eps_t_coeff = self.beta / self.sqrt_one_minus_alpha_bar

        # transforms for displaying images
        self.display_image_transforms = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
            transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
            transforms.ToPILImage()
        ])

        # details of Fashion MNIST dataset
        self.clothing_type_names = [
            'Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
            'Sneaker', 'Bag', 'Boot']

    def diffusion(self, x_0, t):
        '''
        add t timesteps' worth of noise to the image (sample from the
        distribution of images at time t, conditional on the initial image)
        '''

        # mean image at time t
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
        mu_x_t = sqrt_alpha_bar_t * x_0

        # noise added to mean image at time t
        eps = torch.randn_like(x_0)  # sampled Gaussian noise
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[
            t, None, None, None]  # std. dev. of image values at time t
        noise = sqrt_one_minus_alpha_bar_t * eps

        # sampled image at time t
        x_t = mu_x_t + noise
        return x_t, eps

    def reverse_diffusion(self, x_t, t, eps_t):
        '''
        remove one timestep's worth of noise from the image (sample from the
        distribution of images at time t - 1, conditional on the image at
        time t and the noise added at time t to get that image)
        '''

        # reverse diffusion is complete
        if t == 0:
            return x_t

        # mean image at time t - 1
        sqrt_alpha_inv_t = self.sqrt_alpha_inv[t, None, None, None]
        eps_t_coeff = self.eps_t_coeff[t, None, None, None]
        mu_x_t_minus_1 = sqrt_alpha_inv_t * (x_t - eps_t_coeff * eps_t)

        # noise added to the mean image at time t - 1
        eps = torch.randn_like(x_t)  # sampled Gaussian noise
        # standard deviation of image values at time t - 1
        eps_coeff = torch.sqrt(
            (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
            * self.beta[t])[None, None, None]
        noise = eps_coeff * eps

        # sampled image at time t - 1
        return mu_x_t_minus_1 + noise

    @torch.no_grad()
    def generate_images(self, clothing_types):
        # sampled noise to convert into an image via reverse diffusion
        img_chs = self.unet.img_chs
        img_size = self.unet.img_size
        x_t = torch.randn((len(clothing_types), img_chs, img_size, img_size),
                          device=self.device)

        # get context vectors based on provided classes
        clothing_type_labels = []
        for clothing_type in clothing_types:
            if clothing_type not in self.clothing_type_names:
                raise Exception(
                    f'Invalid clothing type: {clothing_type}. Expected one '
                    + f'of: {self.clothing_type_names}')
            clothing_type_label = self.clothing_type_names.index(clothing_type)
            clothing_type_labels.append(clothing_type_label)
        c = torch.Tensor(clothing_type_labels)
        c = F.one_hot(
            c.to(torch.int64), num_classes=len(self.clothing_type_names))\
            .float().to(self.device)

        # perform reverse diffusion
        for i in range(0, self.T)[::-1]:
            # predict and remove added noise for the current timestep
            t = torch.full((1, ), i, device=self.device).float()
            context_eps_t = self.unet(x_t, t, c)
            avg_eps_t = self.unet(
                x_t, t, torch.zeros_like(c, device=self.device).float())
            eps_t = (1 + self.w) * context_eps_t - self.w * avg_eps_t
            x_t = self.reverse_diffusion(x_t, t.int(), eps_t)

        # return images
        generated_images = []
        for i in range(len(clothing_types)):
            generated_images.append(
                self.display_image_transforms(x_t[i].detach().cpu()))
        return generated_images

    @torch.no_grad()
    def _show_reverse_diffusion(
            self, unet, x_0, c, num_imgs_to_show):
        # sampled noise to convert into an image via reverse diffusion
        x_t = torch.randn_like(x_0, device=self.device)

        # plot images across timesteps during reverse diffusion
        num_rows = x_0.shape[0]
        num_cols = num_imgs_to_show + 1
        _, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            figsize=(num_cols * 2, num_rows * 2))
        # axes = np.array(axes).reshape(num_cols, num_rows).T
        interval = self.T // num_imgs_to_show + 1
        plot_num = 1
        for i in range(0, self.T)[::-1]:
            # predict and remove added noise for the current timestep
            t = torch.full((1, ), i, device=self.device).float()
            context_eps_t = unet(x_t, t, c)
            avg_eps_t = unet(
                x_t, t, torch.zeros_like(c, device=self.device).float())
            eps_t = (1 + self.w) * context_eps_t - self.w * avg_eps_t
            x_t = self.reverse_diffusion(x_t, t.int(), eps_t)

            # plot images
            if i % interval == 0:
                for j in range(num_rows):
                    axes[j, plot_num - 1].imshow(
                         self.display_image_transforms(x_t[j].detach().cpu()))
                    axes[j, plot_num - 1].axis('off')
                plot_num += 1
        # plot original images
        for j in range(num_rows):
            axes[j, plot_num - 1].imshow(
                 self.display_image_transforms(x_0[j].detach().cpu()))
            axes[j, plot_num - 1].axis('off')
        # show plot of generated images and original images
        plt.tight_layout()
        plt.show()

    def train(self, epochs, batch_size, vis_interval, num_imgs_to_generate):
        # image dimension details
        IMG_SIZE = 16
        IMG_CHS = 1

        # transforms for loading data
        load_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # convert images to tensors (range [0, 1])
            transforms.Normalize((0.5,), (0.5,))  # normalize to range [-1, 1]
        ])
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True,
            transform=load_transforms)
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True,
            transform=load_transforms)
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        dataloader = DataLoader(
            combined_dataset, batch_size=batch_size, shuffle=True)

        # instantiate U-Net model for predicting noise added to an image
        unet = StyleSynth_UNet(
            img_size=IMG_SIZE,
            img_chs=IMG_CHS,
            T=self.T,
            down_chs=[80, 128, 160],
            group_size=16,
            t_embed_dim=8,
            c_embed_dim=len(self.clothing_type_names),
            conv_hidden_layers=2,
            dense_embed_hidden_layers=3,
            t_embed_hidden_layers=2,
            c_embed_hidden_layers=2,
            transp_conv_hidden_layers=2,
            device=self.device)

        # set up optimizer
        optimizer = Adam(unet.parameters(), lr=0.001)

        # probability of dropping values from context embedding during training
        train_c_drop_prob = 0.05

        # train model
        unet.train()
        display_img = None
        display_img_c = None
        display_clothing_types = None
        for epoch in tqdm(range(epochs)):
            avg_img_loss = 0
            for step, (imgs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # get model inputs
                t = torch.randint(
                    0, self.T, (imgs.shape[0], ), device=self.device).to(
                    self.device).float()
                x_0 = imgs.to(self.device)
                c = F.one_hot(
                    labels.to(torch.int64),
                    num_classes=len(self.clothing_type_names))
                c_mask = torch.bernoulli(
                    torch.ones_like(c).float() - train_c_drop_prob).to(
                        self.device)
                masked_c = c * c_mask

                # get predicted context noise and average noise
                x_t, eps_t = self.diffusion(x_0, t.int())
                pred_eps_t = unet(x_t, t, masked_c)

                # calculate loss and backpropagate
                loss = F.mse_loss(eps_t, pred_eps_t)
                loss.backward()
                # above code (MSE across all elements) is mathematically
                # equivalent to the below code (MSE across elements in each
                # image, followed by MSE across image-wise MSEs)
                # loss_per_elem = F.mse_loss(
                #     eps_t, pred_eps_t, reduction='none')
                # loss_per_img = loss.mean(dim=(1, 2, 3))
                # avg_loss_per_img = loss_per_img.mean()
                # avg_loss_per_img.backward()
                optimizer.step()

                if display_img is None and display_img_c is None and \
                        display_clothing_types is None:
                    display_img = x_0[:num_imgs_to_generate].detach().clone()
                    display_img_c = c[:num_imgs_to_generate].float().detach()\
                        .clone()
                    display_clothing_types = list(map(
                        lambda i: self.clothing_type_names[i],
                        labels.view(-1).tolist()[:num_imgs_to_generate]))

                # update average loss per image
                avg_img_loss = (avg_img_loss * step + loss.item())\
                    / (step + 1)
                # above code is mathematically equivalent to below code
                # (batch_size terms cancel out)
                # total_img_loss = avg_img_loss * step * batch_size\
                #     + loss.item() * batch_size
                # avg_img_loss = total_img_loss / (batch_size * (step + 1))
                del x_t, eps_t, pred_eps_t, loss  # deleting to save memory

            # show results
            if epoch == 0 or (epoch + 1) % vis_interval == 0 or \
                    epoch + 1 == epochs:
                self.logger.info(f'Epoch {epoch + 1} / {epochs}')
                self.logger.info(f'Average Loss Per Image: {avg_img_loss}')
                self.logger.info(f'Clothing types: {display_clothing_types}')
                self._show_reverse_diffusion(
                    unet, display_img, display_img_c, 10)

        # model set to eval mode since training is done
        unet.eval()
        self.unet = unet
