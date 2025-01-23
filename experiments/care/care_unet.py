import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockEncoder(nn.Module):
    def __init__(
        self, in_ch, out_ch, kernel_size=3, dilate=1, n_dim=2, use_batchnorm=False
    ):
        super().__init__()
        pad = kernel_size // 2

        conv_type = nn.Conv2d if n_dim == 2 else nn.Conv3d
        if use_batchnorm:
            bnorm_type = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d

        self.conv = nn.Sequential(
            conv_type(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilate,
            ),
            bnorm_type(out_ch) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            conv_type(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilate,
            ),
            bnorm_type(out_ch) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockBottleneck(nn.Module):
    def __init__(
        self, in_ch, out_ch, kernel_size=3, dilate=1, n_dim=2, use_batchnorm=False
    ):
        super().__init__()
        pad = kernel_size // 2
        conv_type = nn.Conv2d if n_dim == 2 else nn.Conv3d
        bnorm_type = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        self.conv = nn.Sequential(
            conv_type(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilate,
            ),
            bnorm_type(out_ch) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockDecoder(nn.Module):
    def __init__(
        self, in_ch, level, kernel_size=3, dilate=1, n_dim=2, use_batchnorm=False
    ):
        super().__init__()
        pad = kernel_size // 2
        conv_type = nn.Conv2d if n_dim == 2 else nn.Conv3d
        bnorm_type = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        self.conv = nn.Sequential(
            conv_type(
                in_ch,
                32 * 2**level,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilate,
            ),
            bnorm_type(32 * 2**level) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            conv_type(
                32 * 2**level,
                32 * 2 ** max(0, level - 1),
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilate,
            ),
            bnorm_type(32 * 2 ** max(0, level - 1)) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CARE_UNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        n_first_channels=32,
        n_depth=2,
        n_dim=2,
        residual=True,
        use_batchnorm=False,
    ):
        super().__init__()

        self.n_first_channels = n_first_channels
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.residual = residual
        self.n_dim = n_dim

        kernel_size = (
            5 if n_dim == 2 else 3
        )  # the default kernel size is 5 for 2D, 3 for 3D!

        self.downs = nn.ModuleList()
        for i in range(n_depth):
            in_ch, out_ch = self.get_channels_encoder(i)
            self.downs.append(
                ConvBlockEncoder(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    n_dim=n_dim,
                    use_batchnorm=use_batchnorm,
                )
            )

        mid_ch = self.n_first_channels * 2**n_depth
        mid_ch2 = self.n_first_channels * 2 ** max(0, n_depth - 1)

        self.bottleneck = nn.Sequential(
            ConvBlockBottleneck(
                out_ch,
                mid_ch,
                kernel_size=kernel_size,
                n_dim=n_dim,
                use_batchnorm=use_batchnorm,
            ),
            ConvBlockBottleneck(
                mid_ch,
                mid_ch2,
                kernel_size=kernel_size,
                n_dim=n_dim,
                use_batchnorm=use_batchnorm,
            ),
        )

        self.ups = nn.ModuleList()
        for i in reversed(range(n_depth)):
            in_ch, out_ch = self.get_channels_decoder(i)
            self.ups.append(
                ConvBlockDecoder(
                    in_ch=in_ch,
                    level=i,
                    kernel_size=kernel_size,
                    n_dim=n_dim,
                    use_batchnorm=use_batchnorm,
                )
            )

        if n_dim == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        if n_dim == 2:
            self.final_conv = nn.Conv2d(out_ch, self.out_ch, kernel_size=1)
        else:
            self.final_conv = nn.Conv3d(out_ch, self.out_ch, kernel_size=1)

        self.init_weights()

    def get_channels_encoder(self, level):
        if level == 0:
            in_ch = self.in_ch
        else:
            in_ch = 2 ** (level - 1) * self.n_first_channels
        out_ch = 2**level * self.n_first_channels
        return in_ch, out_ch

    def get_channels_decoder(self, level):
        in_ch = 2 ** (level + 1) * self.n_first_channels
        if level == 0:
            out_ch = self.n_first_channels
        else:
            out_ch = 2 ** (level - 1) * self.n_first_channels
        return in_ch, out_ch

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x) -> dict:
        internal_tensor_sizes = []
        skip_connection_tensors = []

        input = x.clone()

        for down in self.downs:
            x = down(x)
            internal_tensor_sizes.append(x.shape[2:])
            skip_connection_tensors.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        internal_tensor_sizes.reverse()
        skip_connection_tensors.reverse()
        for i, up in enumerate(self.ups):
            x = F.interpolate(x, internal_tensor_sizes[i], mode="nearest")
            x = torch.cat((skip_connection_tensors[i], x), dim=1)
            x = up(x)

        output = self.final_conv(x)

        if self.residual:
            output = output + input

        return output
