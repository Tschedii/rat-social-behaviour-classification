"""CNN + Transformer hybrid for rat behavior classification over fixed
32-frame, multi-camera bout windows.

    per-frame RGBM crop --[shared CNN backbone]--> per-frame, per-camera embedding
        --[camera fusion: transformer + attention pooling]--> per-frame embedding
        --[temporal transformer over 32 steps, CLS token]--> bout embedding
        --[concat log-duration]--[linear head]--> class logits

See README.md for the reasoning behind combining a pretrained CNN (spatial /
appearance features, data-efficient) with transformers (temporal dynamics and
cross-camera fusion, where attention naturally handles occlusion / variable
camera reliability) rather than a from-scratch video transformer.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision


def _num_groups_for(channels, target=32):
    """Largest group count <= target that evenly divides channels (GroupNorm requires
    channels % groups == 0, and backbone channel counts aren't all multiples of 32)."""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def convert_bn_to_gn(module, target_groups=32):
    """Replace every BatchNorm2d with GroupNorm, in place.

    Required here, not optional: with native-resolution (1024x1024) input, GPU memory
    forces batch_size=1, so every training step's "batch" is just the frames of a single
    bout — a small, highly correlated sample that gives BatchNorm poor statistics (verified
    empirically: chunking the backbone forward for memory, which should be a pure no-op,
    changed the output when BatchNorm was in place, because each chunk computed its own
    batch statistics from a different subset of frames). GroupNorm normalizes per-sample
    over channel groups, so it has no batch-dependence at all — chunking becomes exactly
    invariant, and training isn't relying on noisy statistics from one bout's frames.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = _num_groups_for(child.num_features, target_groups)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            convert_bn_to_gn(child, target_groups)
    return module


class SpatialAttentionPool(nn.Module):
    """Learned attention over spatial positions of a CNN's final feature map, replacing
    global average pooling so the network can choose WHERE to look (e.g. a rat's tail or
    paws) instead of always averaging uniformly across the whole crop.

    Motivation: Immobile stayed at exactly 0.000 recall through every class-imbalance fix
    (sampler, loss re-weighting) despite a sample count comparable to classes that recovered
    fine (Rearing, Body touch) — pointing away from "under-triggered" and toward "the
    backbone's features don't carry the right signal to begin with." Global average pooling
    discards spatial arrangement entirely; several of the confused classes (Grooming vs.
    Rearing vs. Immobile, Investigation vs. Tail investigation) plausibly differ in exactly
    the kind of localized body-part detail that gets washed out by averaging over the whole
    crop, most of which is background/cage floor once a rat only partly fills the frame.
    """

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        logits = self.score(x).reshape(N, H * W)
        weights = torch.softmax(logits, dim=-1).reshape(N, 1, H, W)
        return (x * weights).sum(dim=(2, 3))  # (N, C)


class CNNBackbone(nn.Module):
    """Per-frame RGBM (4-channel) encoder built from an ImageNet-pretrained backbone.

    The trunk (conv layers only, no pooling/classifier head) is kept under `self.net` with
    the *same submodule names* torchvision uses (conv1/bn1/.../layer4 for resnet18,
    features for efficientnet_b0) specifically so that a checkpoint trained before this
    pooling change still loads via `load_state_dict(..., strict=False)`: every conv/norm
    weight matches by name and warm-starts, only the new attn_pool (if enabled) starts
    fresh. Only avgpool/fc (resnet) and avgpool/classifier (efficientnet) are dropped, and
    neither carries any parameters, so nothing is lost by omitting them.
    """

    def __init__(self, d_model=256, arch="resnet18", pretrained=True, norm_layer="groupnorm",
                 pooling="attention"):
        super().__init__()
        trunk = nn.Module()
        if arch == "resnet18":
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = net.fc.in_features
            old_conv = net.conv1
            new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            net.conv1 = new_conv
            trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
            trunk.layer1, trunk.layer2 = net.layer1, net.layer2
            trunk.layer3, trunk.layer4 = net.layer3, net.layer4
            self._trunk_order = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        elif arch == "efficientnet_b0":
            net = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = net.classifier[1].in_features
            old_conv = net.features[0][0]
            new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            net.features[0][0] = new_conv
            trunk.features = net.features
            self._trunk_order = ["features"]
        else:
            raise ValueError(f"unknown arch {arch}")
        self.net = trunk

        if norm_layer == "groupnorm":
            convert_bn_to_gn(self.net)
        elif norm_layer != "batchnorm":
            raise ValueError(f"unknown norm_layer {norm_layer}")

        self.pooling = pooling
        if pooling == "attention":
            self.attn_pool = SpatialAttentionPool(feat_dim)
        elif pooling != "avg":
            raise ValueError(f"unknown pooling {pooling}")

        self.proj = nn.Linear(feat_dim, d_model)

    def forward(self, x):
        # x: (N, 4, H, W)
        for name in self._trunk_order:
            x = getattr(self.net, name)(x)
        # x: (N, C, H', W') -- spatial feature map, pre-pooling
        pooled = self.attn_pool(x) if self.pooling == "attention" else x.mean(dim=(2, 3))
        return self.proj(pooled)


class FlowBackbone(nn.Module):
    """Lightweight CNN for 2-channel optical flow (dx, dy), fused into the RGB embedding
    before the camera-fusion/temporal transformer stages (see CNNTransformer.forward) — those
    stages, and everything downstream of them, are completely unaffected by adding this,
    so their trained weights transfer untouched when resuming a checkpoint from before this
    branch existed.

    Deliberately small — not a second ResNet18. Optical flow fields are low-frequency/smooth
    (motion between adjacent frames), unlike RGB texture, so a shallow, narrow network run at
    a much smaller resolution (see helpers/rat_h5_dataset.py's flow_size) is enough; the
    native-resolution RGB backbone is already the dominant memory/compute cost in this model,
    and doubling that with a second full backbone would defeat the point of adding motion as
    a cheap additional signal.
    """

    def __init__(self, out_dim=64, norm_layer="groupnorm"):
        super().__init__()

        def conv_block(cin, cout):
            conv = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False)
            norm = nn.GroupNorm(_num_groups_for(cout), cout) if norm_layer == "groupnorm" else nn.BatchNorm2d(cout)
            return nn.Sequential(conv, norm, nn.ReLU(inplace=True))

        self.conv1 = conv_block(2, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128)
        self.attn_pool = SpatialAttentionPool(128)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.proj(self.attn_pool(x))


class CameraFusion(nn.Module):
    """Fuses per-camera embeddings at a single timestep into one embedding via
    a transformer layer (lets cameras attend to each other) followed by
    learned attention pooling, biased by a mask-visibility confidence score."""

    def __init__(self, d_model=256, num_cams=3, nhead=4, dropout=0.1):
        super().__init__()
        self.cam_embed = nn.Embedding(num_cams, d_model)
        self.conf_proj = nn.Linear(1, d_model)
        # Always present (negligible cost) — used only when forward() is given a `scale`
        # tensor. Lets the model know how zoomed-in a given frame was (see
        # helpers/rat_h5_dataset.py's compute_log_scale_stats): the upstream tracker crops
        # each frame to the tracked bounding box and uniformly rescales it to a fixed output
        # size, which preserves shape/aspect ratio but destroys absolute apparent size — this
        # is the one piece of that lost information the pipeline still records.
        self.scale_proj = nn.Linear(1, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, cam_feats, cam_conf, scale=None):
        # cam_feats: (B, num_cams, d_model), cam_conf: (B, num_cams), scale: (B, num_cams) or None
        B, C, D = cam_feats.shape
        cam_ids = torch.arange(C, device=cam_feats.device).unsqueeze(0).expand(B, C)
        x = cam_feats + self.cam_embed(cam_ids) + self.conf_proj(cam_conf.unsqueeze(-1))
        if scale is not None:
            x = x + self.scale_proj(scale.unsqueeze(-1))
        x = self.encoder_layer(x)
        q = self.pool_query.expand(B, 1, D)
        fused, _ = self.pool_attn(q, x, x)
        return fused.squeeze(1)  # (B, d_model)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=4, nhead=8, seq_len=32, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B = x.shape[0]
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])  # CLS output, (B, d_model)


class CNNTransformer(nn.Module):
    """
    Note on resolution: the CNN backbone's cost scales with input pixel count
    (quadratically-ish, since early conv layers dominate activation memory),
    so training at the source .h5's native 1024x1024 (no downsampling, per
    user requirement — see README) uses ~21x the pixels of a typical 224x224
    setup. The transformer stages (camera fusion, temporal) are unaffected —
    they only ever see fixed-size (d_model,) embeddings regardless of input
    resolution, so all the extra cost/memory is confined to the backbone.

    To keep that feasible on a single GPU, the backbone is applied to the
    flattened (B*T*num_cams) frames in small `micro_batch_size` chunks, each
    wrapped in `torch.utils.checkpoint` during training so activations for a
    chunk are freed after its forward pass and recomputed on-demand during
    backward, rather than all (B*T*num_cams) frames' activations being held
    in memory simultaneously. This trades wall-clock time for memory and does
    not change the computed gradients. Use train.py's --grad-accum-steps to
    reach a larger *effective* batch size without needing a larger physical
    batch in memory.
    """

    def __init__(
        self,
        num_classes,
        num_cams=3,
        seq_len=32,
        d_model=256,
        backbone_arch="resnet18",
        pretrained=True,
        temporal_layers=4,
        temporal_heads=8,
        camera_heads=4,
        dropout=0.1,
        micro_batch_size=8,
        use_checkpoint=True,
        backbone_norm="groupnorm",
        backbone_pooling="attention",
        use_flow=False,
        flow_dim=64,
    ):
        super().__init__()
        self.backbone = CNNBackbone(d_model=d_model, arch=backbone_arch, pretrained=pretrained,
                                     norm_layer=backbone_norm, pooling=backbone_pooling)
        self.use_flow = use_flow
        if use_flow:
            # Fusion happens here, before camera_fusion/temporal/head, so those stages (and
            # their trained weights, on resume) are completely unaffected by this branch.
            self.flow_backbone = FlowBackbone(out_dim=flow_dim, norm_layer=backbone_norm)
            self.fusion_proj = nn.Linear(d_model + flow_dim, d_model)
        self.camera_fusion = CameraFusion(d_model=d_model, num_cams=num_cams, nhead=camera_heads, dropout=dropout)
        self.temporal = TemporalTransformer(
            d_model=d_model, num_layers=temporal_layers, nhead=temporal_heads, seq_len=seq_len, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, num_classes),
        )
        self.micro_batch_size = micro_batch_size
        self.use_checkpoint = use_checkpoint

    def _chunked_forward(self, module, x):
        # x: (N, ..., H, W) where N = B*T*num_cams, chunked to bound peak activation memory.
        # Shared by the RGB and (when enabled) flow backbones.
        chunks = x.split(self.micro_batch_size, dim=0)
        outs = []
        for c in chunks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                out = torch.utils.checkpoint.checkpoint(module, c, use_reentrant=False)
            else:
                out = module(c)
            outs.append(out)
        return torch.cat(outs, dim=0)

    def forward(self, images, cam_conf, duration, flow=None, scale=None):
        # images: (B, num_cams, T, 4, H, W); cam_conf: (B, num_cams); duration: (B, 1)
        # flow (optional): (B, num_cams, T, 2, Hf, Wf); scale (optional): (B, num_cams, T)
        B, C, T, ch, H, W = images.shape
        x = images.permute(0, 2, 1, 3, 4, 5).reshape(B * T * C, ch, H, W)
        feats = self._chunked_forward(self.backbone, x)  # (B*T*C, d_model)

        if self.use_flow:
            if flow is None:
                raise ValueError("model was built with use_flow=True but forward() got flow=None")
            _, _, _, fch, fH, fW = flow.shape
            fx = flow.permute(0, 2, 1, 3, 4, 5).reshape(B * T * C, fch, fH, fW)
            flow_feats = self._chunked_forward(self.flow_backbone, fx)  # (B*T*C, flow_dim)
            feats = self.fusion_proj(torch.cat([feats, flow_feats], dim=-1))  # (B*T*C, d_model)

        feats = feats.reshape(B * T, C, -1)             # (B*T, C, d_model)
        conf = cam_conf.unsqueeze(1).expand(B, T, C).reshape(B * T, C)
        scale_bt_c = scale.permute(0, 2, 1).reshape(B * T, C) if scale is not None else None
        fused = self.camera_fusion(feats, conf, scale=scale_bt_c)  # (B*T, d_model)
        fused = fused.reshape(B, T, -1)                 # (B, T, d_model)
        bout_embed = self.temporal(fused)                # (B, d_model)
        logits = self.head(torch.cat([bout_embed, duration], dim=-1))
        return logits

    def backbone_parameters(self):
        return self.backbone.net.parameters()

    def head_parameters(self):
        seen = set(id(p) for p in self.backbone.net.parameters())
        for p in self.parameters():
            if id(p) not in seen:
                yield p
