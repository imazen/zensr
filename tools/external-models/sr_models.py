"""Tensor adapters around pinned author SR modules; no saved model outputs."""
import importlib
import importlib.abc
import importlib.util
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def namespace(name, path):
    """Load inference leaf modules without importing unrelated training registries."""
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


class CPUDeviceLoader(importlib.abc.Loader):
    """Change only hard-coded device selections in the author's CUDA modules."""
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        class Devices(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == 'cuda':
                    return ast.copy_location(ast.Constant('cpu'), node)
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                    if node.args or node.keywords:
                        raise ValueError('CPU oracle does not support CUDA conversion arguments')
                    node.func.attr = 'cpu'
                return node
        tree = Devices().visit(ast.parse(self.path.read_text()))
        module.__file__ = str(self.path)
        exec(compile(ast.fix_missing_locations(tree), str(self.path), 'exec'), module.__dict__)


class CPUDeviceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source):
        self.files = {'s3diff_tile': source/'src/s3diff_tile.py',
            'model': source/'src/model.py', 'my_utils.devices': source/'src/my_utils/devices.py'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.files:
            return importlib.util.spec_from_loader(fullname, CPUDeviceLoader(self.files[fullname]))
        return None


class S3Diff:
    def __init__(self, root, device='cuda'):
        source = root/'source'
        self.device = torch.device(device)
        if self.device.type == 'cpu':
            if torch.cuda.is_available():
                raise ValueError('CPU oracle requires CUDA_VISIBLE_DEVICES to hide GPUs')
            sys.meta_path.insert(0, CPUDeviceFinder(source))
        sys.path[:0] = [str(source/'src'), str(source)]
        for name in ['basicsr', 'basicsr.archs', 'basicsr.ops', 'basicsr.utils']:
            namespace(name, source/Path(*name.split('.')))
        # Both functions remain the author's definitions; only registry side effects
        # (loading every training architecture/dataset) are bypassed.
        sys.modules['basicsr.utils'].get_root_logger = importlib.import_module('basicsr.utils.logger').get_root_logger
        from s3diff_tile import S3Diff as AuthorModel
        from de_net import DEResNet
        from utils.wavelet_color import wavelet_color_fix
        from torchvision.transforms import ToPILImage, ToTensor
        args = SimpleNamespace(latent_tiled_size=96, latent_tiled_overlap=32,
            vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224)
        self.model = AuthorModel(sd_path=str(root/'sd-turbo'), pretrained_path=str(root/'weights/s3diff.pkl'), args=args)
        self.model.set_eval()
        self.model.to(self.device).eval()
        self.de = DEResNet(num_in_ch=3, num_degradation=2)
        self.de.load_model(str(root/'weights/de_net.pth'))
        self.de.to(self.device).eval()
        self.color_fix, self.to_pil, self.to_tensor = wavelet_color_fix, ToPILImage(), ToTensor()
        self.parameters = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.de.parameters())

    @torch.inference_mode()
    def __call__(self, low):
        h, w = low.shape[-2:]
        resized = F.interpolate(low, size=(h*4,w*4), mode='bilinear', align_corners=False).contiguous()
        normalized = (resized*2-1).clamp(-1,1)
        normalized = F.pad(normalized, (0, math.ceil(w*4/64)*64-w*4, 0, math.ceil(h*4/64)*64-h*4), mode='reflect')
        prediction = self.model(normalized, self.de(low),
            pos_prompt=['A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.'],
            neg_prompt=['oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth'])
        # Preserve the recommended author's PIL quantization and wavelet color fix.
        image = self.to_pil((prediction[:, :, :h*4, :w*4]*.5+.5)[0].cpu())
        image = self.color_fix(image, self.to_pil(resized[0].cpu()))
        return self.to_tensor(image).unsqueeze(0)


class ResShift:
    def __init__(self, root, task, device='cuda'):
        source = root/'source'
        sys.path.insert(0, str(source))
        from omegaconf import OmegaConf
        from utils.util_common import instantiate_from_config
        from utils.util_net import reload_model
        config_name = 'bicx4_swinunet_lpips.yaml' if task == 'bicubic' else 'realsr_swinunet_realesrgan256_journal.yaml'
        config = OmegaConf.load(source/'configs'/config_name)
        self.config = config
        self.device = torch.device(device)
        self.diffusion = instantiate_from_config(config.diffusion)
        self.model = instantiate_from_config(config.model).to(self.device).eval()
        checkpoint = root/('resshift_bicsrx4_s4.pth' if task == 'bicubic' else 'resshift_realsrx4_s4_v3.pth')
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        reload_model(self.model, state.get('state_dict',state))
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device).eval()
        state = torch.load(root/'autoencoder_vq_f4.pth', map_location=self.device, weights_only=True)
        reload_model(self.autoencoder, state.get('state_dict',state))
        self.model.requires_grad_(False)
        self.autoencoder.requires_grad_(False)
        self.parameters = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.autoencoder.parameters())

    @torch.inference_mode()
    def __call__(self, low):
        # Author sample_func's padding, conditioning, sampling and cropping. The
        # fixed cohort fits one tile, so no image-file loader or chopping is used.
        h,w = low.shape[-2:]
        offset = self.config.model.params.get('lq_size',64)
        y = F.pad(low*2-1,(0,math.ceil(w/offset)*offset-w,0,math.ceil(h/offset)*offset-h),mode='reflect')
        with (torch.autocast('cuda',dtype=torch.float16) if self.device.type=='cuda' else nullcontext()):
            result = self.diffusion.p_sample_loop(y=y,model=self.model,first_stage_model=self.autoencoder,
                noise=None,noise_repeat=False,clip_denoised=False,denoised_fn=None,model_kwargs={'lq':y},progress=False)
        return (result[:,:,:h*4,:w*4].clamp(-1,1)*.5+.5).float()


def load(root, name, device='cuda'):
    if device not in ('cuda', 'cpu'):
        raise ValueError('unsupported SR oracle device')
    return S3Diff(root, device) if name=='s3diff' else ResShift(root, 'bicubic' if name=='resshift-bicubic' else 'real', device)
