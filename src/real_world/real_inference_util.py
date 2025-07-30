from typing import Dict, Callable, Tuple
import numpy as np
from src.common.cv2_util import get_image_transform
from src.common.pytorch_util import dict_apply

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            if this_imgs_in.ndim == 4:
                t, ci, hi, wi = this_imgs_in.shape
            elif this_imgs_in.ndim == 5:
                B, t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            if out_imgs.ndim == 5:
                # THWC to TCHW
                obs_dict_np[key] = np.moveaxis(out_imgs,-1,2)
            else:
                obs_dict_np[key] = out_imgs
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
    
    base_obs_dict_np, edit_obs_dict_np = None, None
    if (this_data_in.ndim == 2):
        base_obs_dict_np = dict_apply(obs_dict_np, lambda x: np.expand_dims(x, axis=0))
        edit_obs_dict_np = obs_dict_np
    else:
        edit_obs_dict_np = dict_apply(obs_dict_np, lambda x: np.squeeze(x, axis=1) if x.shape[1]==1 else np.mean(x, axis=1))
        base_obs_dict_np = obs_dict_np
    return base_obs_dict_np, edit_obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
