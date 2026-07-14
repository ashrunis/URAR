# -*- coding:utf-8 -*-

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv as DOSSAsymm3dSpconv
from network.segmentator_3d_asymm_spconv_fr import Asymm_3d_spconv as FRAsymm3dSpconv
from network.segmentator_3d_asymm_spconv_ugfa import Asymm_3d_spconv as UGFAAsymm3dSpconv
from network.segmentator_3d_asymm_spconv_fr_ugfa import Asymm_3d_spconv as FRUGFAAsymm3dSpconv
from network.cylinder_fea_generator import cylinder_fea
import network.ptv3_spconv_3d


MODEL_VARIANTS = {
    "doss": DOSSAsymm3dSpconv,
    "fr": FRAsymm3dSpconv,
    "ugfa": UGFAAsymm3dSpconv,
    "fr_ugfa": FRUGFAAsymm3dSpconv,
    "ptv3": None,
    "ptv3_doss": None,
}


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']
    model_variant = str(model_config.get("model_variant", "fr_ugfa")).lower()
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model_variant='{model_variant}'. "
            f"Expected one of {sorted(MODEL_VARIANTS)}."
        )
    if model_variant in {"ptv3", "ptv3_doss"}:
        ptv3_class = "ptv3_doss_asym" if model_variant == "ptv3_doss" else "ptv3_asym"
        return get_model_class(ptv3_class)(model_config)

    segmentator_class = MODEL_VARIANTS[model_variant]

    cylinder_3d_spconv_seg = segmentator_class(
        output_shape=output_shape,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
