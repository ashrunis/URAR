# -*- coding:utf-8 -*-

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea
import network.ptv3_native


CYLINDER_VARIANTS = {
    "doss": {"use_arcface": False, "use_ugfa": False},
    "fr": {"use_arcface": True, "use_ugfa": False},
    "ugfa": {"use_arcface": False, "use_ugfa": True},
    "fr_ugfa": {"use_arcface": True, "use_ugfa": True},
}
MODEL_VARIANTS = set(CYLINDER_VARIANTS) | {"ptv3_native"}


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
    if model_variant == "ptv3_native":
        return get_model_class("ptv3_native_asym")(model_config)

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class,
        **CYLINDER_VARIANTS[model_variant])

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
