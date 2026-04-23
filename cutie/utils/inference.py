from horizon_tc_ui.hb_runtime import HBRuntime
import torch
import numpy as np
import os
def onnx_infer(input: dict , model_name:str):
    model_path = {'image_encoder':"/open_explorer/moka/cutie_onnx_400_624/image_encoder_400_624_sim.onnx",
                  'segment':"/open_explorer/moka/cutie_onnx_400_624/segment_400_624_sim.onnx",
                  'encode_mask':"/open_explorer/moka/cutie_onnx_400_624/encode_mask_400_624_sim.onnx",
                  'readout_memory':"/open_explorer/moka/cutie_onnx_400_624/read_memory_400_624_debug_clip_version.onnx"}
    input_ = {}
    for name , data in input.items():
        input_[name] = data.numpy()
    sess = HBRuntime(model_path[model_name])
    input_names = sess.input_names # list
    output_names = sess.output_names # list
    output = sess.run(output_names, input_)
    output_ = []
    for i in range(len(output)):
        output_.append(torch.from_numpy(output[i]))
    return output_

def hbm_infer(input:dict, model_name:str):
    # model_path = {'image_encoder':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/image_encoder/rgb_version/int16_v37/image_encoder_400_624_sim_rgb_int16_v37_quantized_model.bc",
    #               'segment':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/segment/rgb_version/int16_v37/segment_400_624_sim_rgb_int16_v37_quantized_model.bc",
    #               'encode_mask':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/encode_mask/rgb_version/int16_v37/encode_mask_400_624_rgb_int16_v37_quantized_model.bc",
    #               'readout_memory':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/read_memory/rgb_version/debug_clip_with_max/read_memory_400_624_rgb_debug_clip_with_max_int16_quantized_model.bc"} #v37
    # model_path = {'image_encoder':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/image_encoder/rgb_version/image_encoder_400_624_sim_rgb_int8.hbm",
    #               'segment':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/segment/rgb_version/int16_v37/segment_400_624_sim_rgb_int16_v37.hbm",
    #               'encode_mask':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/encode_mask/rgb_version/int16_v37/encode_mask_400_624_rgb_int16_v37.hbm",
    #               'readout_memory':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/read_memory/rgb_version/debug_clip_with_max/read_memory_400_624_rgb_debug_clip_with_max_int16.hbm"} #v37
    model_path = {'image_encoder':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/image_encoder/rgb_version/int16_v37_no_padding/image_encoder_400_624_sim_rgb_int16_v37_nopadding_quantized_model.bc",
                  'segment':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/segment/rgb_version/int16_v37_no_padding/segment_400_624_sim_rgb_int16_v37_nopadding_quantized_model.bc",
                  'encode_mask':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/encode_mask/rgb_version/int16_v37_no_padding/encode_mask_400_624_rgb_int16_v37_nopadding_quantized_model.bc",
                  'readout_memory':"/open_explorer/moka/cutie_onnx_400_624/quantized_model/read_memory/rgb_version/debug_clip_with_max_no_padding/read_memory_400_624_rgb_debug_clip_with_max_int16_nopadding_quantized_model.bc"} #v37
    input_ = {}
    for name , data in input.items():
        input_[name] = data.numpy()
    sess = HBRuntime(model_path[model_name])
    input_names = sess.input_names # list
    output_names = sess.output_names # list
    output = sess.run(output_names, input_)
    output_ = []
    for i in range(len(output)):
        output_.append(torch.from_numpy(output[i]))
    return output_

def save2txt(tensor, save_root, name):
    os.makedirs(save_root, exist_ok=True)
    path = os.path.join(save_root,name)
    with open(path,"w") as f:
        for data in list(tensor.flatten()):
            f.write(f'{data.item()}\n')
    print(f'Tensor has been saved to {path}')

