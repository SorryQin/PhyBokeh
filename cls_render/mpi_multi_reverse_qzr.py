# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import cupy
import re
import time
import cv2
import numpy as np # Added numpy import
import math
import os
from PIL import Image

kernel_Render_updateOutput = '''
extern "C" __global__ void kernel_Render_updateOutput(
    const int n,
    const int samples,
    const float* xs,
    const float* ys,
    const float* K,
    const float* depth_focus,
    const float* Eta,
    const float* images,
    const float* alphas,
    const float* coffs,
    const float* ellipticity_map, 
    const float* angle_map,       
    float* bokeh_cum,
    float* weight_cum
) {
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        // Calculate current pixel coordinates
        const int b = (idx / SIZE_3(bokeh_cum) / SIZE_2(bokeh_cum)) % SIZE_0(bokeh_cum);
        const int y = (idx / SIZE_3(bokeh_cum)) % SIZE_2(bokeh_cum);
        const int x = idx % SIZE_3(bokeh_cum);

        // [NEW] Get PSF shape params for current pixel
        // Assume map dim is (B, 1, H, W), using OFFSET_4 macro
        const float elip = ellipticity_map[OFFSET_4(ellipticity_map, b, 0, y, x)]; 
        const float ang  = angle_map[OFFSET_4(angle_map, b, 0, y, x)];

        // [NEW] Pre-calculate rotation matrix params
        float cos_a, sin_a;
        sincosf(ang, &sin_a, &cos_a);

        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        float weight = 0.0f;

        const float radius = K[b];
        const float zf = depth_focus[b];
        const float eta = fabsf(Eta[b]);
        const float df = 1.0f / zf;

        for (int i = 0; i < samples; ++i) {
            // [NEW] Core Logic: Rotate + Scale the sampling point to simulate Elliptical PSF
            // 1. Rotate alignment
            float u = xs[i] * cos_a - ys[i] * sin_a;
            float v = xs[i] * sin_a + ys[i] * cos_a;
            
            // 2. Apply ellipticity compression (compress v-axis)
            v *= elip;

            // 3. [MODIFIED] Rotate back
            float xs_eff = u * cos_a - v * sin_a; 
            float ys_eff = u * sin_a + v * cos_a;

            const float df_ = df;

            float alpha_prev = 1.0f;
            for (int l = 0; l < SIZE_1(images); ++l) {
                const float A = coffs[OFFSET_3(coffs, b, l, 0)];
                const float B = coffs[OFFSET_3(coffs, b, l, 1)];
                const float C = coffs[OFFSET_3(coffs, b, l, 2)];

                // [MODIFIED] Use transformed xs_eff, ys_eff
                const float tmp = (1.0f - A * x - B * y - C * df_) / 
                                  (A * xs_eff + B * ys_eff + C / radius + 1e-5f);
                
                const float x_map = x + tmp * xs_eff;
                const float y_map = y + tmp * ys_eff;

                if (x_map >= 0.0f && x_map < SIZE_3(bokeh_cum) && 
                    y_map >= 0.0f && y_map < SIZE_2(bokeh_cum)) {
                    
                    const int x1 = static_cast<int>(x_map);
                    const int x2 = x1 + 1;
                    const int y1 = static_cast<int>(y_map);
                    const int y2 = y1 + 1;

                    const int x2_clamp = (x2 >= SIZE_3(bokeh_cum)) ? SIZE_3(bokeh_cum) - 1 : x2;
                    const int y2_clamp = (y2 >= SIZE_2(bokeh_cum)) ? SIZE_2(bokeh_cum) - 1 : y2;

                    const float f1 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y1, x1)];
                    const float f2 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y1, x2_clamp)];
                    const float f3 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y2_clamp, x1)];
                    const float f4 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y2_clamp, x2_clamp)];
                    const float f12 = f1 + f2;
                    const float f34 = f3 + f4;
                    const float alpha_curr = (y2 - y_map) * f12 + (y_map - y1) * f34;

                    if (alpha_curr < 0.01f) {
                        if (l == SIZE_1(images) - 1) {
                            weight += alpha_prev;
                        }
                        continue;
                    }

                    const float f12_r = f1 * images[OFFSET_5(images, b, l, 0, y1, x1)] + 
                                        f2 * images[OFFSET_5(images, b, l, 0, y1, x2_clamp)];
                    const float f34_r = f3 * images[OFFSET_5(images, b, l, 0, y2_clamp, x1)] + 
                                        f4 * images[OFFSET_5(images, b, l, 0, y2_clamp, x2_clamp)];
                    sum_r += ((y2 - y_map) * f12_r + (y_map - y1) * f34_r) * alpha_prev;

                    const float f12_g = f1 * images[OFFSET_5(images, b, l, 1, y1, x1)] + 
                                        f2 * images[OFFSET_5(images, b, l, 1, y1, x2_clamp)];
                    const float f34_g = f3 * images[OFFSET_5(images, b, l, 1, y2_clamp, x1)] + 
                                        f4 * images[OFFSET_5(images, b, l, 1, y2_clamp, x2_clamp)];
                    sum_g += ((y2 - y_map) * f12_g + (y_map - y1) * f34_g) * alpha_prev;

                    const float f12_b = f1 * images[OFFSET_5(images, b, l, 2, y1, x1)] + 
                                        f2 * images[OFFSET_5(images, b, l, 2, y1, x2_clamp)];
                    const float f34_b = f3 * images[OFFSET_5(images, b, l, 2, y2_clamp, x1)] + 
                                        f4 * images[OFFSET_5(images, b, l, 2, y2_clamp, x2_clamp)];
                    sum_b += ((y2 - y_map) * f12_b + (y_map - y1) * f34_b) * alpha_prev;

                    sum_a += alpha_curr * alpha_prev;
                    weight += alpha_curr * alpha_prev;

                    if (alpha_curr > 0.99f) {
                        break;
                    } else {
                        alpha_prev *= (1.0f - alpha_curr);
                    }
                }
            }
        }

        bokeh_cum[OFFSET_4(bokeh_cum, b, 0, y, x)] = sum_r;
        bokeh_cum[OFFSET_4(bokeh_cum, b, 1, y, x)] = sum_g;
        bokeh_cum[OFFSET_4(bokeh_cum, b, 2, y, x)] = sum_b;
        bokeh_cum[OFFSET_4(bokeh_cum, b, 3, y, x)] = sum_a;
        weight_cum[OFFSET_4(weight_cum, b, 0, y, x)] = weight;
    }
}
'''


def _cupy_kernel_compile(str_function: str, variables: dict) -> str:
    kernel_code = globals()[str_function]
    while True:
        match = re.search(r'(SIZE_)([0-5])(\()([^\)]*)(\))', kernel_code)
        if not match: break
        dim = int(match.group(2))
        tensor_name = match.group(4)
        tensor_size = variables[tensor_name].size()
        kernel_code = kernel_code.replace(match.group(), str(tensor_size[dim]))
    while True:
        match = re.search(r'(OFFSET_)([0-5])(\()([^\)]+)(\))', kernel_code)
        if not match: break
        dim_count = int(match.group(2))
        args = [arg.strip() for arg in match.group(4).split(',')]
        tensor_name = args[0]
        tensor_strides = variables[tensor_name].stride()
        offset_parts = [f'(({args[dim+1]}) * {tensor_strides[dim]})' for dim in range(dim_count)]
        kernel_code = kernel_code.replace(match.group(), f'({"+".join(offset_parts)})')
    return kernel_code

@cupy.memoize(for_each_device=True)
def _cupy_kernel_launch(str_function: str, kernel_code: str) -> cupy.cuda.function.Function:
    return cupy.RawModule(code=kernel_code).get_function(str_function)


class _RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        images: torch.Tensor,
        alphas: torch.Tensor,
        coffs: torch.Tensor,
        K: torch.Tensor or float,
        depth_focus: torch.Tensor or float,
        Eta: torch.Tensor or float,
        ellipticity_map: torch.Tensor, # [NEW]
        angle_map: torch.Tensor,       # [NEW]
        samples_per_side: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _, H, W = images.shape
        bokeh_cum = torch.zeros(batch, 4, H, W, device=images.device, dtype=torch.float32)
        weight_cum = torch.zeros(batch, 1, H, W, device=images.device, dtype=torch.float32)

        def _to_batch_tensor(param, dtype=torch.float32) -> torch.Tensor:
            if isinstance(param, (int, float)):
                return torch.tensor(param, device=images.device, dtype=dtype).repeat(batch)
            return param[:, 0, 0, 0].to(dtype=dtype) if param.ndim == 4 else param.to(dtype=dtype)

        K = _to_batch_tensor(K)
        depth_focus = _to_batch_tensor(depth_focus)
        Eta = _to_batch_tensor(Eta)

        # [NEW] 确保输入 Map 的连续性
        ellipticity_map = ellipticity_map.contiguous().float()
        angle_map = angle_map.contiguous().float()

        x_lin = torch.linspace(-1.0, 1.0, steps=samples_per_side, device=images.device)
        y_lin = torch.linspace(-1.0, 1.0, steps=samples_per_side, device=images.device)
        X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
        circle_mask = (X**2 + Y**2) <= 1.0
        xs = X[circle_mask]
        ys = Y[circle_mask]
        samples = len(xs)

        if images.is_cuda:
            n = weight_cum.nelement()
            # [MODIFIED] 传入新增的 map 变量用于编译
            kernel_code = _cupy_kernel_compile('kernel_Render_updateOutput', {
                'samples': samples, 'xs': xs, 'ys': ys, 'K': K, 'depth_focus': depth_focus,
                'Eta': Eta, 'images': images, 'alphas': alphas, 'coffs': coffs,
                'ellipticity_map': ellipticity_map, 'angle_map': angle_map, # [NEW]
                'bokeh_cum': bokeh_cum, 'weight_cum': weight_cum
            })
            kernel_func = _cupy_kernel_launch('kernel_Render_updateOutput', kernel_code)
            kernel_func(
                grid=tuple([(n + 511) // 512, 1, 1]),
                block=tuple([512, 1, 1]),
                args=(
                    cupy.int32(n),
                    cupy.int32(samples),
                    xs.data_ptr(),
                    ys.data_ptr(),
                    K.data_ptr(),
                    depth_focus.data_ptr(),
                    Eta.data_ptr(),
                    images.data_ptr(),
                    alphas.data_ptr(),
                    coffs.data_ptr(),
                    ellipticity_map.data_ptr(), # [NEW]
                    angle_map.data_ptr(),       # [NEW]
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr()
                )
            )
        else:
            raise NotImplementedError("仅支持CUDA设备加速渲染")
        return bokeh_cum, weight_cum

def render_bokeh(
    images, alphas, coffs, K, depth_focus, Eta, 
    ellipticity_map, angle_map, # [NEW]
    samples_per_side: int = 51
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RenderFunction.apply(images, alphas, coffs, K, depth_focus, Eta, ellipticity_map, angle_map, samples_per_side)

class MPIRendererRT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        images: torch.Tensor,
        alphas: torch.Tensor,
        coffs: torch.Tensor,
        K: torch.Tensor or float,
        depth_focus: torch.Tensor or float,
        Eta: torch.Tensor or float,
        ellipticity_map: torch.Tensor, # [NEW] (B, 1, H, W) range [0, 1]
        angle_map: torch.Tensor,       # [NEW] (B, 1, H, W) radians
        samples_per_side: int = 51
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return render_bokeh(images, alphas, coffs, K, depth_focus, Eta, ellipticity_map, angle_map, samples_per_side)

# ----------------- 测试脚本 (更新以生成测试用的 Map) -----------------


if __name__ == '__main__':
    import os
    from PIL import Image
    
    renderer = MPIRendererRT().cuda()

    # ==================== 参数设置 ====================
    H, W = 512, 512                    
    samples_per_side = 121             
    K = 5.0                           
    depth_focus = 1 / 1.0                  
    Eta = 0.001                          

    # ==================== 1. 加载前景人物（带Alpha） ====================
    # foreground_path = "/data/juicefs_sharing_data/11186867/bokehdiff-master/temp_data_on_the_fly/fg/003_0_transparent.png"     # ← 放一张带透明背景的人物PNG（推荐512x512或更大）
    foreground_path = "/data/juicefs_sharing_data/11186867/bokehdiff-master/temp_data_on_the_fly/fg/975_1_transparent.png"
    if not os.path.exists(foreground_path):
        raise FileNotFoundError(f"请放入前景图: {foreground_path} （带透明通道的PNG）")

    fg_pil = Image.open(foreground_path).convert("RGBA")
    fg_pil = fg_pil.resize((W, H), Image.LANCZOS)
    fg_np = np.array(fg_pil).astype(np.float32) / 255.0          # (H,W,4)
    fg_rgb = fg_np[..., :3]                                      # (H,W,3)
    fg_alpha = fg_np[..., 3:4]                                   # (H,W,1)

    # ==================== 2. 加载背景图 ====================
    # background_path = "/data/juicefs_sharing_data/11186867/bokehdiff-master/temp_data_on_the_fly/bg/00003_0_bg.png"     # ← 随便放一张好看的背景图
    background_path = "/data/juicefs_sharing_data/11186867/bokehdiff-master/temp_data_on_the_fly/bg/7c1bff6d83a5b03.jpg"

    bg_pil = Image.open(background_path).convert("RGB")
    bg_pil = bg_pil.resize((W, H), Image.LANCZOS)
    bg_np = np.array(bg_pil).astype(np.float32) / 255.0          # (H,W,3)

    # ==================== 3. 构造 MPI（只有2层） ====================
    batch_size = 1
    num_layers = 2

    images = torch.zeros((batch_size, num_layers, 3, H, W), device='cuda', dtype=torch.float32)
    alphas = torch.zeros((batch_size, num_layers, 1, H, W), device='cuda', dtype=torch.float32)
    coffs  = torch.zeros((batch_size, num_layers, 3),     device='cuda', dtype=torch.float32)

    # --- Layer 0: 前景人物（合焦）---
    images[0, 0] = torch.from_numpy(fg_rgb.transpose(2,0,1)).unsqueeze(0)   # (1,3,H,W)
    alphas[0, 0] = torch.from_numpy(fg_alpha.transpose(2,0,1)).unsqueeze(0)
    coffs[0, 0]  = torch.tensor([0.0, 0.0, 1.0/1.0])   # 深度 = 1.0（对焦平面）

    # --- Layer 1: 背景（强烈离焦）---
    images[0, 1] = torch.from_numpy(bg_np.transpose(2,0,1)).unsqueeze(0)
    alphas[0, 1] = 1.0
    background_depth = 8.0    # 你设置的背景深度                                                  # 背景完全不透明
    coffs[0, 1]  = torch.tensor([0.0, 0.0, 1.0/background_depth])                   # 深度 ≈ 10.0（很远）

    # ==================== 4. 生成椭圆猫眼地图（和原来一样） ====================
    # grid_y, grid_x = torch.meshgrid(
    #     torch.arange(H, device='cuda', dtype=torch.float32),
    #     torch.arange(W, device='cuda', dtype=torch.float32),
    #     indexing='ij'
    # )
    # cy, cx = H/2.0, W/2.0
    # delta_y = grid_y - cy
    # delta_x = grid_x - cx

    # # 角度图：所有椭圆长轴指向画面中心（经典猫眼效果）
    # angle_map = torch.atan2(delta_y, delta_x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # # 椭圆率图：越靠边缘越扁（0.3~1.0 之间很自然）
    # dist = torch.sqrt(delta_x**2 + delta_y**2)
    # max_dist = torch.hypot(torch.tensor(H/2), torch.tensor(W/2))
    # ellipticity_map = 1.0 - 0.55 * (dist / max_dist)      # 中心≈1.0（圆），边缘≈0.45（很扁）
    # ellipticity_map = ellipticity_map.clamp(min=0.45, max=1.0)
    # ellipticity_map = ellipticity_map.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # # ================= 高级：光斑形状随深度变化（真实镜头行为）================
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')
    delta_y = grid_y - H/2.0
    delta_x = grid_x - W/2.0

    # 1. 基础猫眼（位置决定）
    angle_map = (torch.atan2(delta_y, delta_x) + math.pi/2) % (2*math.pi)
    angle_map = angle_map.unsqueeze(0).unsqueeze(0)

    dist = torch.sqrt(delta_x**2 + delta_y**2)
    max_dist = dist.max()
    base_ellipticity = 1.0 - 0.55 * (dist / max_dist)
    base_ellipticity = base_ellipticity.clamp(min=0.22)

    # 2. 离焦圆化因子（深度决定）← 修复版！
    zf = depth_focus  # 1.0
    defocus_amount = abs(background_depth - zf) / zf  # = 2.0

    # 离焦严重 → 光斑应该更圆
    roundness = 1.0 / (1.0 + 6.0 * defocus_amount)   # ≈ 0.77
    roundness = max(roundness, 0.6)  # 保证不会太扁

    ellipticity_map = base_ellipticity * roundness
    ellipticity_map = ellipticity_map.clamp(min=0.35, max=1.0).unsqueeze(0).unsqueeze(0)
    # ==================== 5. 渲染！ ====================
    print(f"开始渲染 {H}x{W} 真实椭圆猫眼虚化（前景合焦 + 背景大光斑）...")
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        bokeh_cum, weight_cum = renderer(
            images, alphas, coffs,
            K, depth_focus, Eta,
            ellipticity_map, angle_map,
            samples_per_side
        )
        result = bokeh_cum[:, :3] / weight_cum.clamp(min=1e-6)

    torch.cuda.synchronize()
    print(f"渲染完成！耗时 {time.time() - start:.3f} 秒")

    # ==================== 6. 保存结果 ====================
    result_np = result[0].permute(1,2,0).cpu().numpy()
    result_np = np.clip(result_np, 0, 1)
    result_uint8 = (result_np * 255).astype(np.uint8)

    fg_rgb_np = fg_rgb.copy()           # (H, W, 3)
    fg_alpha_np = fg_alpha[..., 0]      # (H, W)
    bg_np_copy = bg_np.copy()

    # Alpha blending: out = fg * alpha + bg * (1 - alpha)
    aif_composite = fg_rgb_np * fg_alpha_np[..., None] + bg_np_copy * (1 - fg_alpha_np[..., None])
    aif_composite = np.clip(aif_composite, 0, 1)
    aif_uint8 = (aif_composite * 255).astype(np.uint8)

    output_aif = "portrait_AIF_composite.jpg"  # AIF = Alpha Image Fusion
    cv2.imwrite(output_aif, cv2.cvtColor(aif_uint8, cv2.COLOR_RGB2BGR))

    output_path = "portrait_elliptical_bokeh_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR))
    print(f"结果已保存：{output_path}")
