import os
from tqdm.auto import tqdm
from opt import config_parser
from models.tensoRF import TensorVMSplit
import time
import csv


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import datetime

from dataLoader import dataset_dict
import sys

def create_training_report(df, logfolder):
    """Create detailed HTML training report with metrics table"""
    try:
        # Calculate summary statistics
        final_iter = df['iter'].iloc[-1]
        final_psnr = df['psnr'].iloc[-1] if 'psnr' in df.columns else 0
        final_loss = df['total_loss'].iloc[-1] if 'total_loss' in df.columns else 0
        max_psnr = df['psnr'].max() if 'psnr' in df.columns else 0
        min_loss = df['total_loss'].min() if 'total_loss' in df.columns else 0
        
        # Cross-scale adaptation summary
        if 'prop_hr' in df.columns and not df['prop_hr'].isna().all():
            avg_prop_hr = df['prop_hr'].mean()
            avg_prop_mr = df['prop_mr'].mean()
            avg_prop_lr = df['prop_lr'].mean()
        else:
            avg_prop_hr = avg_prop_mr = avg_prop_lr = 0
        
        # Performance metrics
        avg_throughput = df['throughput_rays_per_s'].mean() if 'throughput_rays_per_s' in df.columns else 0
        max_gpu_mem = df['gpu_mem_mb'].max() if 'gpu_mem_mb' in df.columns else 0
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FrugalNeRF Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .good {{ color: green; font-weight: bold; }}
        .poor {{ color: red; font-weight: bold; }}
        .normal {{ color: blue; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .cross-scale {{ background-color: #f8f8e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 FrugalNeRF Training Report</h1>
        <p><strong>Training completed at iteration:</strong> {final_iter}</p>
        <p><strong>Report generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Training Summary</h2>
        <table class="metrics-table">
            <tr><th>Metric</th><th>Final Value</th><th>Best Value</th><th>Status</th></tr>
            <tr>
                <td>PSNR (dB)</td>
                <td>{final_psnr:.2f}</td>
                <td>{max_psnr:.2f}</td>
                <td class="{'good' if final_psnr > 20 else 'poor'}">{'✅ Good' if final_psnr > 20 else '❌ Poor'}</td>
            </tr>
            <tr>
                <td>Total Loss</td>
                <td>{final_loss:.6f}</td>
                <td>{min_loss:.6f}</td>
                <td class="{'good' if final_loss < 0.1 else 'poor'}">{'✅ Good' if final_loss < 0.1 else '❌ Poor'}</td>
            </tr>
            <tr>
                <td>Average Throughput (rays/s)</td>
                <td>{avg_throughput:.0f}</td>
                <td>-</td>
                <td class="{'good' if avg_throughput > 1000 else 'poor'}">{'✅ Good' if avg_throughput > 1000 else '❌ Slow'}</td>
            </tr>
            <tr>
                <td>Peak GPU Memory (MB)</td>
                <td>{max_gpu_mem:.0f}</td>
                <td>-</td>
                <td class="{'good' if max_gpu_mem < 8000 else 'poor'}">{'✅ Normal' if max_gpu_mem < 8000 else '⚠️ High'}</td>
            </tr>
        </table>
    </div>
    
    <div class="cross-scale">
        <h2>🔄 Cross-Scale Adaptation</h2>
        <table class="metrics-table">
            <tr><th>Resolution</th><th>Average Proportion (%)</th><th>Status</th><th>Description</th></tr>
            <tr>
                <td>High Resolution</td>
                <td>{avg_prop_hr:.1f}</td>
                <td class="{'good' if avg_prop_hr > 30 else 'poor'}">{'✅ Active' if avg_prop_hr > 30 else '❌ Low'}</td>
                <td>Highest quality rendering</td>
            </tr>
            <tr>
                <td>Mid Resolution</td>
                <td>{avg_prop_mr:.1f}</td>
                <td class="{'good' if avg_prop_mr > 25 else 'poor'}">{'✅ Active' if avg_prop_mr > 25 else '❌ Low'}</td>
                <td>Balanced quality/speed</td>
            </tr>
            <tr>
                <td>Low Resolution</td>
                <td>{avg_prop_lr:.1f}</td>
                <td class="{'good' if avg_prop_lr > 20 else 'poor'}">{'✅ Active' if avg_prop_lr > 20 else '❌ Low'}</td>
                <td>Fastest rendering</td>
            </tr>
        </table>
    </div>
    
    <h2>📈 Training Progress</h2>
    <p>Detailed metrics are available in the CSV file: <code>metrics.csv</code></p>
    <p>Cross-scale adaptation plot: <code>cross_scale_adaptation.png</code></p>
    <p>Loss curves plot: <code>cross_scale_losses.png</code></p>
    
    <h2>💡 Recommendations</h2>
    <ul>
        <li>{"✅ Training completed successfully!" if final_psnr > 20 else "⚠️ Consider training longer or adjusting hyperparameters"}</li>
        <li>{"✅ Cross-scale adaptation is working well" if avg_prop_hr > 30 else "⚠️ High-res proportion is low, consider adjusting self_depth_weight"}</li>
        <li>{"✅ Good training speed" if avg_throughput > 1000 else "⚠️ Training is slow, consider reducing batch size or image resolution"}</li>
        <li>{"✅ GPU memory usage is normal" if max_gpu_mem < 8000 else "⚠️ High GPU memory usage, consider reducing batch size"}</li>
    </ul>
</body>
</html>
        """
        
        # Save HTML report
        with open(os.path.join(logfolder, 'training_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Training report saved: {os.path.join(logfolder, 'training_report.html')}")
        
    except Exception as e:
        print(f"⚠️ Error creating training report: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]



class PatchSampler:
    def __init__(self, total, batch, W, H, patch_size=2):
        self.total = total
        self.frame_num = total // (W*H)
        self.patch_size = patch_size
        self.total_patch = (W-patch_size+1) * (H-patch_size+1)
        self.total_id = self.total_patch * self.frame_num
        self.curr = self.total_id 
        self.W = W
        self.H = H
        self.frame_size = W*H
        self.batch = batch
        print(f"frame_num: {self.frame_num} frame_size: {self.frame_size} patch_size: {self.patch_size}")

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total_id:
            self.ids = torch.LongTensor(np.random.permutation(self.total_id))
            self.curr = 0
        
        pixel = self.ids[self.curr:self.curr+self.batch]
        frame_index = pixel // self.total_patch
        pixel_x = pixel % (self.W - self.patch_size + 1)
        pixel_y = (pixel // (self.W - self.patch_size + 1)) % (self.H - self.patch_size + 1)

        # Generate all indices for the patch
        patch_indices = []
        for dy in range(self.patch_size):
            for dx in range(self.patch_size):
                id_patch = (pixel_x + dx) + (pixel_y + dy) * self.W + frame_index * self.frame_size
                patch_indices.append(id_patch)
        
        return torch.cat(patch_indices, dim=0)



@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, frame_num=args.test_frame_num)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test,_,_ = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, frame_num=args.train_frame_num)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, frame_num=args.test_frame_num)
    novel_dataset = dataset(args.datadir, split='novel', downsample=args.downsample_train, is_stack=False, frame_num=args.train_frame_num)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # metrics csv for analysis/reporting
    metrics_csv_path = os.path.join(logfolder, 'metrics.csv')
    csv_exists = os.path.exists(metrics_csv_path)
    metrics_csv_file = open(metrics_csv_path, 'a', newline='')
    metrics_writer = csv.writer(metrics_csv_file)
    if not csv_exists:
        metrics_writer.writerow([
            'iter','total_loss','loss_hr','loss_mr','loss_lr','depth_loss','psnr','ssim','lpips',
            'prop_hr','prop_mr','prop_lr',
            'throughput_rays_per_s','gpu_mem_mb','iter_time_s','lr','img_W','img_H','focal'
        ])



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    print("Downsampling ratio:", args.down_sampling_ratio)
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    nSamples_MR = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio*args.down_sampling_ratio[0]))
    nSamples_LR = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio*args.down_sampling_ratio[1]))
    

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,down_sampling_ratio = args.down_sampling_ratio,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio,fea2denseAct=args.fea2denseAct,rayMarch_weight_thres=0.0001)
        

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test,SSIMs_test,LPIPSs_test = [],[0],[0],[0]

    allrays, allrgbs, alldepths, alldepthweights = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_depths, train_dataset.all_depth_weights
    all_dense_depths = train_dataset.all_dense_depths
    allrays_real = train_dataset.all_rays_real
    # TODO: parameter for warping
    all_ids, allnearest_ids, all_poses = train_dataset.all_ids, train_dataset.all_nearest_ids, train_dataset.poses #get frame_id, nearest_frame_id, and poses_of_each_frame
    W, H = train_dataset.img_wh
    f = train_dataset.focal
    frameid2_startpoints_in_allray = torch.tensor(train_dataset.frameid2_startpoints_in_allray) # get start position in "allray" for each view

    allrays_novel, all_ids_novel, allnearest_ids_novel, all_poses_novel = novel_dataset.all_rays, novel_dataset.all_ids, novel_dataset.all_nearest_ids, novel_dataset.render_path
    allrays_real_novel = novel_dataset.all_rays_real


    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)

    TV_weight_density, TV_weight_app, TV_weight_color_density = args.TV_weight_density, args.TV_weight_app, args.TV_weight_color_density
    tvreg = TVLoss()
    colortvreg = colorTVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app} color_density: {TV_weight_color_density}")

    sparse_depth_weight = args.Sparse_Depth_weight
    print(f"initial sparse_depth_weight: {sparse_depth_weight}")

    self_depth_weight = args.Self_Depth_weight
    print(f"initial self_depth_weight: {self_depth_weight}")

    dist_loss_weight = args.Dist_weight
    print(f"initial dist_loss_weight: {dist_loss_weight}")

    depth_smooth_weight = args.Depth_Smooth_weight
    print(f"initial depth_smooth_weight: {depth_smooth_weight}")

    mr_color_weight = args.MR_color_weight
    print(f"initial mr_color_weight: {mr_color_weight}")

    lr_color_weight = args.LR_color_weight
    print(f"initial lr_color_weight: {lr_color_weight}")

    warping_patch_size = args.warping_patch_size
    print(f"warping_patch_size: {warping_patch_size}")

    dense_depth_weight = args.Dense_Depth_weight
    print(f"initial dense_depth_weight: {dense_depth_weight}")
    
    occ_loss_weight = args.Occ_loss_weight
    reg_rate = args.reg_rate
    wb_prior = args.wb_prior
    wb_rate = args.wb_rate


    train_frame_len = len(train_dataset.frame_num)
    novel_frame_len = 1000

    if depth_smooth_weight > 0:
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
        novelSampler = PatchSampler(allrays_novel.shape[0], args.novel_batch_size, W, H)
        train_items = args.batch_size
        batch_size = args.batch_size  + args.novel_batch_size * 4
    else:
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
        novelSampler = SimpleSampler(allrays_novel.shape[0], args.novel_batch_size)
        train_items = args.batch_size
        batch_size = args.batch_size + args.novel_batch_size


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    last_ssim = float('nan')
    last_lpips = float('nan')
    
    # Print initial training configuration
    print("\n" + "="*80)
    print("🚀 FRUGALNERF TRAINING STARTED")
    print("="*80)
    print(f"📁 Dataset: {args.dataset_name}")
    print(f"📁 Data directory: {args.datadir}")
    print(f"📁 Log directory: {logfolder}")
    print(f"🔄 Total iterations: {args.n_iters}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📐 Image resolution: {W}x{H}")
    print(f"🔽 Downsample factor: {args.downsample_train}")
    print(f"🎯 Training frames: {args.train_frame_num}")
    print(f"🧪 Test frames: {args.test_frame_num}")
    print("="*80)
    print()
    for iteration in pbar:
        iter_start = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        ray_idx = trainingSampler.nextids()

        rays_train, rgb_train, depth_train = allrays[ray_idx].to(device), allrgbs[ray_idx].to(device), alldepths[ray_idx].to(device)
        dense_depth_train = all_dense_depths[ray_idx].to(device)
        ids_train, nearest_ids_train = all_ids[ray_idx], allnearest_ids[ray_idx] # get correpsonding pose id for each ray
        c2w_train, nearest_c2w_train = torch.tensor(all_poses[ids_train], dtype=torch.float32).to(device), torch.tensor(all_poses[nearest_ids_train], dtype=torch.float32).to(device) # get correpsonding c2w matrix for each ray

        ray_idx_novel = novelSampler.nextids()
        rays_novel = allrays_novel[ray_idx_novel].to(device)
        ids_novel, nearest_ids_novel = all_ids_novel[ray_idx_novel], allnearest_ids_novel[ray_idx_novel]
        c2w_novel, nearest_c2w_novel = torch.tensor(all_poses_novel[ids_novel][:,:3,:], dtype=torch.float32).to(device), torch.tensor(all_poses[nearest_ids_novel], dtype=torch.float32).to(device)

        rays_train_all = torch.cat([rays_train, rays_novel], dim=0)

        rgb_map, depth_map, weight, m, sigma, rgb_ray = renderer(rays_train_all, tensorf, chunk=batch_size, N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, reso=1)
        rgb_map_MR, depth_map_MR, weight_MR, m_MR, sigma_MR, rgb_ray_MR = renderer(rays_train_all, tensorf, chunk=batch_size, N_samples=nSamples_MR, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, reso=args.down_sampling_ratio[0])
        rgb_map_LR, depth_map_LR, weight_LR, m_LR, sigma_LR, rgb_ray_LR = renderer(rays_train_all, tensorf, chunk=batch_size, N_samples=nSamples_LR, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, reso=args.down_sampling_ratio[1])
        

        prop_hr = float('nan'); prop_mr = float('nan'); prop_lr = float('nan')
        
        # Always compute cross-scale proportions for visualization, even if self_depth_weight = 0
        if True:  # Always compute proportions
            patch_ray_idx_i, patch_mask_i = patchify(ray_idx, H, W, warping_patch_size, train_frame_len, device)
            
            patch_ray_idx_j, patch_mask_j = warping(allrays_real, patch_ray_idx_i, H, W, f, depth_map[:train_items], nearest_c2w_train, frameid2_startpoints_in_allray[nearest_ids_train], patch_mask_i, warping_patch_size, device)
            patch_ray_idx_j_MR, patch_mask_j_MR = warping(allrays_real, patch_ray_idx_i, H, W, f, depth_map_MR[:train_items], nearest_c2w_train, frameid2_startpoints_in_allray[nearest_ids_train], patch_mask_i, warping_patch_size, device)
            patch_ray_idx_j_LR, patch_mask_j_LR = warping(allrays_real, patch_ray_idx_i, H, W, f, depth_map_LR[:train_items], nearest_c2w_train, frameid2_startpoints_in_allray[nearest_ids_train], patch_mask_i, warping_patch_size, device)

            with torch.no_grad():
                mask = patch_mask_i & patch_mask_j
                rgb = allrgbs[patch_ray_idx_i[mask].cpu()].to(device)
                projected_rgb  = allrgbs[patch_ray_idx_j[mask].cpu()].to(device)

                mask_MR = patch_mask_i & patch_mask_j_MR
                rgb_MR = allrgbs[patch_ray_idx_i[mask_MR].cpu()].to(device)
                projected_rgb_MR  = allrgbs[patch_ray_idx_j_MR[mask_MR].cpu()].to(device)

                mask_LR = patch_mask_i & patch_mask_j_LR
                rgb_LR = allrgbs[patch_ray_idx_i[mask_LR].cpu()].to(device)
                projected_rgb_LR  = allrgbs[patch_ray_idx_j_LR[mask_LR].cpu()].to(device)

            reprojection_error = cal_reprojection_error(rgb, projected_rgb, mask, warping_patch_size)
            reprojection_error_MR = cal_reprojection_error(rgb_MR, projected_rgb_MR, mask_MR, warping_patch_size)
            reprojection_error_LR = cal_reprojection_error(rgb_LR, projected_rgb_LR, mask_LR, warping_patch_size)

            # novel view (patch set 1 and use HR render color)
            patch_ray_idx_i_novel, patch_mask_i_novel = patchify(ray_idx_novel, H, W, 1, novel_frame_len, device)
            
            patch_ray_idx_j_novel, patch_mask_j_novel = warping(allrays_real_novel, patch_ray_idx_i_novel, H, W, f, depth_map[train_items:], nearest_c2w_novel, frameid2_startpoints_in_allray[nearest_ids_novel], patch_mask_i_novel, 1, device)
            patch_ray_idx_j_novel_MR, patch_mask_j_novel_MR = warping(allrays_real_novel, patch_ray_idx_i_novel, H, W, f, depth_map_MR[train_items:], nearest_c2w_novel, frameid2_startpoints_in_allray[nearest_ids_novel], patch_mask_i_novel, 1, device)
            patch_ray_idx_j_novel_LR, patch_mask_j_novel_LR = warping(allrays_real_novel, patch_ray_idx_i_novel, H, W, f, depth_map_LR[train_items:], nearest_c2w_novel, frameid2_startpoints_in_allray[nearest_ids_novel], patch_mask_i_novel, 1, device)
            
            with torch.no_grad():
                mask_novel = patch_mask_i_novel & patch_mask_j_novel
                rgb_novel = rgb_map[train_items:][mask_novel]
                projected_rgb_novel  = allrgbs[patch_ray_idx_j_novel[mask_novel].cpu()].to(device)

                mask_MR_novel = patch_mask_i_novel & patch_mask_j_novel_MR
                rgb_MR_novel = rgb_map_MR[train_items:][mask_MR_novel]
                projected_rgb_MR_novel  = allrgbs[patch_ray_idx_j_novel_MR[mask_MR_novel].cpu()].to(device)

                mask_LR_novel = patch_mask_i_novel & patch_mask_j_novel_LR
                rgb_LR_novel = rgb_map[train_items:][mask_LR_novel]
                projected_rgb_LR_novel  = allrgbs[patch_ray_idx_j_novel_LR[mask_LR_novel].cpu()].to(device)
            
            reprojection_error_novel = cal_reprojection_error(rgb_novel, projected_rgb_novel, mask_novel, 1)
            reprojection_error_novel_MR = cal_reprojection_error(rgb_MR_novel, projected_rgb_MR_novel, mask_MR_novel, 1)
            reprojection_error_novel_LR = cal_reprojection_error(rgb_LR_novel, projected_rgb_LR_novel, mask_LR_novel, 1)

            reprojection_error = torch.cat([reprojection_error, reprojection_error_novel])
            reprojection_error_MR = torch.cat([reprojection_error_MR, reprojection_error_novel_MR])
            reprojection_error_LR = torch.cat([reprojection_error_LR, reprojection_error_novel_LR])

        HR_error = (rgb_map[:train_items] - rgb_train)  ** 2
        MR_error = (rgb_map_MR[:train_items] - rgb_train)  ** 2
        LR_error = (rgb_map_LR[:train_items] - rgb_train)  ** 2
        loss = torch.mean(HR_error)
        loss_MR = torch.mean(MR_error)
        loss_LR = torch.mean(LR_error)
        
        # loss
        total_loss = loss * 1.0 + loss_MR * mr_color_weight + loss_LR * lr_color_weight
        mr_color_weight *= lr_factor
        lr_color_weight *= lr_factor
        summary_writer.add_scalar('train/loss2', loss_MR.detach().item(), global_step=iteration)
        summary_writer.add_scalar('train/loss3', loss_LR.detach().item(), global_step=iteration)

        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)

        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            loss_reg_L1 = loss_reg_L1 + tensorf.density_L1_MR() * L1_reg_weight / (args.down_sampling_ratio[0]**2)
            loss_reg_L1 = loss_reg_L1 + tensorf.density_L1_LR() * L1_reg_weight / (args.down_sampling_ratio[1]**2)
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            loss_tv = loss_tv + tensorf.TV_loss_density_MR(tvreg) * TV_weight_density / (args.down_sampling_ratio[0]**2)
            loss_tv = loss_tv + tensorf.TV_loss_density_LR(tvreg) * TV_weight_density / (args.down_sampling_ratio[1]**2)
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)

        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            loss_tv = loss_tv + tensorf.TV_loss_app_MR(tvreg) * TV_weight_app  / (args.down_sampling_ratio[0]**2)
            loss_tv = loss_tv + tensorf.TV_loss_app_LR(tvreg) * TV_weight_app / (args.down_sampling_ratio[1]**2)
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
        
        if TV_weight_color_density>0:
            TV_weight_color_density *= lr_factor
            loss_tv = tensorf.TV_loss_color_aware_density(colortvreg) * TV_weight_color_density
            loss_tv = loss_tv + tensorf.TV_loss_color_aware_density_MR(colortvreg) * TV_weight_color_density / (args.down_sampling_ratio[0]**2)
            loss_tv = loss_tv + tensorf.TV_loss_color_aware_density_LR(colortvreg) * TV_weight_color_density / (args.down_sampling_ratio[1]**2)
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_color_tv', loss_tv.detach().item(), global_step=iteration)
        
        if sparse_depth_weight>0:
            sparse_depth_weight *= lr_factor
            depth_mask = depth_train > 0
            loss_depth = 0
            loss_depth = cal_disparity_loss(depth_map[:train_items][depth_mask], depth_train[depth_mask])*sparse_depth_weight
            loss_depth = loss_depth + cal_disparity_loss(depth_map_MR[:train_items][depth_mask], depth_train[depth_mask])*sparse_depth_weight
            loss_depth = loss_depth + cal_disparity_loss(depth_map_LR[:train_items][depth_mask], depth_train[depth_mask])*sparse_depth_weight
            total_loss = total_loss + loss_depth
            summary_writer.add_scalar('train/reg_depth', loss_depth.detach().item(), global_step=iteration)

        if dist_loss_weight>0:
            dist_loss_weight /= lr_factor
            dist_loss = distortion_loss(weight, m, nSamples) * dist_loss_weight
            dist_loss = dist_loss + distortion_loss(weight_MR, m_MR, nSamples_MR) * dist_loss_weight / (args.down_sampling_ratio[0]**2)
            dist_loss = dist_loss + distortion_loss(weight_LR, m_LR, nSamples_LR) * dist_loss_weight / (args.down_sampling_ratio[1]**2)
            total_loss = total_loss + dist_loss 
            summary_writer.add_scalar('train/dist_loss', dist_loss.detach().item(), global_step=iteration)

        if self_depth_weight>0:
            with torch.no_grad():
                _, min_idx = torch.min(torch.stack([reprojection_error_LR, reprojection_error_MR, reprojection_error]), dim=0)
                HR_mask = (min_idx == 2)  & (reprojection_error < 0.9)
                MR_mask = (min_idx == 1)  & (reprojection_error_MR < 0.9)
                LR_mask = (min_idx == 0)  & (reprojection_error_LR < 0.9)

            # compute cross-scale pseudo-GT proportions
            denom = (HR_mask.numel() if hasattr(HR_mask,'numel') else 0) or 1
            prop_hr = float(HR_mask.float().mean().item()*100.0)
            prop_mr = float(MR_mask.float().mean().item()*100.0)
            prop_lr = float(LR_mask.float().mean().item()*100.0)

            self_depth_loss = depth_l2_loss(depth_map[MR_mask], depth_map_MR[MR_mask].detach(), torch.exp(-reprojection_error_MR[MR_mask]))*self_depth_weight
            self_depth_loss = self_depth_loss + depth_l2_loss(depth_map[LR_mask], depth_map_LR[LR_mask].detach(), torch.exp(-reprojection_error_LR[LR_mask]))*self_depth_weight
            self_depth_loss = self_depth_loss + depth_l2_loss(depth_map_MR[HR_mask], depth_map[HR_mask].detach(), torch.exp(-reprojection_error[HR_mask]))*self_depth_weight  / (args.down_sampling_ratio[0]**2)
            self_depth_loss = self_depth_loss + depth_l2_loss(depth_map_MR[LR_mask], depth_map_LR[LR_mask].detach(), torch.exp(-reprojection_error_LR[LR_mask]))*self_depth_weight / (args.down_sampling_ratio[0]**2)
            self_depth_loss = self_depth_loss + depth_l2_loss(depth_map_LR[HR_mask], depth_map[HR_mask].detach(), torch.exp(-reprojection_error[HR_mask]))*self_depth_weight / (args.down_sampling_ratio[1]**2)
            self_depth_loss = self_depth_loss + depth_l2_loss(depth_map_LR[MR_mask], depth_map_MR[MR_mask].detach(), torch.exp(-reprojection_error_MR[MR_mask]))*self_depth_weight / (args.down_sampling_ratio[1]**2)

            if self_depth_weight > 0:
                total_loss = total_loss + self_depth_loss
                summary_writer.add_scalar('train/self_depth_loss', self_depth_loss.detach().item(), global_step=iteration)

        if depth_smooth_weight>0:
            depth_smooth_weight *= lr_factor
            depth_map_patch = depth_map[train_items:].view(args.patch_size, args.patch_size, -1)
            depth_map_MR_patch = depth_map_MR[train_items:].view(args.patch_size, args.patch_size, -1)
            depth_map_LR_patch = depth_map_LR[train_items:].view(args.patch_size, args.patch_size, -1)
            depth_smooth_loss = depth_tv_loss(depth_map_patch) * depth_smooth_weight
            depth_smooth_loss = depth_smooth_loss + depth_tv_loss(depth_map_MR_patch) * depth_smooth_weight / (args.down_sampling_ratio[0]**2)
            depth_smooth_loss = depth_smooth_loss + depth_tv_loss(depth_map_LR_patch) * depth_smooth_weight / (args.down_sampling_ratio[1]**2)
            total_loss = total_loss + depth_smooth_loss 
            summary_writer.add_scalar('train/depth_smooth_loss', depth_smooth_loss.detach().item(), global_step=iteration)

        if dense_depth_weight>0:
            dense_depth_weight *= lr_factor
            dense_depth_loss = 0
            for view_id in args.train_frame_num:
                view_mask = ids_train == view_id
                dense_depth_loss = dense_depth_loss + scale_invariant_depth_loss(dense_depth_train[view_mask], depth_map[:train_items][view_mask]) * dense_depth_weight 
                dense_depth_loss = dense_depth_loss + scale_invariant_depth_loss(dense_depth_train[view_mask], depth_map_MR[:train_items][view_mask]) * dense_depth_weight
                dense_depth_loss = dense_depth_loss + scale_invariant_depth_loss(dense_depth_train[view_mask], depth_map_LR[:train_items][view_mask]) * dense_depth_weight

            total_loss = total_loss + dense_depth_loss
            summary_writer.add_scalar('train/dense_depth_loss', dense_depth_loss.detach().item(), global_step=iteration)

        if occ_loss_weight>0:
            occ_loss_weight *= lr_factor
            occ_loss = torch.mean(cal_occ_loss(sigma, rgb_ray, 10, 15, wb_prior))
            if (tensorf.gridSize//args.down_sampling_ratio[0]).sum() != 0:
                occ_loss = occ_loss + torch.mean(cal_occ_loss(sigma_MR, rgb_ray_MR, int(nSamples_MR*reg_rate), int(nSamples_MR * wb_rate), wb_prior)) / (args.down_sampling_ratio[0]**2)
            if (tensorf.gridSize//args.down_sampling_ratio[1]).sum() != 0:
                occ_loss = occ_loss + torch.mean(cal_occ_loss(sigma_LR, rgb_ray_LR, int(nSamples_LR*reg_rate), int(nSamples_LR * wb_rate), wb_prior)) / (args.down_sampling_ratio[1]**2)
            total_loss = total_loss + occ_loss * occ_loss_weight
            summary_writer.add_scalar('train/occ_loss', occ_loss.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        # throughput & GPU memory metrics
        iter_time = max(1e-6, time.perf_counter() - iter_start)
        rays_this_iter = int(rays_train_all.shape[0])
        throughput = float(rays_this_iter) / iter_time
        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024.0*1024.0)
            except Exception:
                pass

        # log extra metrics
        summary_writer.add_scalar('train/throughput_rays_per_s', throughput, global_step=iteration)
        summary_writer.add_scalar('train/gpu_mem_mb', gpu_mem_mb, global_step=iteration)
        summary_writer.add_scalar('train/loss_hr', float(loss), global_step=iteration)
        summary_writer.add_scalar('train/loss_mr', float(loss_MR.detach().item()), global_step=iteration)
        summary_writer.add_scalar('train/loss_lr', float(loss_LR.detach().item()), global_step=iteration)
        try:
            summary_writer.add_scalar('train/total_loss', float(total_loss.detach().item()), global_step=iteration)
        except Exception:
            pass
        try:
            summary_writer.add_scalar('train/depth_loss', float(loss_depth.detach().item()), global_step=iteration)
        except Exception:
            pass

        # write csv row
        try:
            psnr_cur = float(np.mean(PSNRs)) if len(PSNRs)>0 else np.nan
        except Exception:
            psnr_cur = np.nan
        # learning rate (first param group)
        try:
            current_lr = float(optimizer.param_groups[0]['lr'])
        except Exception:
            current_lr = float('nan')
        metrics_writer.writerow([
            iteration,
            float(total_loss.detach().item()) if 'total_loss' in locals() else float('nan'),
            loss,
            float(loss_MR.detach().item()),
            float(loss_LR.detach().item()),
            float(loss_depth.detach().item()) if 'loss_depth' in locals() else float('nan'),
            psnr_cur,
            last_ssim,
            last_lpips,
            prop_hr, prop_mr, prop_lr,
            throughput,
            gpu_mem_mb,
            iter_time,
            current_lr,
            int(W), int(H), float(f if isinstance(f, (int,float)) else (f[0] if hasattr(f,'__len__') else 0.0))
        ])
        metrics_csv_file.flush()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses and display metrics table.
        if iteration % args.progress_refresh_rate == 0:
            # Calculate current metrics
            current_psnr = float(np.mean(PSNRs)) if len(PSNRs) > 0 else 0.0
            current_loss = float(loss) if 'loss' in locals() else 0.0
            current_lr = float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else 0.0
            
            # Display metrics table every 100 iterations
            if iteration % (args.progress_refresh_rate * 2) == 0:
                print("\n" + "="*80)
                print(f"TRAINING METRICS - Iteration {iteration:05d}")
                print("="*80)
                print(f"{'Metric':<25} {'Current':<15} {'Status':<15} {'Notes':<25}")
                print("-"*80)
                print(f"{'PSNR (dB)':<25} {current_psnr:<15.2f} {'Good' if current_psnr > 20 else 'Poor':<15} {'Higher is better':<25}")
                print(f"{'Loss':<25} {current_loss:<15.6f} {'Good' if current_loss < 0.1 else 'High':<15} {'Lower is better':<25}")
                print(f"{'Learning Rate':<25} {current_lr:<15.6f} {'Normal':<15} {'Decaying':<25}")
                
                # Cross-scale proportions
                if not (np.isnan(prop_hr) or np.isnan(prop_mr) or np.isnan(prop_lr)):
                    print(f"{'High Res Proportion (%)':<25} {prop_hr:<15.1f} {'Active' if prop_hr > 30 else 'Low':<15} {'Cross-scale adaptation':<25}")
                    print(f"{'Mid Res Proportion (%)':<25} {prop_mr:<15.1f} {'Active' if prop_mr > 25 else 'Low':<15} {'Cross-scale adaptation':<25}")
                    print(f"{'Low Res Proportion (%)':<25} {prop_lr:<15.1f} {'Active' if prop_lr > 20 else 'Low':<15} {'Cross-scale adaptation':<25}")
                
                # Performance metrics
                if 'throughput' in locals():
                    print(f"{'Throughput (rays/s)':<25} {throughput:<15.0f} {'Good' if throughput > 1000 else 'Slow':<15} {'Training speed':<25}")
                if 'gpu_mem_mb' in locals():
                    print(f"{'GPU Memory (MB)':<25} {gpu_mem_mb:<15.0f} {'Normal' if gpu_mem_mb < 8000 else 'High':<15} {'Memory usage':<25}")
                
                # Test metrics if available
                if len(PSNRs_test) > 1:
                    print(f"{'Test PSNR (dB)':<25} {float(np.mean(PSNRs_test)):<15.2f} {'Good' if np.mean(PSNRs_test) > 20 else 'Poor':<15} {'Validation quality':<25}")
                if len(SSIMs_test) > 1:
                    print(f"{'Test SSIM':<25} {float(np.mean(SSIMs_test)):<15.3f} {'Good' if np.mean(SSIMs_test) > 0.8 else 'Poor':<15} {'Structural similarity':<25}")
                if len(LPIPSs_test) > 1:
                    print(f"{'Test LPIPS':<25} {float(np.mean(LPIPSs_test)):<15.3f} {'Good' if np.mean(LPIPSs_test) < 0.2 else 'Poor':<15} {'Perceptual distance':<25}")
                
                print("="*80)
                print()
            
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {current_psnr:.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' test_ssim = {float(np.mean(SSIMs_test)):.2f}'
                + f' test_lpips = {float(np.mean(LPIPSs_test)):.2f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test, SSIMs_test, LPIPSs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            summary_writer.add_scalar('test/ssim', np.mean(SSIMs_test), global_step=iteration)
            summary_writer.add_scalar('test/lpips', np.mean(LPIPSs_test), global_step=iteration)
            # cache for CSV
            try:
                last_ssim = float(np.mean(SSIMs_test))
                last_lpips = float(np.mean(LPIPSs_test))
            except Exception:
                last_ssim, last_lpips = float('nan'), float('nan')



        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<640**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)
            tensorf.downsample_volume_grid(tensorf.down_sampling_ratio)
            nSamples_MR = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio*args.down_sampling_ratio[0]))
            nSamples_LR = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio*args.down_sampling_ratio[1]))

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')
    
    # Print training completion summary
    print("\n" + "="*80)
    print("✅ FRUGALNERF TRAINING COMPLETED")
    print("="*80)
    print(f"📁 Model saved: {logfolder}/{args.expname}.th")
    print(f"📊 Metrics CSV: {logfolder}/metrics.csv")
    print(f"📈 Cross-scale plot: {logfolder}/cross_scale_adaptation.png")
    print(f"📉 Loss curves: {logfolder}/cross_scale_losses.png")
    print(f"📋 Training report: {logfolder}/training_report.html")
    print("="*80)

    # Plot cross-scale adaptation curves if matplotlib available
    try:
        if plt is not None:
            import pandas as pd
            df = pd.read_csv(metrics_csv_path)
            
            # Plot 1: Cross-scale geometric adaptation (proportions)
            fig1, ax1 = plt.subplots(figsize=(10,6))
            
            # Filter out NaN values for plotting
            valid_mask = ~(df['prop_hr'].isna() | df['prop_mr'].isna() | df['prop_lr'].isna())
            df_valid = df[valid_mask]
            
            if len(df_valid) > 0:
                ax1.plot(df_valid['iter'], df_valid['prop_hr'], label='High res.', color='blue', linewidth=2)
                ax1.plot(df_valid['iter'], df_valid['prop_mr'], label='Mid res.', color='orange', linewidth=2)
                ax1.plot(df_valid['iter'], df_valid['prop_lr'], label='Low res.', color='green', linewidth=2)
                
                # Add smoothed versions (moving average) for better visualization
                window_size = min(50, len(df_valid) // 10)
                if window_size > 1:
                    ax1.plot(df_valid['iter'], df_valid['prop_hr'].rolling(window=window_size, center=True).mean(), 
                            color='lightblue', alpha=0.7, linewidth=1)
                    ax1.plot(df_valid['iter'], df_valid['prop_mr'].rolling(window=window_size, center=True).mean(), 
                            color='lightcoral', alpha=0.7, linewidth=1)
                    ax1.plot(df_valid['iter'], df_valid['prop_lr'].rolling(window=window_size, center=True).mean(), 
                            color='lightgreen', alpha=0.7, linewidth=1)
            
            ax1.set_xlabel('Training iterations')
            ax1.set_ylabel('Proportion of serving as pseudo-GT(%)')
            ax1.set_title('Cross-scale geometric adaptation during training')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 50)  # Set reasonable y-axis limits
            fig1.tight_layout()
            fig1.savefig(os.path.join(logfolder, 'cross_scale_adaptation.png'), dpi=200, bbox_inches='tight')
            plt.close(fig1)
            
            # Plot 2: Loss curves
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.plot(df['iter'], df['loss_hr'], label='HR loss')
            ax2.plot(df['iter'], df['loss_mr'], label='MR loss')
            ax2.plot(df['iter'], df['loss_lr'], label='LR loss')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Cross-scale Loss Curves')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(os.path.join(logfolder, 'cross_scale_losses.png'), dpi=200, bbox_inches='tight')
            plt.close(fig2)
            
            # Create detailed HTML training report
            create_training_report(df, logfolder)
            
            # Summary table with highlight (best=green, worst=red) for selected metrics
            metrics = ['total_loss','psnr','ssim','throughput_rays_per_s','gpu_mem_mb']
            best_is_min = {'total_loss': True, 'psnr': False, 'ssim': False, 'throughput_rays_per_s': False, 'gpu_mem_mb': True}
            html_rows = []
            header = '<tr>' + ''.join(f'<th>{h}</th>' for h in ['iter']+metrics) + '</tr>'
            for _, row in df.iterrows():
                tds = [f"<td>{int(row['iter'])}</td>"]
                for m in metrics:
                    val = row.get(m, float('nan'))
                    style = ''
                    try:
                        if best_is_min[m]:
                            if val == df[m].min(): style = 'background-color:#c6efce;'
                            if val == df[m].max(): style = 'background-color:#ffc7ce;'
                        else:
                            if val == df[m].max(): style = 'background-color:#c6efce;'
                            if val == df[m].min(): style = 'background-color:#ffc7ce;'
                    except Exception:
                        pass
                    tds.append(f'<td style="{style}">{val:.4f}</td>' if isinstance(val,(int,float)) else f'<td>{val}</td>')
                html_rows.append('<tr>' + ''.join(tds) + '</tr>')
            html = '<html><body><h3>Training Metrics Summary</h3><table border="1">' + header + ''.join(html_rows) + '</table></body></html>'
            with open(os.path.join(logfolder, 'metrics_summary.html'), 'w', encoding='utf-8') as fhtml:
                fhtml.write(html)

            # Paper-style summary table image with green/red highlights
            try:
                # pick representative rows: best/worst by selected metrics and final
                cols = {
                    'Total Loss': ('total_loss', True),
                    'PSNR': ('psnr', False),
                    'SSIM': ('ssim', False),
                    'Throughput (tasks/s)': ('throughput_rays_per_s', False),
                    'GPU Mem (GB)': ('gpu_mem_mb', True),
                    'LR': ('lr', False)
                }
                # compute indices
                idx_best_loss = int(df['total_loss'].idxmin()) if 'total_loss' in df else 0
                idx_best_psnr = int(df['psnr'].idxmax()) if 'psnr' in df else 0
                idx_last = len(df)-1
                rows_idx = [idx_best_loss, idx_best_psnr, idx_last]
                row_names = ['Best Loss', 'Best PSNR', 'Final']

                cell_text = []
                cell_colors = []
                # compute best/worst per column for coloring
                best_vals = {}
                worst_vals = {}
                for title,(k,is_min) in cols.items():
                    if k not in df:
                        continue
                    series = df[k]
                    best_vals[title] = series.min() if is_min else series.max()
                    worst_vals[title] = series.max() if is_min else series.min()
                # build table
                for ridx in rows_idx:
                    row = []
                    colors = []
                    for title,(k,is_min) in cols.items():
                        if k in df:
                            val = df.iloc[ridx][k]
                            row.append(f"{val:.3f}")
                            if val == best_vals.get(title):
                                colors.append('#c6efce')
                            elif val == worst_vals.get(title):
                                colors.append('#ffc7ce')
                            else:
                                colors.append('white')
                        else:
                            row.append('-')
                            colors.append('white')
                    cell_text.append(row)
                    cell_colors.append(colors)

                fig2, ax2 = plt.subplots(figsize=(8, 2))
                ax2.axis('off')
                the_table = plt.table(cellText=cell_text,
                                       rowLabels=row_names,
                                       colLabels=list(cols.keys()),
                                       cellColours=cell_colors,
                                       loc='center')
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(8)
                the_table.scale(1, 1.4)
                fig2.tight_layout()
                fig2.savefig(os.path.join(logfolder, 'metrics_table.png'), dpi=200)
                plt.close(fig2)
            except Exception:
                pass
    except Exception:
        pass


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, frame_num=args.train_frame_num)
        PSNRs_test,SSIMs_test,LPIPSs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} train all ssim: {np.mean(SSIMs_test)} <========================')
        print(f'======> {args.expname} train all lpips: {np.mean(LPIPSs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test,SSIMs_test,LPIPSs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} test all ssim: {np.mean(SSIMs_test)} <========================')
        print(f'======> {args.expname} test all lpips: {np.mean(LPIPSs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)



if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20250614)
    np.random.seed(20250614)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

