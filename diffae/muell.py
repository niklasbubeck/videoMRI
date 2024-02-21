def sample_interpolated_testdata(self, test_dataset, eta=0.0):
    """Autoencode test data and calculate evaluation metrics.
    """
    assert self.config.bs == 1, f"can only interpolate using batch size 1"
    test_dataset = test_dataset
    test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

    keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
    df = pd.DataFrame(columns=keys)

    self.model.eval()
    for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        for  i in range(len(self.model.unets)):
            unet = self.model.get_unet(i)
            if i == 0:
                interpolate = np.random.randint(1 , batch[0].shape[2]-1)
                prev, next = interpolate - 1, interpolate + 1
                (x0_prev, _, _), _ , reference_prev = self.model.preprocess(batch, **self.config.dataset, slice_idx=prev, unet_number=i)
                (x0_inter, lowres, _),fnames, reference_inter = self.model.preprocess(batch, **self.config.dataset, slice_idx=interpolate, unet_number=i)
                (x0_next, _, _), _, reference_next = self.model.preprocess(batch, **self.config.dataset, slice_idx=next, unet_number=i)

                if len(x0_inter.shape) == 4:
                    dim = 3
                elif len(x0_inter.shape) == 5:
                    dim = 4
                batch_size = x0_inter.shape[0]

                x0_prev = x0_prev.to(self.device)
                x0_inter = x0_inter.to(self.device)
                x0_next = x0_next.to(self.device)
                reference_prev = reference_prev.to(self.device)
                reference_inter = reference_inter.to(self.device)
                reference_next = reference_next.to(self.device)

                reference_images_prevs = [self.model.resize_to(reference_prev, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
                reference_images_inters = [self.model.resize_to(reference_inter, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
                reference_images_nexts = [self.model.resize_to(reference_next, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]

                noise_lows_prev = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_prevs]
                noise_lows_inter = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_inters]
                noise_lows_next = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_nexts]

                t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                t_low = t_low.type(torch.long)
                alphas_shape = self.alphas_cumprod[t_low].shape

                
                alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)

                lowres_images_prev = [torch.sqrt(alpha_t) * reference_images_prevs[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_prev)]
                lowres_images_prev[0] = None
                lowres_images_inter = [torch.sqrt(alpha_t) * reference_images_inters[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_inter)]
                lowres_images_inter[0] = None
                lowres_images_next = [torch.sqrt(alpha_t) * reference_images_nexts[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_next)]
                lowres_images_next[0] = None



                xt_prevs = [self.encode_stochastic(self.model.get_unet(i), reference_images_prev, lowres_cond_img=lowres_images_prev[i], lowres_noise_times=t_low ) for i, reference_images_prev in enumerate(reference_images_prevs)]
                xt_nexts = [self.encode_stochastic(self.model.get_unet(i), reference_images_next, lowres_cond_img=lowres_images_prev[i], lowres_noise_times=t_low ) for i, reference_images_next in enumerate(reference_images_nexts)]

                style_emb_prevs = [self.model.get_unet(i).encoder(reference) for i, reference in enumerate(reference_images_prevs)]
                style_emb_nexts = [self.model.get_unet(i).encoder(reference) for i, reference in enumerate(reference_images_nexts)]

                xt_inters = []
                style_emb_inters = []
                for (xt_1, xt_2, style_emb_1, style_emb_2) in zip (xt_prevs, xt_nexts, style_emb_prevs, style_emb_nexts):
                    xt_inter, style_emb_inter = self.only_interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha=0.5)
                    xt_inters.append(xt_inter)
                    style_emb_inters.append(style_emb_inter)

        

            t_low = None
            if lowres is not None: 
                #scale up if necessary
                lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                
                t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                t_low = t_low.type(torch.long)
                noise_low = torch.randn_like(lowres, device=self.device)
                alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
                lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
            # # Scale up if neccessary
            # xt_inter = self.model.resize_to(xt_inter, self.image_sizes[i], target_frames=self.target_frames[i])  
            xt_inter = xt_inters[i]
            print("Lowres:", lowres)
            unet = self.model.get_unet(i)
            for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
                t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                e = unet.unet(xt_inter, t, text_embeds=style_emb_inters[i], lowres_cond_img=lowres, lowres_noise_times=t_low)
                alphas_shape = self.alphas_cumprod[t].shape

                # Equation 12 of Denoising Diffusion Implicit Models
                x0_t = (
                    torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter
                    - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
                ).clamp(-1, 1)
                e = (
                    (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter - x0_t)
                    / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
                )
                sigma = (
                    eta
                    * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                    * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
                )
                xt_inter = (
                    torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
                    + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
                )
                xt_inter = xt_inter + torch.randn_like(xt_inter) * sigma if _t != 0 else xt_inter

            lowres = xt_inter
        ref = reference_inter[0,0,...].cpu().numpy()
        sample = xt_inter[0,0,...].clamp(0,1).cpu().numpy()

        print(ref.shape, sample.shape)
        subject = extract_seven_concurrent_numbers(fnames['sa'][0])[0]
        mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
        results = [subject, interpolate, mse, psnr, ssim]
        print(results)
        df.loc[len(df.index)] = results

        # try to safe the gifs
        
        ref, sample = ref*255, sample*255
        os.makedirs(os.path.join(self.output_dir, "videos", "interpolation", subject), exist_ok=True)
        videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
        videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
        videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
        videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
        df.to_csv(os.path.join(self.output_dir, "videos", "interpolation" ,f"data.csv"))

    return df


def sample_testdata(self, test_dataset, eta=0.0):
    """Autoencode test data and calculate evaluation metrics.
    """
    test_dataset = test_dataset
    test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

    keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
    df = pd.DataFrame(columns=keys)

    self.model.eval()
    for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        random = np.random.randint(0 , batch[0].shape[2])
        for i in range(len(self.model.unets)):
            
            if i == 0:
                (x0, lowres, seg), fnames, reference = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=i)
                x0 = x0.to(self.device)
                reference=reference.to(self.device)
                batch_size = x0.shape[0]
                if len(x0.shape) == 4:
                    dim = 3
                if len(x0.shape) == 5:
                    dim = 4

                # get all st encoded references
                t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                t_low = t_low.type(torch.long)
                alphas_shape = self.alphas_cumprod[t_low].shape

                
                alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)


                reference_images = [self.model.resize_to(reference, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
                # reference_images = [reference_image.to(self.device) for reference_image in reference_images]
                noise_lows = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images]
                lowres_images = [torch.sqrt(alpha_t) * reference_images[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows)]
                lowres_images[0] = None
                print("test1")
                st_encoded_ref = [self.encode_stochastic(self.model.get_unet(i), reference_image, lowres_cond_img=lowres_images[i], lowres_noise_times=t_low) for i, reference_image in enumerate(reference_images)]
                print("test2")
        

            t_low = None
            if lowres is not None: 
                #scale up if necessary
                lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                
                t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                t_low = t_low.type(torch.long)
                noise_low = torch.randn_like(lowres, device=self.device)
                alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
                lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
            
            # scale up if necessary
            # x0 = self.model.resize_to(x0, self.image_sizes[i], target_frames=self.target_frames[i])
            


            # xt = self.encode_stochastic(unet, x0, disable_tqdm=False, lowres_cond_img=lowres, lowres_noise_times=t_low)
            xt = st_encoded_ref[i]
            print("MIN MAX ST ENC: ", xt.min(), xt.max())
            unet = self.model.get_unet(i)
            style_emb = unet.encoder(reference_images[i])


            for _t in tqdm(reversed(range(self.num_timesteps)), desc="decoding ..."):
                t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                
                e = unet.unet(xt, t, text_embeds=style_emb, lowres_cond_img=lowres, lowres_noise_times=t_low)
                # Equation 12 of Denoising Diffusion Implicit Models
                x0_t = (
                    torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
                    - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
                ).clamp(-1, 1)
                e = (
                    (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                    / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
                )
                sigma = (
                    eta
                    * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                    / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
                )
                xt = (
                    torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
                    + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
                )
                xt = xt + torch.randn_like(xt) * sigma.view(*alphas_shape + (1,) *dim) if _t != 0 else xt

            if i == len(self.model.unets) - 1:
                xt = xt.clamp(0, 1)
                continue

            # x0 = xt
            lowres = xt


        for i in range(batch_size):
            ref = reference[i,0,...].cpu().numpy()
            sample = xt[i,0,...].cpu().numpy()
            subject = extract_seven_concurrent_numbers(fnames['sa'][i])[0]
            mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
            results = [subject, random, mse, psnr, ssim]
            print(results)
            df.loc[len(df.index)] = results

            # try to safe the gifs
            
            ref, sample = ref*255, sample*255
            os.makedirs(os.path.join(self.output_dir, "videos", "reconstruction", subject), exist_ok=True)
            videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
        
        df.to_csv(os.path.join(self.output_dir, "videos", "reconstruction" ,f"data.csv"))


    return df









def sample_interpolated_testdata(self, test_dataset, eta=0.0):
        """Autoencode test data and calculate evaluation metrics.
        """
        assert self.config.bs == 1, f"can only interpolate using batch size 1"
        test_dataset = test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

        keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
        df = pd.DataFrame(columns=keys)

        self.model.eval()
        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            for  i in range(len(self.model.unets)):
                unet = self.model.get_unet(i)
                if i == 0:
                    interpolate = np.random.randint(1 , batch[0].shape[2]-1)
                    prev, next = interpolate - 1, interpolate + 1
                    (x0_prev, _, _), _ , _ = self.model.preprocess(batch, **self.config.dataset, slice_idx=prev, unet_number=i)
                    (x0_inter, lowres, _),fnames, reference = self.model.preprocess(batch, **self.config.dataset, slice_idx=interpolate, unet_number=i)
                    (x0_next, _, _), _, _ = self.model.preprocess(batch, **self.config.dataset, slice_idx=next, unet_number=i)

                    x0_prev = x0_prev.to(self.device)
                    x0_inter = x0_inter.to(self.device)
                    x0_next = x0_next.to(self.device)

                    xt_1 = self.encode_stochastic(unet, x0_prev)
                    xt_2 = self.encode_stochastic(unet, x0_next)

                    style_emb_1 = unet.encoder(x0_prev)
                    style_emb_2 = unet.encoder(x0_next)

                    xt_inter, style_emb_inter = self.only_interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha=0.5)

                if len(xt_1.shape) == 4 and len(xt_2.shape) == 4:
                    dim = 3
                elif len(xt_1.shape) == 5 and len(xt_2.shape) == 5:
                    dim = 4
                batch_size = x0_inter.shape[0]

                t_low = None
                if lowres is not None: 
                    #scale up if necessary
                    lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                    
                    t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                    t_low = t_low.type(torch.long)
                    noise_low = torch.randn_like(lowres, device=self.device)
                    alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
                    lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
                # Scale up if neccessary
                xt_inter = self.model.resize_to(xt_inter, self.image_sizes[i], target_frames=self.target_frames[i]) 

                if i > 0:
                    xt_inter = self.encode_stochastic(unet, xt_inter, lowres_cond_img=lowres, lowres_noise_times=t_low)
                    # xt_inter = torch.rand_like(xt_inter, device=self.device) * 0.2 
                    eta = 1.0

                for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
                    t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                    e = unet.unet(xt_inter, t, text_embeds=style_emb_inter, lowres_cond_img=lowres, lowres_noise_times=t_low)
                    alphas_shape = self.alphas_cumprod[t].shape

                    # Equation 12 of Denoising Diffusion Implicit Models
                    x0_t = (
                        torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter
                        - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
                    ).clamp(-1, 1)
                    e = (
                        (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter - x0_t)
                        / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
                    )
                    sigma = (
                        eta
                        * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                        * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
                    )
                    xt_inter = (
                        torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
                        + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
                    )
                    xt_inter = xt_inter + torch.randn_like(xt_inter) * sigma if _t != 0 else xt_inter

                lowres = xt_inter
            ref = reference[0,0,...].cpu().numpy()
            sample = xt_inter[0,0,...].clamp(0,1).cpu().numpy()

            print(ref.shape, sample.shape)
            subject = extract_seven_concurrent_numbers(fnames['sa'][0])[0]
            mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
            results = [subject, interpolate, mse, psnr, ssim]
            print(results)
            df.loc[len(df.index)] = results

            # try to safe the gifs
            
            ref, sample = ref*255, sample*255
            os.makedirs(os.path.join(self.output_dir, "videos", "interpolation", subject), exist_ok=True)
            videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            df.to_csv(os.path.join(self.output_dir, "videos", "interpolation" ,f"data.csv"))

        return df


    def sample_testdata(self, test_dataset, eta=0.0):
        """Autoencode test data and calculate evaluation metrics.
        """
        test_dataset = test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

        keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
        df = pd.DataFrame(columns=keys)

        self.model.eval()
        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            random = np.random.randint(0 , batch[0].shape[2])
            for i in range(len(self.model.unets)):
                unet = self.model.get_unet(i)
                if i == 0:
                    (x0, lowres, seg), fnames, reference = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=i)
                    x0 = x0.to(self.device)
                
                batch_size = x0.shape[0]

                t_low = None
                if lowres is not None: 
                    #scale up if necessary
                    lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                    
                    t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
                    t_low = t_low.type(torch.long)
                    noise_low = torch.randn_like(lowres, device=self.device)
                    alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
                    lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
                
                # scale up if necessary
                x0 = self.model.resize_to(x0, self.image_sizes[i], target_frames=self.target_frames[i])
                


                xt = self.encode_stochastic(unet, x0, disable_tqdm=False, lowres_cond_img=lowres, lowres_noise_times=t_low)
                style_emb = unet.encoder(x0)


                if len(x0.shape) == 4:
                    dim = 3
                if len(x0.shape) == 5:
                    dim = 4


                for _t in tqdm(reversed(range(self.num_timesteps)), desc="decoding ..."):
                    t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                    
                    e = unet.unet(xt, t, text_embeds=style_emb, lowres_cond_img=lowres, lowres_noise_times=t_low)
                    alphas_shape = self.alphas_cumprod[t].shape
                    # Equation 12 of Denoising Diffusion Implicit Models
                    x0_t = (
                        torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
                        - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
                    ).clamp(-1, 1)
                    e = (
                        (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                        / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
                    )
                    sigma = (
                        eta
                        * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                        / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
                    )
                    xt = (
                        torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
                        + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
                    )
                    xt = xt + torch.randn_like(x0) * sigma.view(*alphas_shape + (1,) *dim) if _t != 0 else xt

                if i == len(self.model.unets) - 1:
                    continue

                x0 = xt
                lowres = xt


            for i in range(batch_size):
                ref = reference[i,0,...].cpu().numpy()
                sample = xt[i,0,...].cpu().numpy()
                subject = extract_seven_concurrent_numbers(fnames['sa'][i])[0]
                mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
                results = [subject, random, mse, psnr, ssim]
                print(results)
                df.loc[len(df.index)] = results

                # try to safe the gifs
                
                ref, sample = ref*255, sample*255
                os.makedirs(os.path.join(self.output_dir, "videos", "reconstruction", subject), exist_ok=True)
                videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
                videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
                videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
                videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            
            df.to_csv(os.path.join(self.output_dir, "videos", "reconstruction" ,f"data.csv"))


        return df