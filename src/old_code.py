@torch.enable_grad()
def identity_attack_memory_efficient(
    model: DiffusionPipeline,
    trainset: FaceLatentDataset,
    target_img: Image.Image,
    victim_models: VictimModels,
    test_models: VictimModels,
    # parse_label_map: torch.Tensor,
    parse_model: BiSeNet,
    # parse_loss,
    ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
) -> torch.Tensor:
    """
    """
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)
    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res

    batch_size = config.training.batch_size
    lr = config.training.lr
    alpha = config.training.attack_loss_weight
    beta = config.training.cross_attn_loss_weight
    gamma = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)
    lam = config.training.semseg_lam

    ########################################
    if lam > 0.0:
        from faceutils.utils import SemSegCELoss
        parse_loss_fn = SemSegCELoss(0.7, 1024*1024//16, ignore_label=0)
    ########################################

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = image2tensor(target_img, device)
    target_embedding = victim_models(target_img)

    controller = StructureLossAttentionStore(config.diffusion.diffusion_steps, res=res, batch_size=batch_size)

    prompt_str = trainset.prompt
    prompt = [prompt_str] * batch_size * 2
    prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
    context = torch.cat([uncond_embeds, prompt_embeds])

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    # delta = torch.randn(1, 4, res//8, res//8).to(device) * 0.1
    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0)
    else:
        optimizer = PGD(delta, lr=lr, radius=config.training.pgd_radius)

    for e in range(n_epoch):
        register_attention_control(model, controller)
        for it, batch in enumerate(train_loader):
            batch_latents: torch.Tensor
            images, batch_latents = batch
            batch_latents = batch_latents.to(device)
            batch_latents.requires_grad_(False)

            #######################################
            if lam > 0.0:
                images_parse = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False) #[-1.0, 1.0]
                parse_maps_labels = parse_model((images_parse + 1) / 2)[0].detach().argmax(1)
                parse_maps_labels[parse_maps_labels == 17] = 0  # denote hair as background, which we ignore in the loss
            #######################################

            input_latents = torch.cat([batch_latents, batch_latents + delta]).detach()
            input_latents.requires_grad_(False)

            # initial dinoising loop to pre-calculate the latents of each timestep
            intermediate_latents = [input_latents]
            with torch.no_grad():
                latents = input_latents
                for i in range(start_step, num_inference_steps):
                    t = model.scheduler.timesteps[i]
                    noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                    latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)
                    intermediate_latents.append(latents)
                controller.loss = 0.0
                controller.reset()

            # calculate adv_loss loss from adv_latent z0
            adv_latent = intermediate_latents[-1][batch_size:] # discard the clean latents
            adv_latent.requires_grad_(True)
            adv_image = latent2image(model, adv_latent, return_tensor=True) # [-1, 1]

            adv_embeddings = victim_models(adv_image)
            adv_loss = alpha * cos_dist_loss(adv_embeddings, target_embedding)

            ###################################################
            # calculate parse_loss
            total_loss = 0.0 + adv_loss
            parse_loss = torch.Tensor([0.0])
            if lam > 0.0:
                adv_image_parse = F.interpolate(adv_image, size=(1024, 1024), mode='bilinear', align_corners=False) #[0.0, 1.0]
                parse_map_adv = parse_model(adv_image_parse)[0]
                parse_loss = parse_loss_fn(parse_map_adv, parse_maps_labels)
                total_loss += lam * parse_loss
            ###################################################

            # initialize the gradient as dL/dz0
            grad = torch.autograd.grad(total_loss, adv_latent)[0].detach()
            
            total_loss = total_loss.detach()
            self_attention_loss = 0.0
            cross_attention_loss = 0.0
            
            # memory efficient gradient calculation
            for i in reversed(list(range(start_step, num_inference_steps))):
                t = model.scheduler.timesteps[i]

                # retreive the pre-calculated latents and set the adv half to requires_grad = True
                latent_idx = i - start_step
                latents_pre_calc = intermediate_latents[latent_idx]
                clean_latents_pre_calc, adv_latents_pre_calc = torch.chunk(latents_pre_calc, 2)
                adv_latents_pre_calc.requires_grad_(True)
                latents_pre_calc = torch.cat([clean_latents_pre_calc, adv_latents_pre_calc])

                noise_pred_pre_calc = get_noise_pred(model, latents_pre_calc, t, context, do_classifier_free_guidance, guidance_scale)
                next_latents = diffusion_step(latents_pre_calc, noise_pred_pre_calc, model.scheduler, t, num_inference_steps)

                step_self_att_loss, step_cross_att_loss = attention_losses(controller, prompt_str, "face", model.tokenizer, batch_size, res)
                step_loss = beta * step_cross_att_loss + gamma * step_self_att_loss

                step_loss_for_grad = torch.sum(next_latents[batch_size:] * grad) + step_loss
                grad = torch.autograd.grad(step_loss_for_grad, adv_latents_pre_calc)[0].detach()

                total_loss += step_loss
                self_attention_loss += step_self_att_loss
                cross_attention_loss += step_cross_att_loss

                controller.loss = 0.0
                controller.reset()
            
            grad = grad.sum(dim=0, keepdim=True)
            optimizer(grad)
            logger.log(f"epoch{e}, iteration{it}")
            # logger.log(f"    grad snippet: \n{grad[0, 0, :2, :2].cpu().numpy()}")
            # logger.log(f"    delta snippet: \n{delta[0, 0, :2, :2].cpu().numpy()}")
            logger.log((
                f"    adv_loss: {adv_loss.item():.3f}, "
                f"parse_loss: {parse_loss.item()*lam:.3f}, "
                f"cross_attention_loss: {cross_attention_loss.item()*beta:.3f}, "
                f"self_attention_loss: {self_attention_loss.item()*gamma:.3f}, "
                f"    total_loss: {total_loss.item():.3f}"
            ))
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}")
        
        reset_attention_control(model)

        ckpt_file = os.path.join(ckpt_dir, f"delta_epoch{e}.pth")
        # if (e + 1) % 4 == 0:
        #     torch.save(delta, ckpt_file)
        torch.save(delta, ckpt_file)
        asr, ssim, lpips = eval(model, trainset, delta, target_img, test_models, False, ckpt_dir, True, config, logger)
        logger.log(f"epoch{e} trainset asr on id {trainset.identity}: {asr}")
        logger.log(f"trainset perceptual sim: ssim={ssim:.6f}, lpips={lpips:.6f}")
        logger.log()

    return delta



@torch.enable_grad()
def identity_attack(
    model: DiffusionPipeline,
    trainset: FaceLatentDataset,
    target_img: Image.Image,
    victim_models: VictimModels,
    parse_model: BiSeNet,
    ckpt_dir: str,
    config: Config,
    logger: DatetimeLogger,
):
    device = model.device
    num_inference_steps = config.diffusion.diffusion_steps
    start_step = config.diffusion.start_step
    guidance_scale = config.diffusion.guidance_scale
    do_classifier_free_guidance = (guidance_scale > 0.0)
    # semseg_lam = args.semseg_lam
    # gamma = 0.02 
    # clip_min = -1e6
    # clip_max = 1e6

    res = config.dataset.res

    batch_size = config.training.batch_size
    lr = config.training.lr
    alpha = config.training.attack_loss_weight
    beta = config.training.cross_attn_loss_weight
    gamma = config.training.self_attn_loss_weight
    n_epoch = config.training.total_update_steps // (len(trainset) // batch_size)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    target_img = image2tensor(target_img, device)
    target_embedding = victim_models(target_img)

    controller = StructureLossAttentionStore(num_inference_steps, res=res, batch_size=batch_size)
    register_attention_control(model, controller)

    prompt_str = trainset.prompt
    prompt = [prompt_str] * batch_size * 2
    prompt_embeds, uncond_embeds = embed_prompt(model, prompt)
    context = torch.cat([uncond_embeds, prompt_embeds])

    delta = torch.zeros(1, 4, res//8, res//8).to(device)
    # delta = torch.randn(1, 4, res//8, res//8).to(device) * 0.1
    delta.requires_grad_(True)

    if config.training.optim_algo == 'adamw':
        optimizer = AdamW(delta, lr=lr)
    elif config.training.optim_algo == 'adam':
        optimizer = AdamW(delta, lr=lr, weight_decay=0.0)
    else:
        optimizer = PGD(delta, lr=lr, radius=config.training.pgd_radius)

    for e in range(n_epoch):
        for it, batch in enumerate(train_loader):
            batch_latents: torch.Tensor
            images, batch_latents = batch
            batch_latents = batch_latents.to(device)
            batch_latents.requires_grad_(False)
            
            adv_latents = batch_latents + delta
            input_latents = torch.cat([batch_latents, adv_latents])

            latents = input_latents
            self_attention_loss = 0.0
            cross_attention_loss = 0.0

            for i in range(start_step, num_inference_steps):
                t = model.scheduler.timesteps[i]
                noise_pred = get_noise_pred(model, latents, t, context, do_classifier_free_guidance, guidance_scale)
                latents = diffusion_step(latents, noise_pred, model.scheduler, t, num_inference_steps)

                step_self_att_loss, step_cross_att_loss = attention_losses(controller, prompt_str, "face", model.tokenizer, batch_size, res)

                self_attention_loss += step_self_att_loss
                cross_attention_loss += step_cross_att_loss

                controller.loss = 0.0
                controller.reset()
            
            adv_latent = latents[batch_size:] # discard the clean latents
            adv_image = latent2image(model, adv_latent, return_tensor=True) # [-1, 1]

            adv_embeddings = victim_models(adv_image)
            adv_loss = cos_dist_loss(adv_embeddings, target_embedding)

            loss = alpha * adv_loss + beta * cross_attention_loss + gamma * self_attention_loss
            loss.backward()
            grad = delta.grad
            optimizer(grad)
            logger.log(f"epoch{e}, iteration{it}")
            # logger.log(f"    grad snippet: \n{grad[0, 0, :2, :2].cpu().detach().numpy()}")
            # logger.log(f"    delta snippet: \n{delta[0, 0, :2, :2].cpu().detach().numpy()}")
            logger.log((
                f"    adv_loss: {adv_loss.item()*alpha:.3f}, "
                f"cross_attention_loss: {cross_attention_loss.item()*beta:.3f}, "
                f"self_attention_loss: {self_attention_loss.item()*gamma:.3f}, "
                f"    total_loss: {loss.item():.3f}"
            ))
            logger.log(f"    delta norm, l2: {delta.norm()}, linf: {delta.abs().max()}")
            logger.log()

            delta.grad.zero_()

    reset_attention_control(model)
