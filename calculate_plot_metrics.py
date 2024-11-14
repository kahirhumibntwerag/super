@torch.no_grad()
def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
              cond_key=None, return_original_cond=False, bs=None):
    # Retrieve and prepare main input
    x = super().get_input(batch, k)
    x = x[:bs] if bs is not None else x
    x = x.to(self.device)
    
    # Encode input and get latent representation
    z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

    # Handle conditioning input
    c, xc = None, None
    if self.model.conditioning_key is not None:
        cond_key = cond_key or self.cond_stage_key
        xc = (batch[cond_key] if cond_key in ['caption', 'coordinates_bbox'] else
              batch if cond_key == 'class_label' else
              super().get_input(batch, cond_key).to(self.device)) if cond_key != self.first_stage_key else x
        
        # Encode conditioning if necessary
        c = (self.get_learned_conditioning(xc) if isinstance(xc, (dict, list)) else
             self.get_learned_conditioning(xc.to(self.device))) if not self.cond_stage_trainable or force_c_encode else xc
        c = c[:bs] if bs is not None else c

        # Add positional encodings if needed
        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            c = {__conditioning_keys__[self.model.conditioning_key]: c, 'pos_x': pos_x, 'pos_y': pos_y}

    elif self.use_positional_encodings:
        pos_x, pos_y = self.compute_latent_shifts(batch)
        c = {'pos_x': pos_x, 'pos_y': pos_y}

    # Prepare output list
    out = [z, c]
    if return_first_stage_outputs:
        xrec = self.decode_first_stage(z)
        out.extend([x, xrec])
    if return_original_cond:
        out.append(xc)

    return out