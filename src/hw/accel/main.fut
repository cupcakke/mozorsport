entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[n][half*2]f16 =
  let d = half * 2
  in map (\row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:d] :> [half]f16
    let scale = map (\j ->
      let sum = weights_s[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_s[j][0:half] :> [half]f16) x2)
      let clipped = f16.max clip_min (f16.min clip_max sum)
      in f16.exp clipped
    ) (iota half)
    let y1 = map2 (\a b -> a f16.* b) x1 scale
    let trans = map (\j ->
      weights_t[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_t[j][0:half] :> [half]f16) y1)
    ) (iota half)
    let y2 = map2 (\a b -> a f16.+ b) x2 trans
    in y1 ++ y2 :> [half*2]f16
  ) input

entry rsf_backward [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let d = half * 2
  let zero_mat_ws = replicate half (replicate (half+1) (f16.i32 0))
  let zero_mat_wt = replicate half (replicate (half+1) (f16.i32 0))
  in loop (grad_ws, grad_wt) = (zero_mat_ws, zero_mat_wt) for i < n do
    let row = input[i]
    let g_row = grad_output[i]
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:d] :> [half]f16
    let pre_scale = map (\j ->
      weights_s[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_s[j][0:half] :> [half]f16) x2)
    ) (iota half)
    let scale = map (\ps ->
      let clipped = f16.max clip_min (f16.min clip_max ps)
      in f16.exp clipped
    ) pre_scale
    let y1 = map2 (\a b -> a f16.* b) x1 scale
    let dy1 = g_row[0:half] :> [half]f16
    let dy2 = g_row[half:d] :> [half]f16
    let grad_wt_batch = map (\j ->
      map (\k -> if k < half then dy2[j] f16.* y1[k] else dy2[j]) (iota (half+1))
    ) (iota half)
    let dy1_total = map2 (\dy1_j j ->
      dy1_j f16.+ f16.sum (map (\k -> weights_t[k][j] f16.* dy2[k]) (iota half))
    ) dy1 (iota half)
    let ds = map2 (\j ps ->
      let in_range = ps f16.>= clip_min && ps f16.<= clip_max
      in if in_range then dy1_total[j] f16.* y1[j] else (f16.i32 0)
    ) (iota half) pre_scale
    let grad_ws_batch = map (\j ->
      map (\k -> if k < half then ds[j] f16.* x2[k] else ds[j]) (iota (half+1))
    ) (iota half)
    let new_grad_ws = map2 (map2 (\a b -> a f16.+ b)) grad_ws grad_ws_batch
    let new_grad_wt = map2 (map2 (\a b -> a f16.+ b)) grad_wt grad_wt_batch
    in (new_grad_ws, new_grad_wt)

entry sfd_update_mat [d][e] (weights: *[d][e]f16) (gradients: [d][e]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d][e]f16) : (*[d][e]f16, *[d][e]f16) =
  let new_velocity = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity gradients
  let new_weights = map2 (map2 (\w v -> w f16.- v)) weights (copy new_velocity)
  in (new_weights, new_velocity)

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry batch_forward [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][half*2]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t clip_min clip_max) inputs

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff_f32 = map2 (map2 (map2 (\o t ->
    let diff = (f32.f16 o) - (f32.f16 t)
    in diff * diff
  ))) outputs targets
  let total_f32 = f32.sum (flatten (flatten squared_diff_f32))
  let count_f32 = f32.i64 (batch_size * seq_len * d)
  let mean_f32 = total_f32 / count_f32
  in f16.f32 mean_f32

entry batch_gradients [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let results = map2 (\inp g_out ->
    rsf_backward inp g_out weights_s weights_t clip_min clip_max
  ) inputs grad_outputs
  let gs_list = map (\(gs, _) -> gs) results
  let gt_list = map (\(_, gt) -> gt) results
  let gs_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gs_list
  let gt_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gt_list
  in (copy gs_total, copy gt_total)

entry rsf_backward_full [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, [n][half*2]f16) =
  let d = half * 2
  let per_token = map2 (\row g_row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:d] :> [half]f16
    let pre_scale = map (\j ->
      weights_s[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_s[j][0:half] :> [half]f16) x2)
    ) (iota half)
    let scale = map (\ps ->
      let clipped = f16.max clip_min (f16.min clip_max ps)
      in f16.exp clipped
    ) pre_scale
    let y1 = map2 (\a b -> a f16.* b) x1 scale
    let dy1 = g_row[0:half] :> [half]f16
    let dy2 = g_row[half:d] :> [half]f16
    let dy1_total = map2 (\dy1_j j ->
      dy1_j f16.+ f16.sum (map (\k -> weights_t[k][j] f16.* dy2[k]) (iota half))
    ) dy1 (iota half)
    let ds = map2 (\j ps ->
      let in_range = ps f16.>= clip_min && ps f16.<= clip_max
      in if in_range then dy1_total[j] f16.* y1[j] else (f16.i32 0)
    ) (iota half) pre_scale
    let grad_wt_tok = map (\j -> map (\k -> if k < half then dy2[j] f16.* y1[k] else dy2[j]) (iota (half+1))) (iota half)
    let grad_ws_tok = map (\j -> map (\k -> if k < half then ds[j] f16.* x2[k] else ds[j]) (iota (half+1))) (iota half)
    let dx1 = map2 (\g s -> g f16.* s) dy1_total scale
    let dx2_from_ds = map (\k ->
      f16.sum (map (\j -> ds[j] f16.* weights_s[j][k]) (iota half))
    ) (iota half)
    let dx2 = map2 (\a b -> a f16.+ b) dy2 dx2_from_ds
    let grad_in_row = dx1 ++ dx2 :> [half*2]f16
    in (grad_ws_tok, grad_wt_tok, grad_in_row)
  ) input grad_output
  let gw_s_list = map (\(gw_s, _, _) -> gw_s) per_token
  let gw_t_list = map (\(_, gw_t, _) -> gw_t) per_token
  let g_in_rows = map (\(_, _, g_in) -> g_in) per_token
  let gs_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gw_s_list
  let gt_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gw_t_list
  in (gs_total, gt_total, g_in_rows)

let rsf_inverse_flow [half] (y: [half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : [half*2]f16 =
  let d = half * 2
  let y1 = y[0:half] :> [half]f16
  let y2 = y[half:d] :> [half]f16
  let trans = map (\j ->
    weights_t[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_t[j][0:half] :> [half]f16) y1)
  ) (iota half)
  let x2 = map2 (\a b -> a f16.- b) y2 trans
  let scale = map (\j ->
    let raw = weights_s[j][half] f16.+ f16.sum (map2 (\w x -> w f16.* x) (weights_s[j][0:half] :> [half]f16) x2)
    let clipped = f16.max clip_min (f16.min clip_max raw)
    in f16.exp clipped
  ) (iota half)
  let x1 = map2 (\a b -> a f16./ b) y1 scale
  in x1 ++ x2 :> [half*2]f16

entry batch_rsf_inverse [batch_size][seq_len][half]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  map (\sample ->
    map (\row ->
      rsf_inverse_flow row weights_s weights_t clip_min clip_max
    ) sample
  ) outputs

entry batch_gradients_full [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, *[batch_size][seq_len][half*2]f16) =
  let results = map2 (\inp g_out ->
    rsf_backward_full inp g_out weights_s weights_t clip_min clip_max
  ) inputs grad_outputs
  let gs_list = map (\(gs, _, _) -> gs) results
  let gt_list = map (\(_, gt, _) -> gt) results
  let gin_list = map (\(_, _, gin) -> gin) results
  let gs_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gs_list
  let gt_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gt_list
  in (copy gs_total, copy gt_total, copy gin_list)

entry compute_initial_grad_l2 [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16)
  : *[batch_size][seq_len][d]f16 =
  map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets

entry xavier_fill_inplace [d] (_weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale
    ) (iota d)
  ) (iota d)

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

entry training_step [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (weights_s: *[half][half+1]f16)
  (weights_t: *[half][half+1]f16)
  (velocity_s: *[half][half+1]f16)
  (velocity_t: *[half][half+1]f16)
  (learning_rate: f16)
  (momentum: f16)
  (clip_min: f16)
  (clip_max: f16) : (*[half][half+1]f16, *[half][half+1]f16, *[half][half+1]f16, *[half][half+1]f16, f16) =
  let outputs = batch_forward inputs weights_s weights_t clip_min clip_max
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let (grad_s, grad_t) = batch_gradients inputs grad_outputs weights_s weights_t clip_min clip_max
  let grad_s_c = copy grad_s
  let grad_t_c = copy grad_t
  let (new_weights_s, new_velocity_s) = sfd_update_mat weights_s grad_s_c learning_rate momentum velocity_s
  let (new_weights_t, new_velocity_t) = sfd_update_mat weights_t grad_t_c learning_rate momentum velocity_t
  in (new_weights_s, new_weights_t, new_velocity_s, new_velocity_t, loss)

let oftb_scale : f16 = f16.f32 0.7071067811865476

entry oftb_forward_single [seq_len][dim] (input: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:dim] :> [half]f16
    let new_x1 = map2 (\a b -> (a f16.- b) f16.* oftb_scale) x1 x2
    let new_x2 = map2 (\a b -> (a f16.+ b) f16.* oftb_scale) x1 x2
    in new_x1 ++ new_x2 :> [dim]f16
  ) input

entry oftb_backward_single [seq_len][dim] (grad_output: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let g1 = row[0:half] :> [half]f16
    let g2 = row[half:dim] :> [half]f16
    let new_g1 = map2 (\a b -> (a f16.+ b) f16.* oftb_scale) g1 g2
    let new_g2 = map2 (\a b -> (b f16.- a) f16.* oftb_scale) g1 g2
    in new_g1 ++ new_g2 :> [dim]f16
  ) grad_output

entry oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_forward_single sample) inputs

entry oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_backward_single sample) grad_outputs

entry batch_oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_forward inputs

entry batch_oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_backward grad_outputs

entry embedding_forward [n][vocab_size][dim] (tokens: [n]i64) (weight: [vocab_size][dim]f16) : *[n][dim]f16 =
  map (\tok ->
    let t = if tok >= 0 && tok < vocab_size then tok else 0
    in weight[t]
  ) tokens

entry embedding_backward [n][vocab_size][dim] (tokens: [n]i64) (grad_output: [n][dim]f16) (grad_weight: [vocab_size][dim]f16) : *[vocab_size][dim]f16 =
  loop gw = copy grad_weight for i < n do
    let t = if tokens[i] >= 0 && tokens[i] < vocab_size then tokens[i] else 0
    let row_update = map2 (\g acc -> acc f16.+ g) grad_output[i] gw[t]
    in gw with [t] = row_update

entry embedding_update [vocab_size][dim] (weight: *[vocab_size][dim]f16) (grad_weight: [vocab_size][dim]f16) (lr: f16) : *[vocab_size][dim]f16 =
  map2 (map2 (\w g -> w f16.- lr f16.* g)) weight grad_weight
