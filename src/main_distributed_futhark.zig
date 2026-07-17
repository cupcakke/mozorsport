const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const dtf = @import("distributed/distributed_trainer_futhark.zig");
const DistributedTrainerFuthark = dtf.DistributedTrainerFuthark;
const TrainerConfig = dtf.TrainerConfig;
const TrainerComponents = dtf.TrainerComponents;
const MGT = @import("tokenizer/mgt.zig").MGT;
const nccl = @import("distributed/nccl_bindings.zig");
const modal_gpu = @import("distributed/modal_gpu.zig");
const core_relational = @import("core_relational/mod.zig");

fn extractDatasetText(allocator: std.mem.Allocator, line: []const u8) !?[]const u8 {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        line,
        .{ .allocate = .alloc_always },
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null,
    };
    defer parsed.deinit();

    return switch (parsed.value) {
        .object => |obj| blk: {
            const text_value = obj.get("text") orelse break :blk null;
            break :blk switch (text_value) {
                .string => |text| if (text.len > 0)
                    try allocator.dupe(u8, text)
                else
                    null,
                else => null,
            };
        },
        else => null,
    };
}

fn loadDataset(
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![][]const u8 {
    if (coordinator.world_size == 0) return error.InvalidWorldSize;
    if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;

    const env_total_owned: ?[]u8 = std.process.getEnvVarOwned(allocator, "JAIDE_TOTAL_SAMPLES") catch null;
    defer if (env_total_owned) |o| allocator.free(o);
    const env_max_owned: ?[]u8 = std.process.getEnvVarOwned(allocator, "JAIDE_MAX_SAMPLES") catch null;
    defer if (env_max_owned) |o| allocator.free(o);

    var valid_sample_count: usize = 0;
    if (env_total_owned) |s| {
        valid_sample_count = std.fmt.parseInt(usize, s, 10) catch 0;
    }

    if (valid_sample_count == 0) {
        std.debug.print("[Rank {d}] WARN: JAIDE_TOTAL_SAMPLES not provided, falling back to valid-record scan\n", .{coordinator.rank});
        const count_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| return err;
        defer count_file.close();
        var count_buf_reader = std.io.bufferedReader(count_file.reader());
        var count_stream = count_buf_reader.reader();
        while (try count_stream.readUntilDelimiterOrEofAlloc(allocator, '\n', max_line_size)) |line| {
            defer allocator.free(line);
            const maybe_text = extractDatasetText(allocator, line) catch null;
            if (maybe_text) |text| {
                allocator.free(text);
                valid_sample_count = try std.math.add(usize, valid_sample_count, 1);
            }
        }
    }

    if (env_max_owned) |s| {
        const cap = std.fmt.parseInt(usize, s, 10) catch 0;
        if (cap > 0 and cap < valid_sample_count) valid_sample_count = cap;
    }

    if (valid_sample_count == 0) {
        std.debug.print("[Rank {d}] ERROR: Dataset is empty or contains no valid records\n", .{coordinator.rank});
        return error.EmptyDataset;
    }

    const base_per_rank = valid_sample_count / coordinator.world_size;
    const remainder = valid_sample_count % coordinator.world_size;
    const samples_per_rank = if (coordinator.rank < remainder) base_per_rank + 1 else base_per_rank;
    const start_valid_index = if (coordinator.rank < remainder)
        coordinator.rank * (base_per_rank + 1)
    else
        remainder * (base_per_rank + 1) + (coordinator.rank - remainder) * base_per_rank;

    var samples = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (samples.items) |sample| {
            allocator.free(sample);
        }
        samples.deinit();
    }

    if (samples_per_rank > 0) {
        const load_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| return err;
        defer load_file.close();
        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();

        var appended: usize = 0;
        var valid_idx: usize = 0;
        while (try load_stream.readUntilDelimiterOrEofAlloc(allocator, '\n', max_line_size)) |line| {
            defer allocator.free(line);

            if (valid_idx >= start_valid_index + samples_per_rank) {
                break;
            }

            const maybe_text = try extractDatasetText(allocator, line);
            if (maybe_text) |text_copy| {
                if (valid_idx >= start_valid_index) {
                    try samples.append(text_copy);
                    appended += 1;
                } else {
                    allocator.free(text_copy);
                }
                valid_idx += 1;
            }

            if (appended == samples_per_rank) {
                break;
            }
        }

        if (appended < samples_per_rank) {
            std.debug.print("[Rank {d}] WARN: read {d}/{d} valid samples before EOF (valid_idx={d})\n", .{ coordinator.rank, appended, samples_per_rank, valid_idx });
        }
    }

    if (samples.items.len != samples_per_rank) {
        std.debug.print("[Rank {d}] WARN: partition got {d} valid samples, expected {d} (some records may have been invalid)\n", .{ coordinator.rank, samples.items.len, samples_per_rank });
    }

    if (coordinator.isRoot()) {
        std.debug.print("[Rank {d}] Loaded {d} samples from total {d} valid records (rank slice)\n", .{
            coordinator.rank,
            samples.items.len,
            valid_sample_count,
        });
    }

    return samples.toOwnedSlice();
}

fn deployToModal(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    const api_token = try std.process.getEnvVarOwned(allocator, "MODAL_API_TOKEN");
    defer allocator.free(api_token);

    const model_path: []const u8 = if (args.len > 0) args[0] else "/checkpoints/latest";
    const dataset_path: []const u8 = if (args.len > 1) args[1] else "/data/dataset/train.jsonl";

    var client = try modal_gpu.ModalGPUClient.init(allocator, api_token);
    defer client.deinit();

    const job_id = try client.deployTrainingJob(model_path, dataset_path);
    defer allocator.free(job_id);

    std.debug.print("Deployed training job: {s}\n", .{job_id});

    const max_poll_attempts: usize = 360;
    var poll_attempt: usize = 0;
    while (poll_attempt < max_poll_attempts) : (poll_attempt += 1) {
        const status_raw = try client.getJobStatus(job_id);
        defer allocator.free(status_raw);

        std.debug.print("Job status raw: {s}\n", .{status_raw});

        const parsed_status = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            status_raw,
            .{ .allocate = .alloc_always },
        ) catch null;

        if (parsed_status) |ps| {
            defer ps.deinit();
            const status_field: ?[]const u8 = switch (ps.value) {
                .object => |obj| blk: {
                    const sv = obj.get("status") orelse break :blk null;
                    break :blk switch (sv) {
                        .string => |s| s,
                        else => null,
                    };
                },
                else => null,
            };
            if (status_field) |sf| {
                std.debug.print("Job status: {s}\n", .{sf});
                if (std.mem.eql(u8, sf, "completed") or std.mem.eql(u8, sf, "failed")) {
                    break;
                }
            }
        }

        std.time.sleep(30 * std.time.ns_per_s);
    }

    if (poll_attempt >= max_poll_attempts) {
        std.debug.print("Timeout waiting for Modal job completion after {d} polls\n", .{max_poll_attempts});
        return error.ModalJobTimeout;
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1 and std.mem.eql(u8, args[1], "--deploy")) {
        return deployToModal(allocator, args[2..]);
    }

    const world_size = try std.process.getEnvVarOwned(allocator, "WORLD_SIZE");
    defer allocator.free(world_size);
    const world_size_val = try std.fmt.parseInt(usize, world_size, 10);

    const rank_str = try std.process.getEnvVarOwned(allocator, "RANK");
    defer allocator.free(rank_str);
    const rank = try std.fmt.parseInt(usize, rank_str, 10);

    var local_rank_str_owned: ?[]u8 = null;
    const local_rank: usize = blk: {
        local_rank_str_owned = std.process.getEnvVarOwned(allocator, "LOCAL_RANK") catch null;
        if (local_rank_str_owned) |owned| {
            break :blk try std.fmt.parseInt(usize, owned, 10);
        }
        std.debug.print("[Rank {d}] WARN: LOCAL_RANK not set, falling back to RANK ({d}) for device selection; on multi-node this may select a non-existent local GPU index\n", .{ rank, rank });
        break :blk rank;
    };
    defer if (local_rank_str_owned) |owned| allocator.free(owned);

    const master_addr = try std.process.getEnvVarOwned(allocator, "MASTER_ADDR");
    defer allocator.free(master_addr);

    const master_port = try std.process.getEnvVarOwned(allocator, "MASTER_PORT");
    defer allocator.free(master_port);

    std.debug.print("============================================================\n", .{});
    std.debug.print("JAIDE v40 Distributed Training (Futhark GPU Acceleration)\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Rank: {d}/{d}\n", .{ rank, world_size_val });
    std.debug.print("Master addr/port (env): {s}:{s} (used by external rendezvous; NCCL rendezvous via file-based ID exchange)\n", .{ master_addr, master_port });
    std.debug.print("============================================================\n\n", .{});

    var nccl_id_path_owned: ?[]u8 = null;
    const nccl_id_path: []const u8 = blk: {
        nccl_id_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_NCCL_ID_PATH") catch null;
        break :blk nccl_id_path_owned orelse "/tmp/jaide_nccl_id";
    };
    defer if (nccl_id_path_owned) |owned| allocator.free(owned);

    std.debug.print("[Rank {d}] WARN: NCCL ID exchange uses file path '{s}'; this requires a shared filesystem across all nodes for multi-node operation\n", .{ rank, nccl_id_path });

    const nccl_ready_path = try std.fmt.allocPrint(allocator, "{s}.ready", .{nccl_id_path});
    defer allocator.free(nccl_ready_path);

    var nccl_id: nccl.ncclUniqueId = undefined;

    if (rank == 0) {
        const result = nccl.ncclGetUniqueId(&nccl_id);
        if (result != .ncclSuccess) {
            std.debug.print("Failed to generate NCCL ID\n", .{});
            return error.NCCLGetUniqueIdFailed;
        }

        std.fs.deleteFileAbsolute(nccl_id_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        std.fs.deleteFileAbsolute(nccl_ready_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };

        {
            const id_file = try std.fs.createFileAbsolute(nccl_id_path, .{});
            defer id_file.close();
            try id_file.writeAll(std.mem.asBytes(&nccl_id));
            try id_file.sync();
        }

        {
            const ready_file = try std.fs.createFileAbsolute(nccl_ready_path, .{});
            defer ready_file.close();
            try ready_file.writeAll("ready");
            try ready_file.sync();
        }

        std.debug.print("[Rank 0] Generated NCCL ID (file: {s})\n", .{nccl_id_path});
    } else {
        var attempts: usize = 0;
        while (attempts < 100) : (attempts += 1) {
            const ready_file = std.fs.openFileAbsolute(nccl_ready_path, .{}) catch {
                std.time.sleep(100 * std.time.ns_per_ms);
                continue;
            };
            ready_file.close();
            break;
        }

        if (attempts >= 100) {
            std.debug.print("[Rank {d}] Timeout waiting for NCCL ID from rank 0 (waited ~{d}ms)\n", .{ rank, 100 * 100 });
            return error.NCCLIdTimeout;
        }

        const id_file = try std.fs.openFileAbsolute(nccl_id_path, .{});
        defer id_file.close();

        const bytes_read = try id_file.readAll(std.mem.asBytes(&nccl_id));
        if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
            std.debug.print("[Rank {d}] Failed to read NCCL ID (got {d} bytes, expected {d})\n", .{ rank, bytes_read, @sizeOf(nccl.ncclUniqueId) });
            return error.NCCLIdReadFailed;
        }

        std.debug.print("[Rank {d}] Loaded NCCL ID from rank 0\n", .{rank});
    }

    var coordinator = try GPUCoordinator.init(allocator, world_size_val, rank, local_rank, nccl_id);
    defer coordinator.deinit();

    std.debug.print("[Rank {d}] GPU coordinator initialized\n", .{rank});

    var model_dim_str_owned: ?[]u8 = null;
    defer if (model_dim_str_owned) |owned| allocator.free(owned);
    model_dim_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_MODEL_DIM") catch null;
    const model_dim: usize = if (model_dim_str_owned) |s| blk: {
        break :blk std.fmt.parseInt(usize, s, 10) catch |err| {
            std.debug.print("[Rank {d}] ERROR: invalid JAIDE_MODEL_DIM='{s}': {}\n", .{ rank, s, err });
            return error.InvalidConfig;
        };
    } else 2048;
    if (model_dim == 0) {
        std.debug.print("[Rank {d}] ERROR: JAIDE_MODEL_DIM must be > 0\n", .{rank});
        return error.InvalidConfig;
    }

    var num_layers_str_owned: ?[]u8 = null;
    defer if (num_layers_str_owned) |owned| allocator.free(owned);
    num_layers_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_LAYERS") catch null;
    const num_layers: usize = if (num_layers_str_owned) |s| blk: {
        break :blk std.fmt.parseInt(usize, s, 10) catch |err| {
            std.debug.print("[Rank {d}] ERROR: invalid JAIDE_LAYERS='{s}': {}\n", .{ rank, s, err });
            return error.InvalidConfig;
        };
    } else 24;
    if (num_layers == 0) {
        std.debug.print("[Rank {d}] ERROR: JAIDE_LAYERS must be > 0\n", .{rank});
        return error.InvalidConfig;
    }

    var local_batch_size_str_owned: ?[]u8 = null;
    defer if (local_batch_size_str_owned) |owned| allocator.free(owned);
    local_batch_size_str_owned = std.process.getEnvVarOwned(allocator, "JAIDE_BATCH_SIZE") catch null;
    const local_batch_size: usize = if (local_batch_size_str_owned) |s| blk: {
        break :blk std.fmt.parseInt(usize, s, 10) catch |err| {
            std.debug.print("[Rank {d}] ERROR: invalid JAIDE_BATCH_SIZE='{s}': {}\n", .{ rank, s, err });
            return error.InvalidConfig;
        };
    } else 4;
    if (local_batch_size == 0) {
        std.debug.print("[Rank {d}] ERROR: JAIDE_BATCH_SIZE must be > 0\n", .{rank});
        return error.InvalidConfig;
    }

    var epochs_env_owned: ?[]u8 = null;
    defer if (epochs_env_owned) |owned| allocator.free(owned);
    epochs_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_EPOCHS") catch null;
    const num_epochs: usize = if (epochs_env_owned) |s| blk: {
        break :blk std.fmt.parseInt(usize, s, 10) catch |err| {
            std.debug.print("[Rank {d}] ERROR: invalid JAIDE_EPOCHS='{s}': {}\n", .{ rank, s, err });
            return error.InvalidConfig;
        };
    } else 20;

    var lr_env_owned: ?[]u8 = null;
    defer if (lr_env_owned) |owned| allocator.free(owned);
    lr_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_LEARNING_RATE") catch null;
    const learning_rate: f32 = if (lr_env_owned) |s| blk: {
        const v = std.fmt.parseFloat(f32, s) catch |err| {
            std.debug.print("[Rank {d}] ERROR: invalid JAIDE_LEARNING_RATE='{s}': {}\n", .{ rank, s, err });
            return error.InvalidConfig;
        };
        break :blk v;
    } else 0.0001;
    if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) {
        std.debug.print("[Rank {d}] ERROR: JAIDE_LEARNING_RATE must be finite and positive, got {d}\n", .{ rank, learning_rate });
        return error.InvalidConfig;
    }

    var dataset_path_owned: ?[]u8 = null;
    defer if (dataset_path_owned) |owned| allocator.free(owned);
    dataset_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_DATASET") catch null;
    const dataset_path: []const u8 = dataset_path_owned orelse "/data/dataset/train.jsonl";

    std.debug.print("[Rank {d}] Loading dataset from {s}\n", .{ rank, dataset_path });

    const samples = try loadDataset(allocator, &coordinator, dataset_path, 10 * 1024 * 1024);
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    const vocab_path = "/checkpoints/tokenizer.vocab";

    var vocab_ready_env_owned: ?[]u8 = null;
    defer if (vocab_ready_env_owned) |owned| allocator.free(owned);
    vocab_ready_env_owned = std.process.getEnvVarOwned(allocator, "JAIDE_VOCAB_READY") catch null;
    const vocab_ready: bool = if (vocab_ready_env_owned) |s| std.mem.eql(u8, s, "1") else false;

    if (vocab_ready) {
        std.debug.print("[Rank {d}] JAIDE_VOCAB_READY=1: skipping BPE training, reusing existing vocab at {s}\n", .{ rank, vocab_path });
    } else {
        var temp_tokenizer = try MGT.init(allocator, &.{}, &.{}, 32000, .english);
        defer temp_tokenizer.deinit();

        if (coordinator.isRoot()) {
            std.debug.print("[Rank 0] WARN: BPE vocabulary is trained only on rank 0 shard; other rank shards are excluded from vocabulary training\n", .{});
            std.debug.print("[Rank 0] WARN: vocab broadcast assumes shared filesystem at '{s}'; non-root ranks must be able to read this path\n", .{vocab_path});

            temp_tokenizer.trainBPE(samples, 32000) catch |err| {
                std.debug.print("[Rank 0] ERROR: trainBPE failed: {} (synchronizing before exit)\n", .{err});
                coordinator.synchronize() catch {};
                return err;
            };
            std.fs.makeDirAbsolute("/checkpoints") catch |e| switch (e) {
                error.PathAlreadyExists => {},
                else => {
                    coordinator.synchronize() catch {};
                    return e;
                },
            };
            temp_tokenizer.saveVocab(vocab_path) catch |err| {
                std.debug.print("[Rank 0] ERROR: saveVocab failed: {} (synchronizing before exit)\n", .{err});
                coordinator.synchronize() catch {};
                return err;
            };
            std.debug.print("[Rank 0] Tokenizer trained and saved to {s}\n", .{vocab_path});
        }

        try coordinator.synchronize();
    }

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, 32000, .english);
    var trainer = blk: {
        errdefer tokenizer.deinit();
        try tokenizer.loadVocab(vocab_path);
        std.debug.print("[Rank {d}] Tokenizer loaded, next_token_id={d}\n", .{ rank, tokenizer.next_token_id });

        var trainer_cfg: TrainerConfig = .{};
        trainer_cfg.learning_rate = learning_rate;

        const components = TrainerComponents{
            .tokenizer = tokenizer,
            .embedding_accel = null,
        };

        break :blk try DistributedTrainerFuthark.initWithComponents(
            allocator,
            &coordinator,
            model_dim,
            num_layers,
            local_batch_size,
            trainer_cfg,
            components,
        );
    };
    defer trainer.deinit();

    std.debug.print("[Rank {d}] learning_rate={d}\n", .{ rank, learning_rate });
    std.debug.print("[Rank {d}] Futhark-accelerated trainer initialized (model_dim={d}, layers={d}, embedding_accel=disabled)\n", .{ rank, model_dim, num_layers });

    if (coordinator.isRoot()) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Starting Futhark-accelerated training\n", .{});
        std.debug.print("Dataset: {d} samples (per rank)\n", .{samples.len});
        std.debug.print("Batch size: {d} (per rank)\n", .{local_batch_size});
        std.debug.print("Epochs: {d}\n", .{num_epochs});
        std.debug.print("============================================================\n\n", .{});
    }

    {
        const n_kg_cpus: usize = std.Thread.getCpuCount() catch 4;
        const n_kg_workers: usize = @min(n_kg_cpus, @max(1, samples.len));
        const kg_ctxs = try allocator.alloc(KGWorkerCtx, n_kg_workers);
        defer allocator.free(kg_ctxs);
        const kg_threads = try allocator.alloc(std.Thread, n_kg_workers);
        defer allocator.free(kg_threads);

        const kg_base = samples.len / n_kg_workers;
        const kg_rem = samples.len % n_kg_workers;
        var kg_off: usize = 0;
        for (kg_ctxs, 0..) |*ctx, wi| {
            const chunk = kg_base + (if (wi < kg_rem) @as(usize, 1) else @as(usize, 0));
            ctx.* = .{
                .samples = samples[kg_off .. kg_off + chunk],
                .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator),
                .local_nsir = undefined,
                .err = null,
            };
            kg_off += chunk;
        }

        var spawned: usize = 0;
        for (kg_ctxs, 0..) |*ctx, wi| {
            kg_threads[wi] = std.Thread.spawn(.{}, kgWorkerFn, .{ctx}) catch |spawn_err| {
                for (kg_threads[0..spawned]) |t| t.join();
                for (kg_ctxs) |*c| c.arena.deinit();
                return spawn_err;
            };
            spawned += 1;
        }
        for (kg_threads[0..n_kg_workers]) |thread| thread.join();

        for (kg_ctxs) |*ctx| {
            defer ctx.arena.deinit();
            if (ctx.err) |worker_err| {
                std.debug.print("[Rank {d}] WARN: KG worker failed: {} (skipping worker shard)\n", .{ rank, worker_err });
                continue;
            }
            var node_it = ctx.local_nsir.nodes.iterator();
            while (node_it.next()) |entry| {
                const n = entry.value_ptr.*;
                const new_node = core_relational.nsir_core.Node.init(allocator, n.id, n.data, n.qubit, n.phase) catch |err| {
                    std.debug.print("[Rank {d}] WARN: KG Node.init failed: {} (skipping node)\n", .{ rank, err });
                    continue;
                };
                trainer.nsir_graph.addNode(new_node) catch |err| {
                    std.debug.print("[Rank {d}] WARN: KG addNode failed: {} (skipping node)\n", .{ rank, err });
                };
            }
        }

        trainer.r_gpu.distributeGraph(trainer.nsir_graph) catch |err| {
            std.debug.print("[Rank {d}] WARN: r_gpu.distributeGraph (KG) failed: {} (continuing)\n", .{ rank, err });
        };
        trainer.signal_engine.propagateStep() catch |err| {
            std.debug.print("[Rank {d}] WARN: signal_engine.propagateStep (KG) failed: {} (continuing)\n", .{ rank, err });
        };
    }

    if (coordinator.isRoot()) {
        std.debug.print("[Rank 0] Knowledge graph populated.\n", .{});
    }

    var loss_history = std.ArrayList(EpochMetric).init(allocator);
    defer loss_history.deinit();

    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        var epoch_timer = try std.time.Timer.start();

        const avg_loss = trainer.trainEpoch(samples) catch |err| {
            std.debug.print("[Rank {d}] trainEpoch ERROR (epoch={d}): {} (synchronizing before exit)\n", .{ rank, epoch + 1, err });
            coordinator.synchronize() catch {};
            return err;
        };

        const elapsed_ns = epoch_timer.read();
        const elapsed = @as(f64, @floatFromInt(elapsed_ns)) / 1.0e9;

        var root_work_err: ?anyerror = null;

        if (coordinator.isRoot()) {
            root_epoch_work: {
                loss_history.append(.{ .epoch = epoch + 1, .loss = avg_loss, .time_s = elapsed }) catch |err| {
                    std.debug.print("[Rank 0] ERROR: loss_history.append failed: {}\n", .{err});
                    root_work_err = err;
                    break :root_epoch_work;
                };

                std.debug.print("[Epoch {d}/{d}] Loss: {d:.6} | Time: {d:.2}s\n", .{ epoch + 1, num_epochs, avg_loss, elapsed });

                {
                    var dir_buf: [256]u8 = undefined;
                    const dir_path = std.fmt.bufPrint(&dir_buf, "/checkpoints/epoch_{d:0>3}", .{epoch + 1}) catch "/checkpoints";
                    std.fs.makeDirAbsolute("/checkpoints") catch |e| switch (e) {
                        error.PathAlreadyExists => {},
                        else => std.debug.print("[Rank 0] makeDirAbsolute(/checkpoints) failed: {} (continuing)\n", .{e}),
                    };
                    std.fs.makeDirAbsolute(dir_path) catch |e| switch (e) {
                        error.PathAlreadyExists => {},
                        else => std.debug.print("[Rank 0] makeDirAbsolute({s}) failed: {} (continuing)\n", .{ dir_path, e }),
                    };

                    var checkpoint_path_buf: [256]u8 = undefined;
                    const checkpoint_path = std.fmt.bufPrint(
                        &checkpoint_path_buf,
                        "/checkpoints/epoch_{d:0>3}/model.ckpt",
                        .{epoch + 1},
                    ) catch {
                        root_work_err = error.NoSpaceLeft;
                        break :root_epoch_work;
                    };

                    trainer.saveCheckpoint(checkpoint_path) catch |err| {
                        std.debug.print("[Rank 0] ERROR: saveCheckpoint failed: {}\n", .{err});
                        root_work_err = err;
                        break :root_epoch_work;
                    };
                    std.debug.print("  Checkpoint saved: {s}\n", .{checkpoint_path});
                }

                writeTrainingMetrics(
                    allocator,
                    loss_history.items,
                    model_dim,
                    num_layers,
                    local_batch_size,
                    learning_rate,
                    samples.len,
                    num_epochs,
                ) catch |err| {
                    std.debug.print("[Rank 0] ERROR: writeTrainingMetrics failed: {}\n", .{err});
                    root_work_err = err;
                    break :root_epoch_work;
                };
            }
        }

        if (root_work_err) |rwe| {
            std.debug.print("[Rank {d}] Root epoch work failed: {} (synchronizing before exit)\n", .{ rank, rwe });
            coordinator.synchronize() catch {};
            return rwe;
        }

        try coordinator.synchronize();
    }

    if (coordinator.isRoot() and num_epochs > 0) {
        std.debug.print("\n============================================================\n", .{});
        std.debug.print("Futhark-accelerated training completed successfully!\n", .{});
        std.debug.print("Final checkpoint saved to /checkpoints/\n", .{});
        std.debug.print("============================================================\n", .{});
    }
}

const EpochMetric = struct { epoch: usize, loss: f64, time_s: f64 };

const KGWorkerCtx = struct {
    samples: []const []const u8,
    arena: std.heap.ArenaAllocator,
    local_nsir: core_relational.SelfSimilarRelationalGraph,
    err: ?anyerror,
};

fn kgWorkerFn(ctx: *KGWorkerCtx) void {
    const a = ctx.arena.allocator();
    var chaos = core_relational.ChaosCoreKernel.init(a);
    var local_crev = core_relational.CREVPipeline.init(a, &chaos) catch |err| {
        ctx.err = err;
        return;
    };
    var local_nsir = core_relational.SelfSimilarRelationalGraph.init(a) catch |err| {
        ctx.err = err;
        return;
    };
    for (ctx.samples) |text| {
        if (text.len == 0) continue;
        _ = local_crev.processTextStream(text) catch |err| {
            ctx.err = err;
            return;
        };
        _ = local_nsir.encodeInformation(std.mem.sliceAsBytes(text)) catch |err| {
            ctx.err = err;
            return;
        };
    }
    ctx.local_nsir = local_nsir;
}

fn writeTrainingMetrics(
    allocator: std.mem.Allocator,
    epoch_metrics: []const EpochMetric,
    model_dim: usize,
    num_layers: usize,
    batch_size: usize,
    learning_rate: f64,
    sample_count: usize,
    planned_epochs: usize,
) !void {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    const safe_lr: f64 = if (std.math.isFinite(learning_rate)) learning_rate else 0.0;

    try writer.print(
        "{{\n  \"model_dim\": {d},\n  \"num_layers\": {d},\n  \"batch_size\": {d},\n  \"learning_rate\": {d},\n  \"sample_count\": {d},\n  \"planned_epochs\": {d},\n  \"loss_curve\": [\n",
        .{ model_dim, num_layers, batch_size, safe_lr, sample_count, planned_epochs },
    );

    for (epoch_metrics, 0..) |m, i| {
        const safe_loss: f64 = if (std.math.isFinite(m.loss)) m.loss else 0.0;
        const safe_time: f64 = if (std.math.isFinite(m.time_s)) m.time_s else 0.0;
        try writer.print(
            "    {{ \"epoch\": {d}, \"loss\": {d:.6}, \"time_s\": {d:.2} }}{s}\n",
            .{ m.epoch, safe_loss, safe_time, if (i + 1 < epoch_metrics.len) "," else "" },
        );
    }

    try writer.print("  ]\n}}\n", .{});

    const tmp_path = "/checkpoints/training_metrics.json.tmp";
    {
        const tmp_file = try std.fs.createFileAbsolute(tmp_path, .{ .truncate = true });
        errdefer tmp_file.close();
        try tmp_file.writeAll(buf.items);
        try tmp_file.sync();
        tmp_file.close();
    }
    try std.fs.renameAbsolute(tmp_path, "/checkpoints/training_metrics.json");
}
