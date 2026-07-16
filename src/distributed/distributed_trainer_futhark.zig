const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const FutharkArray1DF16 = accel.FutharkArray1DF16;
const FutharkArray3DF16 = accel.FutharkArray3DF16;
const PinnedMemory = accel.PinnedMemory;
const LearnedEmbedding = @import("../core/learned_embedding.zig").LearnedEmbedding;
const EmbeddingAccelerator = accel.EmbeddingAccelerator;
const futhark = @import("../hw/accel/futhark_bindings.zig");
const core_relational = @import("../core_relational/mod.zig");
const CREVPipeline = core_relational.CREVPipeline;
const ChaosCoreKernel = core_relational.ChaosCoreKernel;
const nsir = core_relational.nsir_core;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const EntangledStochasticSymmetryOptimizer = core_relational.EntangledStochasticSymmetryOptimizer;
const SurpriseMemoryManager = core_relational.SurpriseMemoryManager;
const TemporalGraph = core_relational.TemporalGraph;
const QuantumState = core_relational.QuantumState;
const ReasoningOrchestrator = core_relational.ReasoningOrchestrator;
const SignalPropagationEngine = core_relational.SignalPropagationEngine;
const ZRuntime = core_relational.ZRuntime;
const RelationalGraphProcessingUnit = core_relational.RelationalGraphProcessingUnit;
const FNDSManager = core_relational.FNDSManager;
const VPU = core_relational.VPU;
const PatternLocation = core_relational.PatternLocation;
const Tensor = @import("../core/tensor.zig").Tensor;
const sfd = @import("../optimizer/sfd.zig");

const _referenced_futhark_2d = FutharkArray2DF16;
const _referenced_futhark_1d = FutharkArray1DF16;
const _referenced_core_tensor = Tensor;

pub const CHECKPOINT_MAGIC: [8]u8 = .{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
pub const CHECKPOINT_TRAILER: u32 = 0xDEADBEEF;

pub const MGTLanguage = @typeInfo(@TypeOf(MGT.init)).@"fn".params[4].type.?;

pub const TokenizerFactory = *const fn (
    allocator: std.mem.Allocator,
    vocabulary: []const []const u8,
    anchors: []const []const u8,
    max_merges: usize,
) anyerror!MGT;

const OwnedTokenList = struct {
    allocator: std.mem.Allocator,
    items: [][]const u8,

    fn deinit(self: *OwnedTokenList) void {
        for (self.items) |item| self.allocator.free(item);
        self.allocator.free(self.items);
    }
};

fn appendOwnedToken(
    allocator: std.mem.Allocator,
    items: *std.ArrayList([]const u8),
    token: []const u8,
) !void {
    if (token.len == 0) return TrainerError.InvalidTokenizerData;
    const owned = try allocator.dupe(u8, token);
    items.append(owned) catch |err| {
        allocator.free(owned);
        return err;
    };
}

fn loadTokenList(
    allocator: std.mem.Allocator,
    path: []const u8,
    maximum_size: usize,
) !OwnedTokenList {
    if (path.len == 0 or maximum_size == 0) return TrainerError.InvalidTokenizerData;
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{ .mode = .read_only })
    else
        try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    const length_u64 = try file.getEndPos();
    const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
    if (length == 0 or length > maximum_size) return TrainerError.InvalidTokenizerData;
    const data = try allocator.alloc(u8, length);
    defer allocator.free(data);
    try file.reader().readNoEof(data);

    var items = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (items.items) |item| allocator.free(item);
        items.deinit();
    }

    var content = std.mem.trimLeft(u8, data, " \t\r\n");
    if (content.len >= 3 and std.mem.eql(u8, content[0..3], "\xEF\xBB\xBF")) {
        content = std.mem.trimLeft(u8, content[3..], " \t\r\n");
    }
    if (content.len == 0) return TrainerError.InvalidTokenizerData;

    if (content[0] == '[') {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            content,
            .{ .allocate = .alloc_always },
        );
        defer parsed.deinit();
        switch (parsed.value) {
            .array => |array| {
                for (array.items) |value| {
                    switch (value) {
                        .string => |token| try appendOwnedToken(allocator, &items, token),
                        else => return TrainerError.InvalidTokenizerData,
                    }
                }
            },
            else => return TrainerError.InvalidTokenizerData,
        }
    } else {
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |raw_line| {
            const line = if (raw_line.len > 0 and raw_line[raw_line.len - 1] == '\r')
                raw_line[0 .. raw_line.len - 1]
            else
                raw_line;
            if (line.len == 0) continue;
            try appendOwnedToken(allocator, &items, line);
        }
    }

    if (items.items.len == 0) return TrainerError.InvalidTokenizerData;
    return OwnedTokenList{
        .allocator = allocator,
        .items = try items.toOwnedSlice(),
    };
}

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    max_line_size: usize = 10 * 1024 * 1024,
    max_tokenizer_file_size: usize = 1024 * 1024 * 1024,
    checkpoint_version: u32 = 9,
    reasoning_cycles: usize = 50,
    fnds_max_depth: usize = 6,
    fnds_branching: usize = 4,
    fnds_kg_max_depth: usize = 4,
    fnds_kg_branching: usize = 3,
    training_fnds_index_name: []const u8 = "distributed_training_tokens",
    knowledge_fnds_index_name: []const u8 = "knowledge_graph_patterns",
    embedding_seed: u64 = 42,
    spectral_iterations: usize = 5,
    gradient_clip_norm: f32 = 1.0,
    use_normalized_gradient_flow: bool = true,
    default_max_seq_len: usize = 256,
    temporal_sequence_tick: i64 = 1,
    training_variable_name: []const u8 = "distributed_training_session",
    max_id_length: usize = 1 << 20,
    max_edge_group_count: usize = 1 << 24,
    max_node_data_length: usize = 1 << 24,
    max_distributed_integer: u64 = 16_777_216,
    tokenizer_vocabulary: []const []const u8 = &.{},
    tokenizer_anchors: []const []const u8 = &.{},
    tokenizer_vocabulary_path: ?[]const u8 = null,
    tokenizer_anchors_path: ?[]const u8 = null,
    tokenizer_max_merges: usize = 50_000,
    tokenizer_language: ?MGTLanguage = null,
    tokenizer_factory: ?TokenizerFactory = null,
    esso_initial_temperature: f64 = 1.0,
    esso_cooling_rate: f64 = 0.995,
    esso_max_iterations: usize = 1000,
    relational_gpu_rows: usize = 4,
    relational_gpu_columns: usize = 4,
};

pub const TrainerComponents = struct {
    tokenizer: MGT,
    embedding_accel: ?EmbeddingAccelerator = null,
};

pub const TrainerError = error{
    InvalidModelDim,
    InvalidNumLayers,
    InvalidBatchSize,
    InvalidWorldSize,
    InvalidRank,
    InvalidMaxLineSize,
    InvalidCheckpointVersion,
    InvalidLearningRate,
    InvalidMomentum,
    InvalidHyperparameterAfterCast,
    InvalidWeightsShape,
    InvalidWeightValue,
    InvalidClipRange,
    InvalidLoss,
    InvalidPinnedMemorySize,
    IndexOutOfBounds,
    TokenIndexOutOfRange,
    CheckpointVersionMismatch,
    CheckpointMagicMismatch,
    CheckpointCorrupted,
    ModelDimMismatch,
    NumLayersMismatch,
    VocabSizeMismatch,
    EmptyDataset,
    DatasetSampleCountMismatch,
    InvalidDatasetPartition,
    InvalidEnvironmentValue,
    ValueOverflow,
    ConvertPrecisionLoss,
    AllocationFailed,
    InvalidQualityByte,
    NodeIdTooLong,
    NodeDataTooLong,
    EdgeCountTooLarge,
    FloatToIntegerConversionFailed,
    DistributedIntegerPrecisionExceeded,
    InvalidDistributedInteger,
    InvalidReductionWeight,
    InvalidFloat16Value,
    InvalidGradient,
    InvalidSimilarityValue,
    InvalidQuantumState,
    InvalidEmbeddingWeight,
    InvalidEmbeddingShape,
    InvalidEdgeWeight,
    InvalidEdgeQuality,
    InvalidGraphIdentifier,
    InvalidGraphSize,
    InvalidTokenizerData,
    InvalidTokenizerConfiguration,
    InvalidOptimizerConfiguration,
    InvalidRelationalGPUConfiguration,
    InvalidTemporalConfiguration,
    InvalidSpectralState,
    InvalidCheckpointEmbeddingFlag,
    TrailingCheckpointData,
    TimestampOutOfRange,
    FileTooLarge,
    FutharkContextUnavailable,
    FutharkForwardFailed,
    FutharkTransformFailed,
    FutharkBackwardTransformFailed,
    FutharkGradientFailed,
    FutharkFullGradientFailed,
    FutharkProjectionFailed,
    FutharkGradientCopyFailed,
    EmptyKnowledgeGraphInput,
};

fn createConfiguredTokenizer(
    allocator: std.mem.Allocator,
    config: TrainerConfig,
) !MGT {
    if (config.tokenizer_max_merges == 0 or config.max_tokenizer_file_size == 0) return TrainerError.InvalidTokenizerConfiguration;

    var vocabulary_path_owned: ?[]u8 = null;
    defer if (vocabulary_path_owned) |path| allocator.free(path);
    var anchors_path_owned: ?[]u8 = null;
    defer if (anchors_path_owned) |path| allocator.free(path);

    var owned_vocabulary: ?OwnedTokenList = null;
    defer if (owned_vocabulary) |*list| list.deinit();
    var owned_anchors: ?OwnedTokenList = null;
    defer if (owned_anchors) |*list| list.deinit();

    var vocabulary = config.tokenizer_vocabulary;
    if (vocabulary.len == 0) {
        const vocabulary_path = config.tokenizer_vocabulary_path orelse blk: {
            vocabulary_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_VOCAB") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
                else => return err,
            };
            break :blk vocabulary_path_owned.?;
        };
        owned_vocabulary = try loadTokenList(allocator, vocabulary_path, config.max_tokenizer_file_size);
        vocabulary = owned_vocabulary.?.items;
    }
    if (vocabulary.len == 0) return TrainerError.InvalidTokenizerData;

    var anchors = config.tokenizer_anchors;
    if (anchors.len == 0) {
        const configured_anchor_path = config.tokenizer_anchors_path;
        const anchor_path_opt: ?[]const u8 = if (configured_anchor_path) |path|
            path
        else blk: {
            anchors_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_ANCHORS") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => break :blk null,
                else => return err,
            };
            break :blk anchors_path_owned.?;
        };
        if (anchor_path_opt) |anchor_path| {
            owned_anchors = try loadTokenList(allocator, anchor_path, config.max_tokenizer_file_size);
            anchors = owned_anchors.?.items;
        }
    }

    if (config.tokenizer_factory) |tokenizer_factory| {
        return tokenizer_factory(allocator, vocabulary, anchors, config.tokenizer_max_merges);
    }

    var language_name_owned: ?[]u8 = null;
    defer if (language_name_owned) |name| allocator.free(name);
    const language = config.tokenizer_language orelse blk: {
        language_name_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_LANGUAGE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
            else => return err,
        };
        break :blk std.meta.stringToEnum(MGTLanguage, language_name_owned.?) orelse return TrainerError.InvalidTokenizerConfiguration;
    };
    return MGT.init(allocator, vocabulary, anchors, config.tokenizer_max_merges, language);
}

const LayerSnapshot = struct {
    weights_s: []f16,
    weights_t: []f16,
    velocity_s: []f16,
    velocity_t: []f16,
};

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: *RSFAccelerator,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: u64,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,
    embedding: ?LearnedEmbedding,
    embedding_accel: ?EmbeddingAccelerator,
    crev_pipeline: CREVPipeline,
    crev_kernel: *ChaosCoreKernel,
    nsir_graph: *SelfSimilarRelationalGraph,
    esso: EntangledStochasticSymmetryOptimizer,
    surprise_memory: SurpriseMemoryManager,
    temporal_graph: TemporalGraph,
    signal_engine: *SignalPropagationEngine,
    z_runtime: *ZRuntime,
    r_gpu: RelationalGraphProcessingUnit,
    fnds_manager: FNDSManager,
    vpu: VPU,
    spectral_normalizer: sfd.SpectralNormalizer,
    spectral_u: ?[]f32,
    spectral_v: ?[]f32,
    training_fnds_tree_id: ?[32]u8,
    training_fnds_index_id: ?[]u8,
    knowledge_fnds_tree_id: ?[32]u8,
    knowledge_fnds_index_id: ?[]u8,
    knowledge_graph_nonce: [32]u8,
    temporal_logical_time: i64,
    training_variable_created: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, 1, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        const tokenizer = try createConfiguredTokenizer(allocator, config);
        return initWithComponents(allocator, coordinator, model_dim, num_layers, local_batch_size, config, .{
            .tokenizer = tokenizer,
            .embedding_accel = null,
        });
    }

    pub fn initWithComponents(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
        components_in: TrainerComponents,
    ) !DistributedTrainerFuthark {
        var components = components_in;
        var tokenizer_transferred = false;
        errdefer if (!tokenizer_transferred) {
            var t = components.tokenizer;
            t.deinit();
        };
        var embedding_accel_transferred = false;
        errdefer if (!embedding_accel_transferred) {
            if (components.embedding_accel) |ea| {
                var e = ea;
                e.deinit();
            }
        };

        if (model_dim == 0 or model_dim % 2 != 0) return TrainerError.InvalidModelDim;
        if (num_layers == 0) return TrainerError.InvalidNumLayers;
        if (local_batch_size == 0) return TrainerError.InvalidBatchSize;
        if (coordinator.world_size == 0) return TrainerError.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return TrainerError.InvalidRank;
        if (config.max_line_size == 0) return TrainerError.InvalidMaxLineSize;
        if (config.checkpoint_version == 0) return TrainerError.InvalidCheckpointVersion;
        if (!std.math.isFinite(config.esso_initial_temperature) or config.esso_initial_temperature <= 0.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.esso_cooling_rate) or config.esso_cooling_rate <= 0.0 or config.esso_cooling_rate > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.esso_max_iterations == 0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.relational_gpu_rows == 0 or config.relational_gpu_columns == 0) return TrainerError.InvalidRelationalGPUConfiguration;
        if (config.temporal_sequence_tick <= 0) return TrainerError.InvalidTemporalConfiguration;
        if (config.training_fnds_index_name.len == 0 or config.knowledge_fnds_index_name.len == 0) return TrainerError.InvalidGraphIdentifier;
        if (std.mem.eql(u8, config.training_fnds_index_name, config.knowledge_fnds_index_name)) return TrainerError.InvalidGraphIdentifier;
        if (!std.math.isFinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0.0) return TrainerError.InvalidGradient;
        try validateHyperparameters(config.learning_rate, config.momentum);

        const actual_model_dim = model_dim;

        const accelerator_ptr = try allocator.create(RSFAccelerator);
        var accelerator_ptr_committed = false;
        errdefer if (!accelerator_ptr_committed) allocator.destroy(accelerator_ptr);
        accelerator_ptr.* = try RSFAccelerator.initMultiLayer(actual_model_dim, num_layers, allocator);
        var accelerator_committed = false;
        errdefer if (!accelerator_committed) accelerator_ptr.deinit();

        var embedding = try LearnedEmbedding.init(
            allocator,
            components.tokenizer.next_token_id,
            actual_model_dim,
            config.embedding_seed,
        );
        var embedding_committed = false;
        errdefer if (!embedding_committed) embedding.deinit();

        var embedding_accel: ?EmbeddingAccelerator = null;
        if (components.embedding_accel) |ea| {
            embedding_accel = ea;
            components.embedding_accel = null;
            embedding_accel_transferred = true;
        } else {
            embedding_accel = EmbeddingAccelerator.init(
                &accelerator_ptr.ctx,
                components.tokenizer.next_token_id,
                actual_model_dim,
                config.embedding_seed,
            ) catch |err| blk: {
                std.debug.print("[Rank {d}] WARN: EmbeddingAccelerator.init failed: {} (continuing without accelerator)\n", .{ coordinator.rank, err });
                break :blk null;
            };
            embedding_accel_transferred = true;
        }
        var embedding_accel_committed = false;
        errdefer if (!embedding_accel_committed) {
            if (embedding_accel) |*ea| ea.deinit();
        };

        const crev_kernel_ptr = try allocator.create(ChaosCoreKernel);
        var crev_kernel_ptr_committed = false;
        errdefer if (!crev_kernel_ptr_committed) allocator.destroy(crev_kernel_ptr);
        crev_kernel_ptr.* = ChaosCoreKernel.init(allocator);
        var crev_kernel_committed = false;
        errdefer if (!crev_kernel_committed) crev_kernel_ptr.deinit();

        var crev_pipeline = try CREVPipeline.init(allocator, crev_kernel_ptr);
        var crev_pipeline_committed = false;
        errdefer if (!crev_pipeline_committed) crev_pipeline.deinit();

        const nsir_graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var nsir_graph_ptr_committed = false;
        errdefer if (!nsir_graph_ptr_committed) allocator.destroy(nsir_graph_ptr);
        nsir_graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var nsir_graph_committed = false;
        errdefer if (!nsir_graph_committed) nsir_graph_ptr.deinit();

        var esso = EntangledStochasticSymmetryOptimizer.init(
            allocator,
            config.esso_initial_temperature,
            config.esso_cooling_rate,
            config.esso_max_iterations,
        );
        var esso_committed = false;
        errdefer if (!esso_committed) esso.deinit();

        var surprise_memory = SurpriseMemoryManager.init(
            allocator,
            &crev_kernel_ptr.storage,
            &crev_kernel_ptr.flow_analyzer,
        );
        var surprise_memory_committed = false;
        errdefer if (!surprise_memory_committed) surprise_memory.deinit();

        var temporal_graph_inst = TemporalGraph.init(allocator);
        var temporal_graph_committed = false;
        errdefer if (!temporal_graph_committed) temporal_graph_inst.deinit();

        const z_runtime_ptr = try ZRuntime.init(allocator);
        var z_runtime_committed = false;
        errdefer if (!z_runtime_committed) z_runtime_ptr.deinit();

        var r_gpu_inst = try RelationalGraphProcessingUnit.init(
            allocator,
            config.relational_gpu_rows,
            config.relational_gpu_columns,
        );
        var r_gpu_committed = false;
        errdefer if (!r_gpu_committed) r_gpu_inst.deinit();

        var fnds_manager_inst = try FNDSManager.init(allocator);
        var fnds_manager_committed = false;
        errdefer if (!fnds_manager_committed) fnds_manager_inst.deinit();

        var vpu_inst = try VPU.init(allocator);
        var vpu_committed = false;
        errdefer if (!vpu_committed) vpu_inst.deinit();

        const signal_engine_ptr = try allocator.create(SignalPropagationEngine);
        var signal_engine_ptr_committed = false;
        errdefer if (!signal_engine_ptr_committed) allocator.destroy(signal_engine_ptr);
        signal_engine_ptr.* = SignalPropagationEngine.init(
            allocator,
            nsir_graph_ptr,
            &crev_kernel_ptr.flow_analyzer,
        );
        var signal_engine_committed = false;
        errdefer if (!signal_engine_committed) signal_engine_ptr.deinit();

        const spectral_normalizer = sfd.SpectralNormalizer.init(config.spectral_iterations);
        var knowledge_graph_nonce: [32]u8 = undefined;
        std.crypto.random.bytes(knowledge_graph_nonce[0..]);

        tokenizer_transferred = true;
        accelerator_ptr_committed = true;
        accelerator_committed = true;
        embedding_committed = true;
        embedding_accel_committed = true;
        crev_kernel_ptr_committed = true;
        crev_kernel_committed = true;
        crev_pipeline_committed = true;
        nsir_graph_ptr_committed = true;
        nsir_graph_committed = true;
        esso_committed = true;
        surprise_memory_committed = true;
        temporal_graph_committed = true;
        z_runtime_committed = true;
        r_gpu_committed = true;
        fnds_manager_committed = true;
        vpu_committed = true;
        signal_engine_ptr_committed = true;
        signal_engine_committed = true;

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = components.tokenizer,
            .accelerator = accelerator_ptr,
            .model_dim = actual_model_dim,
            .num_layers = num_layers,
            .vocab_size = components.tokenizer.next_token_id,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
            .embedding = embedding,
            .embedding_accel = embedding_accel,
            .crev_pipeline = crev_pipeline,
            .crev_kernel = crev_kernel_ptr,
            .nsir_graph = nsir_graph_ptr,
            .esso = esso,
            .surprise_memory = surprise_memory,
            .temporal_graph = temporal_graph_inst,
            .signal_engine = signal_engine_ptr,
            .z_runtime = z_runtime_ptr,
            .r_gpu = r_gpu_inst,
            .fnds_manager = fnds_manager_inst,
            .vpu = vpu_inst,
            .spectral_normalizer = spectral_normalizer,
            .spectral_u = null,
            .spectral_v = null,
            .training_fnds_tree_id = null,
            .training_fnds_index_id = null,
            .knowledge_fnds_tree_id = null,
            .knowledge_fnds_index_id = null,
            .knowledge_graph_nonce = knowledge_graph_nonce,
            .temporal_logical_time = 0,
            .training_variable_created = false,
        };
    }

    fn resetSpectralState(self: *DistributedTrainerFuthark) void {
        if (self.spectral_u) |u| self.allocator.free(u);
        if (self.spectral_v) |v| self.allocator.free(v);
        self.spectral_u = null;
        self.spectral_v = null;
    }

    fn releaseFndsResources(self: *DistributedTrainerFuthark) void {
        if (self.training_fnds_index_id) |index_id| {
            _ = self.fnds_manager.removeIndex(index_id);
            self.allocator.free(index_id);
            self.training_fnds_index_id = null;
        }
        if (self.knowledge_fnds_index_id) |index_id| {
            _ = self.fnds_manager.removeIndex(index_id);
            self.allocator.free(index_id);
            self.knowledge_fnds_index_id = null;
        }
        if (self.training_fnds_tree_id) |tree_id| {
            _ = self.fnds_manager.removeTree(tree_id);
            self.training_fnds_tree_id = null;
        }
        if (self.knowledge_fnds_tree_id) |tree_id| {
            _ = self.fnds_manager.removeTree(tree_id);
            self.knowledge_fnds_tree_id = null;
        }
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch |err| {
            std.debug.print("[Rank {d}] WARN: accelerator.sync during deinit failed: {} (proceeding)\n", .{ self.coordinator.rank, err });
        };
        self.resetSpectralState();
        self.releaseFndsResources();
        self.vpu.deinit();
        self.fnds_manager.deinit();
        self.r_gpu.deinit();
        self.z_runtime.deinit();
        self.signal_engine.deinit();
        self.allocator.destroy(self.signal_engine);
        self.temporal_graph.deinit();
        self.surprise_memory.deinit();
        self.esso.deinit();
        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.crev_pipeline.deinit();
        self.crev_kernel.deinit();
        self.allocator.destroy(self.crev_kernel);
        if (self.embedding_accel) |*ea| ea.deinit();
        if (self.embedding) |*emb| emb.deinit();
        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.tokenizer.deinit();
    }

    pub fn rebindSignalEngine(self: *DistributedTrainerFuthark) void {
        self.signal_engine.deinit();
        self.signal_engine.* = SignalPropagationEngine.init(
            self.allocator,
            self.nsir_graph,
            &self.crev_kernel.flow_analyzer,
        );
    }

    /// Reset all unbounded relational state accumulated during training.
    /// Call once per epoch to prevent unbounded memory growth from crashing the
    /// process with SIGSEGV.  Model weights are not touched.
    fn pruneRelationalState(self: *DistributedTrainerFuthark) !void {
        // Drop and re-create the NSIR graph (encodeInformation grows it every step).
        self.nsir_graph.deinit();
        self.nsir_graph.* = try SelfSimilarRelationalGraph.init(self.allocator);

        // Drop and re-create the temporal graph (O(n²) growth: adds every nsir
        // node on every step).
        self.temporal_graph.deinit();
        self.temporal_graph = TemporalGraph.init(self.allocator);
        self.temporal_logical_time = 0;

        // Release FNDS trees/indices — they are lazily re-created on the next step.
        self.releaseFndsResources();

        // Re-bind the signal engine to the fresh graph; this also clears the
        // activation_traces hashmap that otherwise grows without bound.
        self.rebindSignalEngine();

        // Reset surprise memory so its internal record list does not grow forever.
        self.surprise_memory.deinit();
        self.surprise_memory = SurpriseMemoryManager.init(
            self.allocator,
            &self.crev_kernel.storage,
            &self.crev_kernel.flow_analyzer,
        );

        self.training_variable_created = false;
    }

    pub fn reinitEmbedding(self: *DistributedTrainerFuthark) !void {
        var new_embedding = try LearnedEmbedding.init(
            self.allocator,
            self.tokenizer.next_token_id,
            self.model_dim,
            self.config.embedding_seed,
        );
        errdefer new_embedding.deinit();

        var new_embedding_accel: ?EmbeddingAccelerator = null;
        if (self.embedding_accel != null) {
            new_embedding_accel = EmbeddingAccelerator.init(
                &self.accelerator.ctx,
                self.tokenizer.next_token_id,
                self.model_dim,
                self.config.embedding_seed,
            ) catch |err| blk: {
                std.debug.print("[Rank {d}] WARN: EmbeddingAccelerator reinit failed: {} (continuing without accelerator)\n", .{ self.coordinator.rank, err });
                break :blk null;
            };
        }
        errdefer if (new_embedding_accel) |*ea| ea.deinit();

        self.resetSpectralState();
        if (self.embedding) |*old| old.deinit();
        self.embedding = new_embedding;
        if (self.embedding_accel) |*old_accel| old_accel.deinit();
        self.embedding_accel = new_embedding_accel;
        self.vocab_size = self.tokenizer.next_token_id;
    }

    fn validateHyperparameters(learning_rate: f32, momentum: f32) TrainerError!void {
        if (!std.math.isFinite(learning_rate)) return TrainerError.InvalidLearningRate;
        if (!std.math.isFinite(momentum)) return TrainerError.InvalidMomentum;
        if (learning_rate < 0.0 or learning_rate > 65504.0) return TrainerError.InvalidLearningRate;
        if (momentum < 0.0 or momentum >= 1.0) return TrainerError.InvalidMomentum;
        const lr_f16: f16 = @floatCast(learning_rate);
        const momentum_f16: f16 = @floatCast(momentum);
        if (!std.math.isFinite(lr_f16)) return TrainerError.InvalidHyperparameterAfterCast;
        if (learning_rate > 0.0 and lr_f16 == @as(f16, 0.0)) return TrainerError.InvalidHyperparameterAfterCast;
        const momentum_back: f32 = @floatCast(momentum_f16);
        if (!std.math.isFinite(momentum_back) or !(momentum_back < 1.0)) return TrainerError.InvalidHyperparameterAfterCast;
    }

    fn checkedF32ToF16(value: f32) TrainerError!f16 {
        if (!std.math.isFinite(value)) return TrainerError.InvalidFloat16Value;
        if (value < -65504.0 or value > 65504.0) return TrainerError.InvalidFloat16Value;
        const converted: f16 = @floatCast(value);
        if (!std.math.isFinite(converted)) return TrainerError.InvalidFloat16Value;
        return converted;
    }

    fn safeUsizeToU32(value: usize) TrainerError!u32 {
        if (value > std.math.maxInt(u32)) return TrainerError.ValueOverflow;
        return @as(u32, @intCast(value));
    }

    fn openReadFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        return std.fs.cwd().openFile(path, .{ .mode = .read_only });
    }

    fn createWriteFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .mode = 0o600, .truncate = true });
        return std.fs.cwd().createFile(path, .{ .mode = 0o600, .truncate = true });
    }

    fn deletePath(path: []const u8) void {
        if (std.fs.path.isAbsolute(path)) {
            std.fs.deleteFileAbsolute(path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
            };
            return;
        }
        std.fs.cwd().deleteFile(path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
        };
    }

    fn renamePath(from: []const u8, to: []const u8) !void {
        if (std.fs.path.isAbsolute(from)) return std.fs.renameAbsolute(from, to);
        return std.fs.cwd().rename(from, to);
    }

    fn writeF32(writer: anytype, value: f32) !void {
        try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
    }

    fn readF32(reader: anytype) !f32 {
        const bits = try reader.readInt(u32, .little);
        return @as(f32, @bitCast(bits));
    }

    fn writeF64(writer: anytype, value: f64) !void {
        try writer.writeInt(u64, @as(u64, @bitCast(value)), .little);
    }

    fn readF64(reader: anytype) !f64 {
        const bits = try reader.readInt(u64, .little);
        return @as(f64, @bitCast(bits));
    }

    fn parseOptionalEnvironmentU64(
        self: *DistributedTrainerFuthark,
        name: []const u8,
    ) !?u64 {
        const owned = std.process.getEnvVarOwned(self.allocator, name) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return null,
            else => {
                std.debug.print("[Rank {d}] WARN: reading env '{s}' failed: {}\n", .{ self.coordinator.rank, name, err });
                return TrainerError.InvalidEnvironmentValue;
            },
        };
        defer self.allocator.free(owned);
        if (owned.len == 0) return TrainerError.InvalidEnvironmentValue;
        return std.fmt.parseInt(u64, owned, 10) catch return TrainerError.InvalidEnvironmentValue;
    }

    fn parseOptionalEnvironmentUsize(
        self: *DistributedTrainerFuthark,
        name: []const u8,
    ) !?usize {
        const value_opt = try self.parseOptionalEnvironmentU64(name);
        if (value_opt) |v| {
            return std.math.cast(usize, v) orelse TrainerError.InvalidEnvironmentValue;
        }
        return null;
    }

    fn getMaximumSequenceLength(self: *DistributedTrainerFuthark) !usize {
        const parsed = self.parseOptionalEnvironmentUsize("JAIDE_MAX_SEQ_LEN") catch |err| {
            std.debug.print("[Rank {d}] WARN: JAIDE_MAX_SEQ_LEN invalid: {} (using default {d})\n", .{ self.coordinator.rank, err, self.config.default_max_seq_len });
            return self.config.default_max_seq_len;
        };
        const result = parsed orelse self.config.default_max_seq_len;
        if (result == 0 or result > self.config.max_distributed_integer) return TrainerError.InvalidEnvironmentValue;
        return result;
    }

    fn isTokenizableText(self: *DistributedTrainerFuthark, text: []const u8) !bool {
        var token_list = std.ArrayList(u32).init(self.allocator);
        defer token_list.deinit();
        self.tokenizer.encode(text, &token_list) catch return false;
        return token_list.items.len > 1;
    }

    fn extractDatasetText(self: *DistributedTrainerFuthark, line: []const u8) !?[]u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
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
                        try self.allocator.dupe(u8, text)
                    else
                        null,
                    else => null,
                };
            },
            else => null,
        };
    }

    fn countUsableDatasetSamples(self: *DistributedTrainerFuthark, dataset_path: []const u8) !u64 {
        const file = try openReadFile(dataset_path);
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        var reader = buffered_reader.reader();
        var count: u64 = 0;
        while (try reader.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);
            const maybe_text = try self.extractDatasetText(line);
            if (maybe_text) |text| {
                defer self.allocator.free(text);
                if (self.isTokenizableText(text) catch false) {
                    count = try std.math.add(u64, count, 1);
                }
            }
        }
        return count;
    }

    fn appendDatasetRange(
        self: *DistributedTrainerFuthark,
        dataset_path: []const u8,
        start_valid_index: usize,
        count: usize,
        samples: *std.ArrayList([]const u8),
    ) !void {
        if (count == 0) return;
        const end_valid_index = try std.math.add(usize, start_valid_index, count);

        const file = try openReadFile(dataset_path);
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        var reader = buffered_reader.reader();

        var valid_index: usize = 0;
        var appended: usize = 0;

        while (try reader.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);
            if (appended == count or valid_index >= end_valid_index) break;

            const maybe_text = try self.extractDatasetText(line);
            const text = maybe_text orelse continue;
            var text_owned = true;
            defer if (text_owned) self.allocator.free(text);

            const usable = self.isTokenizableText(text) catch false;
            if (!usable) continue;

            if (valid_index >= start_valid_index) {
                samples.append(text) catch |err| return err;
                text_owned = false;
                appended = try std.math.add(usize, appended, 1);
            }
            valid_index = try std.math.add(usize, valid_index, 1);
        }

        if (appended != count) return TrainerError.InvalidDatasetPartition;
    }

    fn readLayerMatrix(self: *DistributedTrainerFuthark, layer_idx: usize, kind: accel.WeightKind) ![]f16 {
        return self.accelerator.readLayerWeightsFlat(layer_idx, kind, self.allocator) catch |err| {
            std.debug.print("[Rank {d}] readLayerMatrix layer={d} kind={} err={}\n", .{ self.coordinator.rank, layer_idx, kind, err });
            return err;
        };
    }

    fn allReduceFloat32Values(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) return;
        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32(device_values, device_values, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), device_values, byte_count);
        try self.coordinator.synchronize();
    }

    fn allReduceMaximumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (value > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;
        if (self.coordinator.world_size <= 1) return value;
        var arr = [1]f32{@as(f32, @floatFromInt(value))};
        const byte_count = @sizeOf(f32);
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(arr[0..]), byte_count);
        try self.coordinator.allReduceFloat32Max(device_values, device_values, arr.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(arr[0..]), device_values, byte_count);
        try self.coordinator.synchronize();
        if (!std.math.isFinite(arr[0]) or arr[0] < 0.0 or arr[0] > @as(f32, @floatFromInt(self.config.max_distributed_integer))) return TrainerError.InvalidDistributedInteger;
        return @as(u64, @intFromFloat(arr[0]));
    }

    fn allReduceSumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (self.coordinator.world_size <= 1) return value;
        const exact_integer_limit: u64 = 1 << 24;
        const world_size_u64 = std.math.cast(u64, self.coordinator.world_size) orelse return TrainerError.ValueOverflow;
        if (world_size_u64 == 0) return TrainerError.InvalidWorldSize;
        if (world_size_u64 > exact_integer_limit) return TrainerError.DistributedIntegerPrecisionExceeded;

        var radix_bits: u6 = 1;
        while (radix_bits < 24) {
            const candidate_bits: u6 = radix_bits + 1;
            const candidate_mask = (@as(u64, 1) << candidate_bits) - 1;
            if (candidate_mask > exact_integer_limit / world_size_u64) break;
            radix_bits = candidate_bits;
        }

        const bits_per_limb: usize = @intCast(radix_bits);
        const limb_count = (64 + bits_per_limb - 1) / bits_per_limb;
        const limb_mask = (@as(u64, 1) << radix_bits) - 1;
        var limb_values = [_]f32{0.0} ** 64;
        var limb_index: usize = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const shift: u6 = @intCast(limb_index * bits_per_limb);
            const limb = (value >> shift) & limb_mask;
            limb_values[limb_index] = @as(f32, @floatFromInt(limb));
        }

        try self.allReduceFloat32Values(limb_values[0..limb_count]);

        var result: u128 = 0;
        var carry: u128 = 0;
        limb_index = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const reduced = limb_values[limb_index];
            if (!std.math.isFinite(reduced) or reduced < 0.0 or reduced > @as(f32, @floatFromInt(exact_integer_limit))) return TrainerError.InvalidDistributedInteger;
            const rounded = @round(reduced);
            if (rounded != reduced) return TrainerError.InvalidDistributedInteger;
            const limb_sum: u64 = @intFromFloat(rounded);
            const total = @as(u128, limb_sum) + carry;
            const digit = total & @as(u128, limb_mask);
            const shift: u7 = @intCast(limb_index * bits_per_limb);
            result |= digit << shift;
            carry = total >> radix_bits;
        }

        if (carry != 0 or result > @as(u128, std.math.maxInt(u64))) return TrainerError.ValueOverflow;
        return @intCast(result);
    }

    fn reduceWeightedDeltaInPlace(
        self: *DistributedTrainerFuthark,
        delta: []f16,
        local_fraction: f32,
    ) !void {
        if (delta.len == 0) return;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;
        var values = try self.allocator.alloc(f32, delta.len);
        defer self.allocator.free(values);
        for (delta, 0..) |value, index| {
            const converted: f32 = @floatCast(value);
            values[index] = converted * local_fraction;
        }
        try self.allReduceFloat32Values(values);
        for (values, 0..) |value, index| {
            delta[index] = try checkedF32ToF16(value);
        }
    }

    fn applyLayerMatrix(
        self: *DistributedTrainerFuthark,
        layer_idx: usize,
        base: []const f16,
        delta: []const f16,
        kind: accel.WeightKind,
    ) !void {
        if (base.len != delta.len) return TrainerError.InvalidWeightsShape;
        const half = self.model_dim / 2;
        const columns = try std.math.add(usize, half, 1);
        const expected_length = try std.math.mul(usize, half, columns);
        if (base.len != expected_length) return TrainerError.InvalidWeightsShape;

        var merged = try self.allocator.alloc(f16, base.len);
        defer self.allocator.free(merged);
        for (base, delta, 0..) |base_value, delta_value, index| {
            const merged_value = @as(f32, @floatCast(base_value)) + @as(f32, @floatCast(delta_value));
            merged[index] = try checkedF32ToF16(merged_value);
        }
        switch (kind) {
            .weights_s => try self.accelerator.setLayerWeightsS(layer_idx, merged, half, columns),
            .weights_t => try self.accelerator.setLayerWeightsT(layer_idx, merged, half, columns),
            .velocity_s => try self.accelerator.setLayerVelocityS(layer_idx, merged, half, columns),
            .velocity_t => try self.accelerator.setLayerVelocityT(layer_idx, merged, half, columns),
        }
    }

    fn subtractLayerSnapshot(current: []f16, original: []const f16) !void {
        if (current.len != original.len) return TrainerError.InvalidWeightsShape;
        for (current, original) |*current_value, original_value| {
            const difference = @as(f32, @floatCast(current_value.*)) - @as(f32, @floatCast(original_value));
            current_value.* = try checkedF32ToF16(difference);
        }
    }

    fn freeLayerSnapshots(self: *DistributedTrainerFuthark, snapshots: []LayerSnapshot) void {
        for (snapshots) |snapshot| {
            if (snapshot.weights_s.len > 0) self.allocator.free(snapshot.weights_s);
            if (snapshot.weights_t.len > 0) self.allocator.free(snapshot.weights_t);
            if (snapshot.velocity_s.len > 0) self.allocator.free(snapshot.velocity_s);
            if (snapshot.velocity_t.len > 0) self.allocator.free(snapshot.velocity_t);
        }
        self.allocator.free(snapshots);
    }

    fn captureLayerSnapshots(self: *DistributedTrainerFuthark) ![]LayerSnapshot {
        const snapshots = try self.allocator.alloc(LayerSnapshot, self.num_layers);
        var initialized_count: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < initialized_count) : (idx += 1) {
                if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
            }
            self.allocator.free(snapshots);
        }
        for (snapshots) |*snapshot| {
            snapshot.* = .{
                .weights_s = &.{},
                .weights_t = &.{},
                .velocity_s = &.{},
                .velocity_t = &.{},
            };
        }
        for (snapshots, 0..) |*snapshot, layer_index| {
            snapshot.weights_s = try self.readLayerMatrix(layer_index, .weights_s);
            snapshot.weights_t = try self.readLayerMatrix(layer_index, .weights_t);
            snapshot.velocity_s = try self.readLayerMatrix(layer_index, .velocity_s);
            snapshot.velocity_t = try self.readLayerMatrix(layer_index, .velocity_t);
            initialized_count = layer_index + 1;
        }
        return snapshots;
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return TrainerError.InvalidWorldSize;
        if (self.coordinator.rank >= self.coordinator.world_size) return TrainerError.InvalidRank;

        const declared_total_opt = try self.parseOptionalEnvironmentU64("JAIDE_TOTAL_SAMPLES");
        const maximum_samples_opt = try self.parseOptionalEnvironmentU64("JAIDE_MAX_SAMPLES");

        var valid_sample_count: u64 = 0;
        if (declared_total_opt) |declared| {
            valid_sample_count = declared;
        } else {
            if (self.coordinator.isRoot()) {
                valid_sample_count = try self.countUsableDatasetSamples(dataset_path);
            }
            valid_sample_count = try self.allReduceMaximumU64(valid_sample_count);
        }

        if (maximum_samples_opt) |maximum| {
            if (maximum == 0) return TrainerError.InvalidEnvironmentValue;
            if (maximum < valid_sample_count) valid_sample_count = maximum;
        }

        if (valid_sample_count == 0) return TrainerError.EmptyDataset;

        const world_u64: u64 = @as(u64, self.coordinator.world_size);
        const rank_u64: u64 = @as(u64, self.coordinator.rank);
        const base_per_rank = valid_sample_count / world_u64;
        const remainder = valid_sample_count % world_u64;
        const samples_per_rank_u64: u64 = if (rank_u64 < remainder) base_per_rank + 1 else base_per_rank;
        const start_valid_index_u64: u64 = if (rank_u64 < remainder)
            rank_u64 * (base_per_rank + 1)
        else
            remainder * (base_per_rank + 1) + (rank_u64 - remainder) * base_per_rank;

        const samples_per_rank = std.math.cast(usize, samples_per_rank_u64) orelse return TrainerError.ValueOverflow;
        const start_valid_index = std.math.cast(usize, start_valid_index_u64) orelse return TrainerError.ValueOverflow;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| self.allocator.free(sample);
            samples.deinit();
        }

        if (samples_per_rank > 0) {
            try self.appendDatasetRange(dataset_path, start_valid_index, samples_per_rank, &samples);
        }

        if (samples.items.len != samples_per_rank) return TrainerError.InvalidDatasetPartition;

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples from total {d} (rank slice)\n", .{ self.coordinator.rank, samples.items.len, valid_sample_count });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return TrainerError.InvalidBatchSize;

        const local_batch_count: u64 = if (samples.len == 0) 0 else blk: {
            const inc = try std.math.add(usize, samples.len, self.local_batch_size - 1);
            break :blk @as(u64, inc / self.local_batch_size);
        };
        const target_batch_count = try self.allReduceMaximumU64(local_batch_count);

        var total_weighted_loss: f64 = 0.0;
        var total_sample_weight: f64 = 0.0;

        var batch_index: u64 = 0;
        var batch_start: usize = 0;
        while (batch_index < target_batch_count) : (batch_index += 1) {
            var batch: [][]const u8 = &.{};
            if (batch_start < samples.len) {
                const remaining = samples.len - batch_start;
                const batch_length = @min(self.local_batch_size, remaining);
                const batch_end = try std.math.add(usize, batch_start, batch_length);
                batch = samples[batch_start..batch_end];
                batch_start = batch_end;
            }

            const step_result = self.trainStepFuthark(batch) catch |err| {
                std.debug.print("[Rank {d}] trainStepFuthark ERROR at step {d}: {}\n", .{ self.coordinator.rank, self.global_step, err });
                return err;
            };

            if (!std.math.isFinite(step_result.loss)) return TrainerError.InvalidLoss;
            if (step_result.sample_weight > 0.0) {
                total_weighted_loss += @as(f64, step_result.loss) * step_result.sample_weight;
                total_sample_weight += step_result.sample_weight;
            }

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, step_result.loss });
            }

            self.global_step = try std.math.add(u64, self.global_step, 1);
        }

        var reduce_buf = [2]f32{
            @as(f32, @floatCast(total_weighted_loss)),
            @as(f32, @floatCast(total_sample_weight)),
        };
        try self.allReduceFloat32Values(reduce_buf[0..]);

        const global_loss_sum: f64 = @as(f64, reduce_buf[0]);
        const global_weight: f64 = @as(f64, reduce_buf[1]);

        if (global_weight <= 0.0) {
            std.debug.print("[WARNING] No samples processed across all ranks\n", .{});
            try self.pruneRelationalState();
            return 0.0;
        }
        const result: f32 = @floatCast(global_loss_sum / global_weight);
        if (!std.math.isFinite(result)) return TrainerError.InvalidLoss;
        // Prune unbounded relational state accumulated during this epoch so the
        // next epoch starts with fresh, small structures (prevents SIGSEGV from
        // host-RAM exhaustion after many steps).
        try self.pruneRelationalState();
        return result;
    }

    fn ensureTrainingFndsTree(self: *DistributedTrainerFuthark) ![32]u8 {
        if (self.training_fnds_tree_id) |tree_id| return tree_id;
        const tree_id = try self.fnds_manager.createTree(self.config.fnds_max_depth, self.config.fnds_branching);
        self.training_fnds_tree_id = tree_id;
        return tree_id;
    }

    fn ensureTrainingFndsIndex(self: *DistributedTrainerFuthark) ![]const u8 {
        if (self.training_fnds_index_id) |index_id| return index_id;
        const index_id = try self.allocator.dupe(u8, self.config.training_fnds_index_name);
        errdefer self.allocator.free(index_id);
        try self.fnds_manager.createIndex(index_id);
        if (self.fnds_manager.getIndex(index_id) == null) {
            _ = self.fnds_manager.removeIndex(index_id);
            return TrainerError.InvalidGraphIdentifier;
        }
        self.training_fnds_index_id = index_id;
        return index_id;
    }

    fn ensureKnowledgeFndsTree(self: *DistributedTrainerFuthark) ![32]u8 {
        if (self.knowledge_fnds_tree_id) |tree_id| return tree_id;
        const tree_id = try self.fnds_manager.createTree(self.config.fnds_kg_max_depth, self.config.fnds_kg_branching);
        self.knowledge_fnds_tree_id = tree_id;
        return tree_id;
    }

    fn ensureKnowledgeFndsIndex(self: *DistributedTrainerFuthark) ![]const u8 {
        if (self.knowledge_fnds_index_id) |index_id| return index_id;
        const index_id = try self.allocator.dupe(u8, self.config.knowledge_fnds_index_name);
        errdefer self.allocator.free(index_id);
        try self.fnds_manager.createIndex(index_id);
        if (self.fnds_manager.getIndex(index_id) == null) {
            _ = self.fnds_manager.removeIndex(index_id);
            return TrainerError.InvalidGraphIdentifier;
        }
        self.knowledge_fnds_index_id = index_id;
        return index_id;
    }

    fn runCoreRelationalPass(
        self: *DistributedTrainerFuthark,
        token_lists: []const std.ArrayList(u32),
    ) !void {
        var has_tokens = false;
        var maximum_sequence_span: usize = 0;
        for (token_lists) |token_list| {
            if (token_list.items.len == 0) continue;
            has_tokens = true;
            if (token_list.items.len > maximum_sequence_span) maximum_sequence_span = token_list.items.len;
            const tensor_bytes = std.mem.sliceAsBytes(token_list.items);
            _ = self.nsir_graph.encodeInformation(tensor_bytes) catch |err| {
                std.debug.print("[Rank {d}] WARN: nsir_graph.encodeInformation failed: {}\n", .{ self.coordinator.rank, err });
            };
            _ = self.surprise_memory.storeWithSurprise(tensor_bytes, null) catch |err| {
                std.debug.print("[Rank {d}] WARN: surprise_memory.storeWithSurprise failed: {}\n", .{ self.coordinator.rank, err });
            };
        }

        if (!has_tokens) return;

        var graph_embeddings_opt: ?std.ArrayList(core_relational.F64x4) = self.vpu.computeGraphEmbeddings(self.nsir_graph) catch |err| blk: {
            std.debug.print("[Rank {d}] WARN: VPU.computeGraphEmbeddings failed: {}\n", .{ self.coordinator.rank, err });
            break :blk null;
        };
        if (graph_embeddings_opt) |*embeddings| {
            defer embeddings.deinit();
            if (embeddings.items.len > 0) {
                const hash = self.nsir_graph.getTopologyHash() catch self.nsir_graph.topology_hash;
                const theta: f64 = @as(f64, @floatFromInt(hash[0])) / 255.0 * std.math.pi;
                const phi: f64 = @as(f64, @floatFromInt(hash[1])) / 255.0 * std.math.pi;
                self.vpu.quantumVectorOps(embeddings.items, theta, phi);
            }
        }

        const tree_id = try self.ensureTrainingFndsTree();
        const index_id = try self.ensureTrainingFndsIndex();
        for (token_lists, 0..) |token_list, sample_index| {
            if (token_list.items.len == 0) continue;
            var node_id_buffer: [96]u8 = undefined;
            const node_id = try std.fmt.bufPrint(
                &node_id_buffer,
                "rank_{d}_step_{d}_sample_{d}",
                .{ self.coordinator.rank, self.global_step, sample_index },
            );
            const token_bytes = std.mem.sliceAsBytes(token_list.items);
            _ = try self.fnds_manager.insertIntoTree(tree_id, node_id, token_bytes, 0);
            const pattern_length = @min(token_bytes.len, 8 * @sizeOf(u32));
            const pattern_bytes = token_bytes[0..pattern_length];
            var location = try PatternLocation.init(
                self.allocator,
                tree_id,
                0,
                node_id,
                0,
                pattern_bytes.len,
                1.0,
            );
            var transferred = false;
            defer if (!transferred) location.deinit();
            try self.fnds_manager.addPatternToIndex(index_id, pattern_bytes, location);
            transferred = true;
        }

        try self.r_gpu.distributeGraph(self.nsir_graph);

        {
            var orchestrator = ReasoningOrchestrator.init(
                self.allocator,
                self.nsir_graph,
                &self.esso,
                self.crev_kernel,
            );
            defer orchestrator.deinit();
            _ = try orchestrator.runHierarchicalReasoning(self.config.reasoning_cycles);
        }

        {
            const logical_time = self.temporal_logical_time;
            var node_iterator = self.nsir_graph.nodes.iterator();
            while (node_iterator.next()) |entry| {
                const node = entry.value_ptr;
                const quantum_state = QuantumState.init(
                    node.qubit.a.re,
                    node.qubit.a.im,
                    node.qubit.b.re,
                    node.qubit.b.im,
                    node.phase,
                    0.0,
                );
                self.temporal_graph.addNodeAtTime(node.id, quantum_state, logical_time) catch |err| switch (err) {
                    error.NodeAlreadyExists => {},
                    else => return err,
                };
            }
            const sequence_span = std.math.cast(i64, maximum_sequence_span) orelse return TrainerError.TimestampOutOfRange;
            const logical_delta = std.math.mul(i64, sequence_span, self.config.temporal_sequence_tick) catch return TrainerError.TimestampOutOfRange;
            self.temporal_graph.advanceTime(logical_delta);
            self.temporal_logical_time = std.math.add(i64, self.temporal_logical_time, logical_delta) catch return TrainerError.TimestampOutOfRange;
        }

        try self.signal_engine.propagateStep();

        if (!self.training_variable_created) {
            _ = try self.z_runtime.createVariable(self.config.training_variable_name, null);
            self.training_variable_created = true;
        }
    }

    pub const StepResult = struct {
        loss: f32,
        sample_weight: f64,
    };

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !StepResult {
        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| list.deinit();
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            errdefer token_list.deinit();
            self.tokenizer.encode(text, &token_list) catch |err| {
                std.debug.print("[Rank {d}] WARN: tokenizer.encode failed: {} (skipping)\n", .{ self.coordinator.rank, err });
                token_list.deinit();
                continue;
            };
            token_lists.append(token_list) catch |err| {
                token_list.deinit();
                return err;
            };
        }

        var local_active_samples: u64 = 0;
        for (token_lists.items) |list| {
            if (list.items.len >= 2) {
                local_active_samples = try std.math.add(u64, local_active_samples, 1);
            }
        }

        const global_active_samples = try self.allReduceSumU64(local_active_samples);
        if (global_active_samples == 0) {
            return StepResult{ .loss = 0.0, .sample_weight = 0.0 };
        }

        const maximum_sequence_length = try self.getMaximumSequenceLength();

        var local_maximum_prediction_length: u64 = 0;
        for (token_lists.items) |*list| {
            const max_token_count = try std.math.add(usize, maximum_sequence_length, 1);
            if (list.items.len > max_token_count) list.shrinkRetainingCapacity(max_token_count);
            if (list.items.len >= 2) {
                const pred_len: u64 = @intCast(list.items.len - 1);
                if (pred_len > local_maximum_prediction_length) local_maximum_prediction_length = pred_len;
            }
        }

        const max_prediction_length_u64 = try self.allReduceMaximumU64(local_maximum_prediction_length);
        if (max_prediction_length_u64 == 0) return StepResult{ .loss = 0.0, .sample_weight = 0.0 };
        const max_sequence_length: usize = std.math.cast(usize, max_prediction_length_u64) orelse return TrainerError.ValueOverflow;

        var local_token_count: u64 = 0;
        for (token_lists.items) |list| {
            if (list.items.len >= 2) {
                local_token_count = try std.math.add(u64, local_token_count, @intCast(list.items.len - 1));
            }
        }

        const effective_batch_size: usize = blk: {
            var count: usize = 0;
            for (token_lists.items) |list| {
                if (list.items.len >= 2) count += 1;
            }
            if (count == 0) count = 1;
            break :blk count;
        };

        const batch_rows = try std.math.mul(usize, effective_batch_size, max_sequence_length);
        const data_elements = try std.math.mul(usize, batch_rows, self.model_dim);
        const data_size = try std.math.mul(usize, data_elements, @sizeOf(f16));

        var pinned_input = try PinnedMemory.alloc(data_size);
        defer pinned_input.free();
        var pinned_target = try PinnedMemory.alloc(data_size);
        defer pinned_target.free();

        const input_data = pinned_input.asSlice(f16) orelse return TrainerError.AllocationFailed;
        const target_data = pinned_target.asSlice(f16) orelse return TrainerError.AllocationFailed;
        if (input_data.len != data_elements or target_data.len != data_elements) return TrainerError.InvalidPinnedMemorySize;
        @memset(input_data, @as(f16, 0.0));
        @memset(target_data, @as(f16, 0.0));

        var active_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer active_lists.deinit();
        for (token_lists.items) |list| {
            if (list.items.len >= 2) try active_lists.append(list);
        }

        if (self.embedding) |*embedding| {
            embedding.zeroGrad();
            for (active_lists.items, 0..) |token_list, batch_index| {
                const prediction_length = @min(token_list.items.len - 1, max_sequence_length);
                const input_tokens = token_list.items[0..prediction_length];

                var embedding_tensor = try embedding.forward(self.allocator, input_tokens, prediction_length);
                defer embedding_tensor.deinit();
                if (embedding_tensor.shape.dims[0] < prediction_length) return TrainerError.InvalidEmbeddingShape;

                var sequence_index: usize = 0;
                while (sequence_index < prediction_length) : (sequence_index += 1) {
                    const row_offset = try std.math.mul(usize, batch_index, max_sequence_length);
                    const row_index = try std.math.add(usize, row_offset, sequence_index);
                    const base_index = try std.math.mul(usize, row_index, self.model_dim);
                    var column: usize = 0;
                    while (column < self.model_dim) : (column += 1) {
                        const source_index = try std.math.add(
                            usize,
                            try std.math.mul(usize, sequence_index, self.model_dim),
                            column,
                        );
                        const destination_index = try std.math.add(usize, base_index, column);
                        if (source_index >= embedding_tensor.data.len or destination_index >= input_data.len) return TrainerError.IndexOutOfBounds;
                        input_data[destination_index] = try checkedF32ToF16(embedding_tensor.data[source_index]);
                    }
                    const next_token_raw: usize = @intCast(token_list.items[sequence_index + 1]);
                    if (next_token_raw >= embedding.vocab_size) {
                        std.debug.print("[Rank {d}] target token id {d} >= vocab_size {d}\n", .{ self.coordinator.rank, next_token_raw, embedding.vocab_size });
                        return TrainerError.TokenIndexOutOfRange;
                    }
                    var column2: usize = 0;
                    while (column2 < self.model_dim) : (column2 += 1) {
                        const weight_index = try std.math.add(
                            usize,
                            try std.math.mul(usize, next_token_raw, self.model_dim),
                            column2,
                        );
                        const destination_index = try std.math.add(usize, base_index, column2);
                        if (weight_index >= embedding.weight.data.len or destination_index >= target_data.len) return TrainerError.IndexOutOfBounds;
                        target_data[destination_index] = try checkedF32ToF16(embedding.weight.data[weight_index]);
                    }
                }
            }
        } else {
            for (active_lists.items, 0..) |token_list, batch_index| {
                const prediction_length = @min(token_list.items.len - 1, max_sequence_length);
                var sequence_index: usize = 0;
                while (sequence_index < prediction_length) : (sequence_index += 1) {
                    const token_index: usize = @intCast(token_list.items[sequence_index]);
                    const next_token: usize = @intCast(token_list.items[sequence_index + 1]);
                    if (token_index >= self.model_dim or next_token >= self.model_dim) {
                        std.debug.print("[Rank {d}] one-hot mode token id out of range (idx={d} next={d} model_dim={d})\n", .{ self.coordinator.rank, token_index, next_token, self.model_dim });
                        return TrainerError.TokenIndexOutOfRange;
                    }
                    const row_offset = try std.math.mul(usize, batch_index, max_sequence_length);
                    const row_index = try std.math.add(usize, row_offset, sequence_index);
                    const base_index = try std.math.mul(usize, row_index, self.model_dim);
                    const input_index = try std.math.add(usize, base_index, token_index);
                    const target_index = try std.math.add(usize, base_index, next_token);
                    if (input_index >= input_data.len or target_index >= target_data.len) return TrainerError.IndexOutOfBounds;
                    input_data[input_index] = 1.0;
                    target_data[target_index] = 1.0;
                }
            }
            if (self.global_step == 0 and self.coordinator.isRoot() and active_lists.items.len > 0) {
                const first_list = active_lists.items[0].items;
                const dump_n: usize = @min(@as(usize, 12), first_list.len);
                std.debug.print("[Rank 0 step0] one-hot mode, first sample tokens (len={d}): ", .{first_list.len});
                var i: usize = 0;
                while (i < dump_n) : (i += 1) std.debug.print("{d} ", .{first_list[i]});
                std.debug.print("\n[Rank 0 step0] effective_batch_size={d} max_sequence_length={d} model_dim={d}\n", .{ effective_batch_size, max_sequence_length, self.model_dim });
            }
        }

        var inputs = try FutharkArray3DF16.newFromFlat(
            &self.accelerator.ctx,
            input_data,
            effective_batch_size,
            max_sequence_length,
            self.model_dim,
        );
        defer inputs.free(&self.accelerator.ctx);
        var targets = try FutharkArray3DF16.newFromFlat(
            &self.accelerator.ctx,
            target_data,
            effective_batch_size,
            max_sequence_length,
            self.model_dim,
        );
        defer targets.free(&self.accelerator.ctx);

        if (active_lists.items.len > 0 and self.embedding != null) {
            try self.propagateEmbeddingGradients(
                &inputs,
                &targets,
                active_lists.items,
                effective_batch_size,
                max_sequence_length,
            );
        }

        const learning_rate_f16 = try checkedF32ToF16(self.learning_rate);
        const momentum_f16 = try checkedF32ToF16(self.momentum);

        if (self.coordinator.world_size <= 1) {
            const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, learning_rate_f16, momentum_f16);
            try self.accelerator.sync();
            try self.runCoreRelationalPass(active_lists.items);
            const loss: f32 = @floatCast(loss_f16);
            if (!std.math.isFinite(loss)) return TrainerError.InvalidLoss;
            return StepResult{
                .loss = loss,
                .sample_weight = @as(f64, @floatFromInt(local_token_count)),
            };
        }

        const snapshots = try self.captureLayerSnapshots();
        defer self.freeLayerSnapshots(snapshots);

        const local_loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, learning_rate_f16, momentum_f16);
        try self.accelerator.sync();
        const local_loss: f32 = @floatCast(local_loss_f16);
        if (!std.math.isFinite(local_loss)) return TrainerError.InvalidLoss;

        const global_token_count_u64 = try self.allReduceSumU64(local_token_count);
        if (global_token_count_u64 == 0) return StepResult{ .loss = 0.0, .sample_weight = 0.0 };
        const local_fraction: f32 = @floatCast(
            @as(f64, @floatFromInt(local_token_count)) /
                @as(f64, @floatFromInt(global_token_count_u64)),
        );
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;

        for (snapshots, 0..) |snapshot, layer_index| {
            const current_ws = try self.readLayerMatrix(layer_index, .weights_s);
            defer self.allocator.free(current_ws);
            const current_wt = try self.readLayerMatrix(layer_index, .weights_t);
            defer self.allocator.free(current_wt);
            const current_vs = try self.readLayerMatrix(layer_index, .velocity_s);
            defer self.allocator.free(current_vs);
            const current_vt = try self.readLayerMatrix(layer_index, .velocity_t);
            defer self.allocator.free(current_vt);

            try subtractLayerSnapshot(current_ws, snapshot.weights_s);
            try subtractLayerSnapshot(current_wt, snapshot.weights_t);
            try subtractLayerSnapshot(current_vs, snapshot.velocity_s);
            try subtractLayerSnapshot(current_vt, snapshot.velocity_t);

            try self.reduceWeightedDeltaInPlace(current_ws, local_fraction);
            try self.reduceWeightedDeltaInPlace(current_wt, local_fraction);
            try self.reduceWeightedDeltaInPlace(current_vs, local_fraction);
            try self.reduceWeightedDeltaInPlace(current_vt, local_fraction);

            try self.applyLayerMatrix(layer_index, snapshot.weights_s, current_ws, .weights_s);
            try self.applyLayerMatrix(layer_index, snapshot.weights_t, current_wt, .weights_t);
            try self.applyLayerMatrix(layer_index, snapshot.velocity_s, current_vs, .velocity_s);
            try self.applyLayerMatrix(layer_index, snapshot.velocity_t, current_vt, .velocity_t);
        }
        try self.accelerator.sync();

        try self.runCoreRelationalPass(active_lists.items);

        var global_loss = [1]f32{local_loss * local_fraction};
        try self.allReduceFloat32Values(global_loss[0..]);
        if (!std.math.isFinite(global_loss[0])) return TrainerError.InvalidLoss;
        return StepResult{
            .loss = global_loss[0],
            .sample_weight = @as(f64, @floatFromInt(local_token_count)),
        };
    }

    fn makeTemporaryPath(
        self: *DistributedTrainerFuthark,
        path: []const u8,
        suffix: []const u8,
    ) ![]u8 {
        const timestamp = std.time.nanoTimestamp();
        return std.fmt.allocPrint(self.allocator, "{s}.{s}.{d}.{d}.tmp", .{ path, suffix, self.coordinator.rank, timestamp });
    }

    fn readWholeFile(self: *DistributedTrainerFuthark, path: []const u8) ![]u8 {
        const file = try openReadFile(path);
        defer file.close();
        const length_u64 = try file.getEndPos();
        const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
        const data = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(data);
        try file.reader().readNoEof(data);
        return data;
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        try self.coordinator.synchronize();
        if (!self.coordinator.isRoot()) return;
        try self.accelerator.sync();

        const tokenizer_tmp = try self.makeTemporaryPath(path, "tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);

        try self.tokenizer.saveVocab(tokenizer_tmp);

        const tokenizer_data = try self.readWholeFile(tokenizer_tmp);
        defer self.allocator.free(tokenizer_data);

        const checkpoint_tmp = try self.makeTemporaryPath(path, "checkpoint");
        defer self.allocator.free(checkpoint_tmp);
        var checkpoint_committed = false;
        defer if (!checkpoint_committed) deletePath(checkpoint_tmp);

        {
            const file = try createWriteFile(checkpoint_tmp);
            var file_closed = false;
            defer if (!file_closed) file.close();

            var buffered_writer = std.io.bufferedWriter(file.writer());
            const writer = buffered_writer.writer();

            try writer.writeAll(CHECKPOINT_MAGIC[0..]);
            try writer.writeInt(u32, self.config.checkpoint_version, .little);
            try writer.writeInt(u64, self.global_step, .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.num_layers)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.local_batch_size)), .little);
            try writeF32(writer, self.learning_rate);
            try writeF32(writer, self.momentum);

            var li_save: usize = 0;
            while (li_save < self.num_layers) : (li_save += 1) {
                const ws = try self.readLayerMatrix(li_save, .weights_s);
                defer self.allocator.free(ws);
                try writer.writeInt(u64, @as(u64, ws.len), .little);
                for (ws) |w| try writeF32(writer, @floatCast(w));
                const wt = try self.readLayerMatrix(li_save, .weights_t);
                defer self.allocator.free(wt);
                try writer.writeInt(u64, @as(u64, wt.len), .little);
                for (wt) |w| try writeF32(writer, @floatCast(w));
                const vs = try self.readLayerMatrix(li_save, .velocity_s);
                defer self.allocator.free(vs);
                try writer.writeInt(u64, @as(u64, vs.len), .little);
                for (vs) |v| try writeF32(writer, @floatCast(v));
                const vt = try self.readLayerMatrix(li_save, .velocity_t);
                defer self.allocator.free(vt);
                try writer.writeInt(u64, @as(u64, vt.len), .little);
                for (vt) |v| try writeF32(writer, @floatCast(v));
            }

            try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_min)));
            try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_max)));

            if (self.embedding) |*embedding| {
                try writer.writeByte(1);
                try writer.writeInt(u64, @as(u64, embedding.vocab_size), .little);
                try writer.writeInt(u64, @as(u64, embedding.dim), .little);
                try writer.writeInt(u64, @as(u64, embedding.weight.data.len), .little);
                for (embedding.weight.data) |w| {
                    if (!std.math.isFinite(w)) return TrainerError.InvalidEmbeddingWeight;
                    try writeF32(writer, w);
                }
                try writer.writeInt(u64, @as(u64, embedding.velocity.data.len), .little);
                for (embedding.velocity.data) |v| {
                    if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
                    try writeF32(writer, v);
                }
            } else {
                try writer.writeByte(0);
            }

            const node_count = try safeUsizeToU32(self.nsir_graph.nodes.count());
            try writer.writeInt(u32, node_count, .little);
            var node_iter = self.nsir_graph.nodes.iterator();
            while (node_iter.next()) |entry| {
                const node = entry.value_ptr.*;
                const id_len = try safeUsizeToU32(node.id.len);
                try writer.writeInt(u32, id_len, .little);
                try writer.writeAll(node.id);
                const data_len = try safeUsizeToU32(node.data.len);
                try writer.writeInt(u32, data_len, .little);
                try writer.writeAll(node.data);
                try writeF64(writer, node.qubit.a.re);
                try writeF64(writer, node.qubit.a.im);
                try writeF64(writer, node.qubit.b.re);
                try writeF64(writer, node.qubit.b.im);
                try writeF64(writer, node.phase);
            }

            const edge_key_count = try safeUsizeToU32(self.nsir_graph.edges.count());
            try writer.writeInt(u32, edge_key_count, .little);
            var edge_iter = self.nsir_graph.edges.iterator();
            while (edge_iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const edge_list = entry.value_ptr.*;
                const src_len = try safeUsizeToU32(key.source.len);
                try writer.writeInt(u32, src_len, .little);
                try writer.writeAll(key.source);
                const tgt_len = try safeUsizeToU32(key.target.len);
                try writer.writeInt(u32, tgt_len, .little);
                try writer.writeAll(key.target);
                const count = try safeUsizeToU32(edge_list.items.len);
                try writer.writeInt(u32, count, .little);
                for (edge_list.items) |edge| {
                    if (!std.math.isFinite(edge.weight)) return TrainerError.InvalidEdgeWeight;
                    try writeF64(writer, edge.weight);
                    try writer.writeByte(@intFromEnum(edge.quality));
                    try writeF64(writer, edge.quantum_correlation.re);
                    try writeF64(writer, edge.quantum_correlation.im);
                    try writeF64(writer, edge.fractal_dimension);
                }
            }

            try writer.writeInt(u64, @as(u64, tokenizer_data.len), .little);
            try writer.writeAll(tokenizer_data);
            try writer.writeInt(u32, CHECKPOINT_TRAILER, .little);

            try buffered_writer.flush();
            try file.sync();
            file.close();
            file_closed = true;
        }

        try renamePath(checkpoint_tmp, path);
        checkpoint_committed = true;
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);
        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    fn readCheckpointF16Array(
        self: *DistributedTrainerFuthark,
        reader: anytype,
        expected_length: usize,
    ) ![]f16 {
        const saved_length_u64 = try reader.readInt(u64, .little);
        const saved_length = std.math.cast(usize, saved_length_u64) orelse return TrainerError.InvalidWeightsShape;
        if (saved_length != expected_length) return TrainerError.InvalidWeightsShape;
        const values = try self.allocator.alloc(f16, saved_length);
        errdefer self.allocator.free(values);
        var i: usize = 0;
        while (i < saved_length) : (i += 1) {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return TrainerError.InvalidWeightValue;
            values[i] = try checkedF32ToF16(v);
        }
        return values;
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const file = try openReadFile(path);
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        const reader = buffered_reader.reader();

        var magic_buf: [8]u8 = undefined;
        try reader.readNoEof(magic_buf[0..]);
        if (!std.mem.eql(u8, magic_buf[0..], CHECKPOINT_MAGIC[0..])) return TrainerError.CheckpointMagicMismatch;

        const version = try reader.readInt(u32, .little);
        if (version != self.config.checkpoint_version) return TrainerError.CheckpointVersionMismatch;

        const saved_global_step = try reader.readInt(u64, .little);
        const saved_model_dim_u64 = try reader.readInt(u64, .little);
        const saved_num_layers_u64 = try reader.readInt(u64, .little);
        const saved_vocab_size_u64 = try reader.readInt(u64, .little);
        const saved_local_batch_size_u64 = try reader.readInt(u64, .little);
        const saved_learning_rate = try readF32(reader);
        const saved_momentum = try readF32(reader);

        const saved_model_dim = std.math.cast(usize, saved_model_dim_u64) orelse return TrainerError.ModelDimMismatch;
        const saved_num_layers = std.math.cast(usize, saved_num_layers_u64) orelse return TrainerError.NumLayersMismatch;
        const saved_vocab_size = std.math.cast(usize, saved_vocab_size_u64) orelse return TrainerError.VocabSizeMismatch;
        const saved_local_batch_size = std.math.cast(usize, saved_local_batch_size_u64) orelse return TrainerError.InvalidBatchSize;
        const saved_step_i64 = std.math.cast(i64, saved_global_step) orelse return TrainerError.TimestampOutOfRange;
        const loaded_temporal_logical_time = std.math.mul(i64, saved_step_i64, self.config.temporal_sequence_tick) catch return TrainerError.TimestampOutOfRange;

        if (saved_model_dim != self.model_dim) return TrainerError.ModelDimMismatch;
        if (saved_num_layers != self.num_layers) return TrainerError.NumLayersMismatch;
        if (saved_local_batch_size == 0) return TrainerError.InvalidBatchSize;
        try validateHyperparameters(saved_learning_rate, saved_momentum);

        const half = self.model_dim / 2;
        const columns = try std.math.add(usize, half, 1);
        const expected_length = try std.math.mul(usize, half, columns);

        const snapshots = try self.allocator.alloc(LayerSnapshot, self.num_layers);
        var snapshots_initialized: usize = 0;
        var snapshots_committed = false;
        errdefer if (!snapshots_committed) {
            var idx: usize = 0;
            while (idx < snapshots_initialized) : (idx += 1) {
                if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
            }
            self.allocator.free(snapshots);
        };
        for (snapshots) |*snapshot| {
            snapshot.* = .{
                .weights_s = &.{},
                .weights_t = &.{},
                .velocity_s = &.{},
                .velocity_t = &.{},
            };
        }
        for (snapshots, 0..) |*snapshot, layer_index| {
            snapshot.weights_s = try self.readCheckpointF16Array(reader, expected_length);
            snapshot.weights_t = try self.readCheckpointF16Array(reader, expected_length);
            snapshot.velocity_s = try self.readCheckpointF16Array(reader, expected_length);
            snapshot.velocity_t = try self.readCheckpointF16Array(reader, expected_length);
            snapshots_initialized = layer_index + 1;
        }

        const clip_min_f32 = try readF32(reader);
        const clip_max_f32 = try readF32(reader);
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) return TrainerError.InvalidClipRange;
        const clip_min = try checkedF32ToF16(clip_min_f32);
        const clip_max = try checkedF32ToF16(clip_max_f32);
        if (!(@as(f32, @floatCast(clip_min)) < @as(f32, @floatCast(clip_max)))) return TrainerError.ConvertPrecisionLoss;

        const has_embedding = try reader.readByte();
        if (has_embedding > 1) return TrainerError.InvalidCheckpointEmbeddingFlag;

        var loaded_embedding: ?LearnedEmbedding = null;
        var loaded_embedding_committed = false;
        errdefer if (!loaded_embedding_committed) {
            if (loaded_embedding) |*emb| emb.deinit();
        };
        if (has_embedding == 1) {
            const embedding_vocab_u64 = try reader.readInt(u64, .little);
            const embedding_dim_u64 = try reader.readInt(u64, .little);
            const embedding_vocab = std.math.cast(usize, embedding_vocab_u64) orelse return TrainerError.VocabSizeMismatch;
            const embedding_dim = std.math.cast(usize, embedding_dim_u64) orelse return TrainerError.ModelDimMismatch;
            if (embedding_vocab != saved_vocab_size) return TrainerError.VocabSizeMismatch;
            if (embedding_dim != self.model_dim) return TrainerError.ModelDimMismatch;

            var new_embedding = try LearnedEmbedding.init(self.allocator, embedding_vocab, embedding_dim, self.config.embedding_seed);
            errdefer new_embedding.deinit();

            const w_len_u64 = try reader.readInt(u64, .little);
            const w_len = std.math.cast(usize, w_len_u64) orelse return TrainerError.InvalidEmbeddingShape;
            if (w_len != new_embedding.weight.data.len) return TrainerError.InvalidEmbeddingShape;
            for (new_embedding.weight.data) |*value| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
                value.* = v;
            }
            const vel_len_u64 = try reader.readInt(u64, .little);
            const vel_len = std.math.cast(usize, vel_len_u64) orelse return TrainerError.InvalidEmbeddingShape;
            if (vel_len != new_embedding.velocity.data.len) return TrainerError.InvalidEmbeddingShape;
            for (new_embedding.velocity.data) |*value| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
                value.* = v;
            }
            loaded_embedding = new_embedding;
        }

        var new_graph_ptr = try self.allocator.create(SelfSimilarRelationalGraph);
        var new_graph_ptr_committed = false;
        errdefer if (!new_graph_ptr_committed) self.allocator.destroy(new_graph_ptr);
        new_graph_ptr.* = try SelfSimilarRelationalGraph.init(self.allocator);
        var new_graph_committed = false;
        errdefer if (!new_graph_committed) new_graph_ptr.deinit();

        const node_count = try reader.readInt(u32, .little);
        var ni: u32 = 0;
        while (ni < node_count) : (ni += 1) {
            const id_len = try reader.readInt(u32, .little);
            if (id_len > self.config.max_id_length) return TrainerError.NodeIdTooLong;
            const id = try self.allocator.alloc(u8, id_len);
            defer self.allocator.free(id);
            try reader.readNoEof(id);

            const data_len = try reader.readInt(u32, .little);
            if (data_len > self.config.max_node_data_length) return TrainerError.NodeDataTooLong;
            const data_bytes = try self.allocator.alloc(u8, data_len);
            defer self.allocator.free(data_bytes);
            try reader.readNoEof(data_bytes);

            const a_re = try readF64(reader);
            const a_im = try readF64(reader);
            const b_re = try readF64(reader);
            const b_im = try readF64(reader);
            const phase = try readF64(reader);
            if (!std.math.isFinite(a_re) or !std.math.isFinite(a_im) or !std.math.isFinite(b_re) or !std.math.isFinite(b_im) or !std.math.isFinite(phase)) return TrainerError.InvalidQuantumState;

            const qubit = nsir.Qubit.init(
                std.math.Complex(f64).init(a_re, a_im),
                std.math.Complex(f64).init(b_re, b_im),
            );
            const node = try nsir.Node.init(new_graph_ptr.allocator, id, data_bytes, qubit, phase);
            try new_graph_ptr.addNode(node);
        }

        const edge_key_count = try reader.readInt(u32, .little);
        if (edge_key_count > self.config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
        var ei: u32 = 0;
        while (ei < edge_key_count) : (ei += 1) {
            const src_len = try reader.readInt(u32, .little);
            if (src_len > self.config.max_id_length) return TrainerError.NodeIdTooLong;
            const source = try self.allocator.alloc(u8, src_len);
            defer self.allocator.free(source);
            try reader.readNoEof(source);

            const tgt_len = try reader.readInt(u32, .little);
            if (tgt_len > self.config.max_id_length) return TrainerError.NodeIdTooLong;
            const target = try self.allocator.alloc(u8, tgt_len);
            defer self.allocator.free(target);
            try reader.readNoEof(target);

            const count = try reader.readInt(u32, .little);
            if (count > self.config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
            var k: u32 = 0;
            while (k < count) : (k += 1) {
                const weight = try readF64(reader);
                const quality_byte = try reader.readByte();
                if (quality_byte > @intFromEnum(nsir.EdgeQuality.fractal)) return TrainerError.InvalidQualityByte;
                const quality: nsir.EdgeQuality = @enumFromInt(quality_byte);
                const qc_re = try readF64(reader);
                const qc_im = try readF64(reader);
                const fd = try readF64(reader);
                if (!std.math.isFinite(weight) or !std.math.isFinite(qc_re) or !std.math.isFinite(qc_im) or !std.math.isFinite(fd)) return TrainerError.InvalidEdgeWeight;
                const edge = try nsir.Edge.init(
                    new_graph_ptr.allocator,
                    source,
                    target,
                    quality,
                    weight,
                    std.math.Complex(f64).init(qc_re, qc_im),
                    fd,
                );
                try new_graph_ptr.addEdge(source, target, edge);
            }
        }

        const tokenizer_length_u64 = try reader.readInt(u64, .little);
        const tokenizer_length = std.math.cast(usize, tokenizer_length_u64) orelse return TrainerError.InvalidTokenizerData;
        if (tokenizer_length == 0) return TrainerError.InvalidTokenizerData;
        const tokenizer_data = try self.allocator.alloc(u8, tokenizer_length);
        defer self.allocator.free(tokenizer_data);
        try reader.readNoEof(tokenizer_data);

        const trailer = try reader.readInt(u32, .little);
        if (trailer != CHECKPOINT_TRAILER) return TrainerError.CheckpointCorrupted;

        var trailing_byte: [1]u8 = undefined;
        const trailing_read = try reader.read(trailing_byte[0..]);
        if (trailing_read != 0) return TrainerError.TrailingCheckpointData;

        const tokenizer_tmp = try self.makeTemporaryPath(path, "load-tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);

        {
            const tokenizer_file = try createWriteFile(tokenizer_tmp);
            var closed = false;
            defer if (!closed) tokenizer_file.close();
            try tokenizer_file.writer().writeAll(tokenizer_data);
            try tokenizer_file.sync();
            tokenizer_file.close();
            closed = true;
        }

        var new_tokenizer = try createConfiguredTokenizer(self.allocator, self.config);
        var new_tokenizer_committed = false;
        errdefer if (!new_tokenizer_committed) new_tokenizer.deinit();
        try new_tokenizer.loadVocab(tokenizer_tmp);
        if (new_tokenizer.next_token_id != saved_vocab_size) return TrainerError.VocabSizeMismatch;
        if (loaded_embedding) |emb_val| {
            if (emb_val.vocab_size != new_tokenizer.next_token_id) return TrainerError.VocabSizeMismatch;
        }

        var new_accelerator_ptr = try self.allocator.create(RSFAccelerator);
        var new_accelerator_ptr_committed = false;
        errdefer if (!new_accelerator_ptr_committed) self.allocator.destroy(new_accelerator_ptr);
        new_accelerator_ptr.* = try RSFAccelerator.initMultiLayer(self.model_dim, self.num_layers, self.allocator);
        var new_accelerator_committed = false;
        errdefer if (!new_accelerator_committed) new_accelerator_ptr.deinit();

        for (snapshots, 0..) |snapshot, layer_index| {
            try new_accelerator_ptr.setLayerWeightsS(layer_index, snapshot.weights_s, half, columns);
            try new_accelerator_ptr.setLayerWeightsT(layer_index, snapshot.weights_t, half, columns);
            try new_accelerator_ptr.setLayerVelocityS(layer_index, snapshot.velocity_s, half, columns);
            try new_accelerator_ptr.setLayerVelocityT(layer_index, snapshot.velocity_t, half, columns);
        }
        try new_accelerator_ptr.setClipRange(clip_min, clip_max);
        try new_accelerator_ptr.sync();

        var new_embedding_accel: ?EmbeddingAccelerator = null;
        var new_embedding_accel_committed = false;
        errdefer if (!new_embedding_accel_committed) {
            if (new_embedding_accel) |*ea| ea.deinit();
        };
        if (self.embedding_accel != null or loaded_embedding != null) {
            const vocab_for_accel: usize = if (loaded_embedding) |emb_val| emb_val.vocab_size else new_tokenizer.next_token_id;
            new_embedding_accel = EmbeddingAccelerator.init(
                &new_accelerator_ptr.ctx,
                vocab_for_accel,
                self.model_dim,
                self.config.embedding_seed,
            ) catch |err| blk: {
                std.debug.print("[Rank {d}] WARN: EmbeddingAccelerator reinit after load failed: {}\n", .{ self.coordinator.rank, err });
                break :blk null;
            };
        }

        var new_signal_engine_ptr = try self.allocator.create(SignalPropagationEngine);
        var new_signal_engine_ptr_committed = false;
        errdefer if (!new_signal_engine_ptr_committed) self.allocator.destroy(new_signal_engine_ptr);
        new_signal_engine_ptr.* = SignalPropagationEngine.init(
            self.allocator,
            new_graph_ptr,
            &self.crev_kernel.flow_analyzer,
        );
        var new_signal_engine_committed = false;
        errdefer if (!new_signal_engine_committed) new_signal_engine_ptr.deinit();

        self.signal_engine.deinit();
        self.allocator.destroy(self.signal_engine);
        self.signal_engine = new_signal_engine_ptr;
        new_signal_engine_ptr_committed = true;
        new_signal_engine_committed = true;

        if (self.embedding_accel) |*old_accel| old_accel.deinit();
        self.embedding_accel = new_embedding_accel;
        new_embedding_accel_committed = true;

        if (self.embedding) |*old_emb| old_emb.deinit();
        self.embedding = loaded_embedding;
        loaded_embedding_committed = true;

        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.accelerator = new_accelerator_ptr;
        new_accelerator_ptr_committed = true;
        new_accelerator_committed = true;

        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.nsir_graph = new_graph_ptr;
        new_graph_ptr_committed = true;
        new_graph_committed = true;

        self.tokenizer.deinit();
        self.tokenizer = new_tokenizer;
        new_tokenizer_committed = true;

        self.resetSpectralState();
        self.releaseFndsResources();
        self.temporal_graph.deinit();
        self.temporal_graph = TemporalGraph.init(self.allocator);
        self.temporal_logical_time = loaded_temporal_logical_time;
        self.vocab_size = saved_vocab_size;
        self.local_batch_size = saved_local_batch_size;
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;
        self.global_step = saved_global_step;
        self.training_variable_created = false;

        self.freeLayerSnapshots(snapshots);
        snapshots_committed = true;
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }

    fn freeFuthark3D(fctx: ?*futhark.struct_futhark_context, arr: ?*futhark.struct_futhark_f16_3d) void {
        if (arr) |a| _ = futhark.futhark_free_f16_3d(fctx, a);
    }

    fn forwardThroughLayers(
        self: *DistributedTrainerFuthark,
        futhark_context: ?*futhark.struct_futhark_context,
        input: ?*futhark.struct_futhark_f16_3d,
        layer_count: usize,
        clip_min: u16,
        clip_max: u16,
    ) !*futhark.struct_futhark_f16_3d {
        if (input == null or layer_count == 0 or layer_count > self.num_layers) return TrainerError.FutharkForwardFailed;
        var current = input;
        var current_owned = false;
        errdefer if (current_owned) freeFuthark3D(futhark_context, current);

        var layer_index: usize = 0;
        while (layer_index < layer_count) : (layer_index += 1) {
            var forward_output: ?*futhark.struct_futhark_f16_3d = null;
            const forward_result = futhark.futhark_entry_batch_forward(
                futhark_context,
                &forward_output,
                current,
                self.accelerator.layers[layer_index].weights_s.arr,
                self.accelerator.layers[layer_index].weights_t.arr,
                clip_min,
                clip_max,
            );
            if (forward_result != 0 or forward_output == null) {
                if (forward_output) |output| freeFuthark3D(futhark_context, output);
                return TrainerError.FutharkForwardFailed;
            }

            var transformed_output: ?*futhark.struct_futhark_f16_3d = null;
            const transform_result = futhark.futhark_entry_batch_oftb_forward(
                futhark_context,
                &transformed_output,
                forward_output,
            );
            freeFuthark3D(futhark_context, forward_output);
            if (transform_result != 0 or transformed_output == null) {
                if (transformed_output) |output| freeFuthark3D(futhark_context, output);
                return TrainerError.FutharkTransformFailed;
            }

            if (current_owned) freeFuthark3D(futhark_context, current);
            current = transformed_output;
            current_owned = true;
        }

        return current.?;
    }

    fn propagateEmbeddingGradients(
        self: *DistributedTrainerFuthark,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        token_items: []const std.ArrayList(u32),
        effective_batch_size: usize,
        max_sequence_length: usize,
    ) !void {
        if (self.embedding == null or token_items.len == 0) return;
        const embedding = &self.embedding.?;
        const futhark_context = self.accelerator.ctx.ctx orelse return TrainerError.FutharkContextUnavailable;
        const clip_min: u16 = @bitCast(self.accelerator.clip_min);
        const clip_max: u16 = @bitCast(self.accelerator.clip_max);

        const final_activation = try self.forwardThroughLayers(
            futhark_context,
            inputs.arr,
            self.num_layers,
            clip_min,
            clip_max,
        );
        defer freeFuthark3D(futhark_context, final_activation);
        try self.accelerator.sync();

        var gradient: ?*futhark.struct_futhark_f16_3d = null;
        defer if (gradient) |value| freeFuthark3D(futhark_context, value);

        const initial_gradient_result = futhark.futhark_entry_compute_initial_grad_l2(
            futhark_context,
            &gradient,
            final_activation,
            targets.arr,
        );
        if (initial_gradient_result != 0 or gradient == null) {
            if (gradient) |value| freeFuthark3D(futhark_context, value);
            gradient = null;
            return TrainerError.FutharkGradientFailed;
        }

        var layer_backward = self.num_layers;
        while (layer_backward > 0) : (layer_backward -= 1) {
            var transformed_gradient: ?*futhark.struct_futhark_f16_3d = null;
            const backward_transform_result = futhark.futhark_entry_batch_oftb_backward(
                futhark_context,
                &transformed_gradient,
                gradient,
            );
            if (backward_transform_result != 0 or transformed_gradient == null) {
                if (transformed_gradient) |value| freeFuthark3D(futhark_context, value);
                return TrainerError.FutharkBackwardTransformFailed;
            }
            freeFuthark3D(futhark_context, gradient);
            gradient = transformed_gradient;

            const layer_index = layer_backward - 1;
            var recomputed_activation: ?*futhark.struct_futhark_f16_3d = null;
            defer if (recomputed_activation) |value| freeFuthark3D(futhark_context, value);
            const layer_input: ?*futhark.struct_futhark_f16_3d = if (layer_index == 0)
                inputs.arr
            else blk: {
                recomputed_activation = try self.forwardThroughLayers(
                    futhark_context,
                    inputs.arr,
                    layer_index,
                    clip_min,
                    clip_max,
                );
                break :blk recomputed_activation;
            };

            var gradient_tuple: ?*futhark.struct_futhark_opaque_tup3_grad_full = null;
            const full_gradient_result = futhark.futhark_entry_batch_gradients_full(
                futhark_context,
                &gradient_tuple,
                layer_input,
                gradient,
                self.accelerator.layers[layer_index].weights_s.arr,
                self.accelerator.layers[layer_index].weights_t.arr,
                clip_min,
                clip_max,
            );
            if (full_gradient_result != 0 or gradient_tuple == null) {
                if (gradient_tuple) |value| _ = futhark.futhark_free_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16(futhark_context, value);
                return TrainerError.FutharkFullGradientFailed;
            }
            freeFuthark3D(futhark_context, gradient);
            gradient = null;

            var gradient_input: ?*futhark.struct_futhark_f16_3d = null;
            const projection_result = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_2(
                futhark_context,
                &gradient_input,
                gradient_tuple,
            );
            _ = futhark.futhark_free_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16(futhark_context, gradient_tuple);
            if (projection_result != 0 or gradient_input == null) {
                if (gradient_input) |value| freeFuthark3D(futhark_context, value);
                return TrainerError.FutharkProjectionFailed;
            }
            gradient = gradient_input;
        }
        try self.accelerator.sync();

        const gradient_pointer = gradient orelse return TrainerError.FutharkGradientFailed;
        const total_elements = try std.math.mul(
            usize,
            try std.math.mul(usize, effective_batch_size, max_sequence_length),
            self.model_dim,
        );
        const host_gradient = try self.allocator.alloc(f16, total_elements);
        defer self.allocator.free(host_gradient);

        const copy_result = futhark.futhark_values_f16_3d(futhark_context, gradient_pointer, @ptrCast(host_gradient.ptr));
        if (copy_result != 0) return TrainerError.FutharkGradientCopyFailed;
        try self.accelerator.sync();

        var flat_tokens = std.ArrayList(u32).init(self.allocator);
        defer flat_tokens.deinit();

        const per_sample_gradient_length = try std.math.mul(usize, max_sequence_length, self.model_dim);
        const total_gradient_length = try std.math.mul(usize, token_items.len, per_sample_gradient_length);
        const batch_gradient = try self.allocator.alloc(f32, total_gradient_length);
        defer self.allocator.free(batch_gradient);
        @memset(batch_gradient, 0.0);

        for (token_items, 0..) |token_list, batch_index| {
            if (token_list.items.len < 2) continue;
            const sequence_length = @min(token_list.items.len - 1, max_sequence_length);
            var sequence_index: usize = 0;
            while (sequence_index < sequence_length) : (sequence_index += 1) {
                var column: usize = 0;
                while (column < self.model_dim) : (column += 1) {
                    const source_index = try std.math.add(
                        usize,
                        try std.math.mul(
                            usize,
                            try std.math.mul(usize, batch_index, max_sequence_length),
                            self.model_dim,
                        ),
                        try std.math.add(usize, try std.math.mul(usize, sequence_index, self.model_dim), column),
                    );
                    const destination_index = try std.math.add(
                        usize,
                        try std.math.mul(usize, batch_index, per_sample_gradient_length),
                        try std.math.add(usize, try std.math.mul(usize, sequence_index, self.model_dim), column),
                    );
                    if (source_index >= host_gradient.len or destination_index >= batch_gradient.len) return TrainerError.IndexOutOfBounds;
                    batch_gradient[destination_index] = @floatCast(host_gradient[source_index]);
                }
                try flat_tokens.append(token_list.items[sequence_index]);
            }
            var padding_index = sequence_length;
            while (padding_index < max_sequence_length) : (padding_index += 1) try flat_tokens.append(0);
        }

        if (self.config.use_normalized_gradient_flow) {
            var maximum_absolute_value: f32 = 0.0;
            for (batch_gradient) |value| {
                if (!std.math.isFinite(value)) return TrainerError.InvalidGradient;
                const absolute_value = @abs(value);
                if (absolute_value > maximum_absolute_value) maximum_absolute_value = absolute_value;
            }
            if (maximum_absolute_value > 0.0) {
                var scaled_norm_squared: f64 = 0.0;
                for (batch_gradient) |value| {
                    const scaled = @as(f64, @floatCast(value)) / @as(f64, @floatCast(maximum_absolute_value));
                    scaled_norm_squared += scaled * scaled;
                }
                const norm = @as(f64, @floatCast(maximum_absolute_value)) * @sqrt(scaled_norm_squared);
                if (!std.math.isFinite(norm)) return TrainerError.InvalidGradient;
                const clip_norm: f64 = @as(f64, @floatCast(self.config.gradient_clip_norm));
                if (norm > clip_norm and norm > 1e-12) {
                    const scale: f32 = @floatCast(clip_norm / norm);
                    for (batch_gradient) |*value| value.* *= scale;
                }
            }
        }

        if (flat_tokens.items.len > 0) {
            embedding.backward(flat_tokens.items, batch_gradient, max_sequence_length);
        }

        if (self.coordinator.world_size > 1) {
            const gradient_bytes = try std.math.mul(usize, embedding.grad.data.len, @sizeOf(f32));
            if (gradient_bytes > 0) {
                const device_gradient = try self.coordinator.allocDeviceMemory(gradient_bytes);
                defer self.coordinator.freeDeviceMemory(device_gradient);
                try self.coordinator.copyHostToDevice(device_gradient, std.mem.sliceAsBytes(embedding.grad.data), gradient_bytes);
                try self.coordinator.allReduceFloat32Avg(device_gradient, device_gradient, embedding.grad.data.len);
                try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(embedding.grad.data), device_gradient, gradient_bytes);
                try self.coordinator.synchronize();
            }
        }

        embedding.applyGradients(self.learning_rate, self.momentum);
        if (self.config.spectral_iterations > 0 and self.global_step % 10 == 0) {
            try self.applyEmbeddingSpectralNormalization();
        }
    }

    fn ensureSpectralState(
        self: *DistributedTrainerFuthark,
        rows: usize,
        columns: usize,
    ) !void {
        if (self.spectral_u) |u| {
            if (self.spectral_v) |v| {
                if (u.len == rows and v.len == columns) return;
            }
        }

        self.resetSpectralState();
        const u = try self.allocator.alloc(f32, rows);
        errdefer self.allocator.free(u);
        const v = try self.allocator.alloc(f32, columns);
        errdefer self.allocator.free(v);

        var norm_squared: f64 = 0.0;
        for (u) |*value| {
            value.* = std.crypto.random.float(f32) - 0.5;
            norm_squared += @as(f64, value.*) * @as(f64, value.*);
        }
        const norm = @sqrt(norm_squared);
        if (!std.math.isFinite(norm) or norm <= 1e-12) return TrainerError.InvalidSpectralState;
        const inverse_norm: f32 = @floatCast(1.0 / norm);
        for (u) |*value| value.* *= inverse_norm;
        @memset(v, 0.0);

        self.spectral_u = u;
        self.spectral_v = v;
    }

    fn applyEmbeddingSpectralNormalization(self: *DistributedTrainerFuthark) !void {
        if (self.embedding == null) return;
        const embedding = &self.embedding.?;
        const rows = embedding.vocab_size;
        const columns = embedding.dim;
        if (rows == 0 or columns == 0 or self.spectral_normalizer.power_iterations == 0) return;

        try self.ensureSpectralState(rows, columns);
        const u = self.spectral_u.?;
        const v = self.spectral_v.?;

        var iteration: usize = 0;
        while (iteration < self.spectral_normalizer.power_iterations) : (iteration += 1) {
            @memset(v, 0.0);
            var row: usize = 0;
            while (row < rows) : (row += 1) {
                var column: usize = 0;
                while (column < columns) : (column += 1) {
                    v[column] += embedding.weight.data[row * columns + column] * u[row];
                }
            }
            var v_norm_squared: f64 = 0.0;
            for (v) |value| v_norm_squared += @as(f64, value) * @as(f64, value);
            const v_norm = @sqrt(v_norm_squared);
            if (!std.math.isFinite(v_norm) or v_norm <= 1e-12) return TrainerError.InvalidSpectralState;
            const inverse_v_norm: f32 = @floatCast(1.0 / v_norm);
            for (v) |*value| value.* *= inverse_v_norm;

            @memset(u, 0.0);
            row = 0;
            while (row < rows) : (row += 1) {
                var column: usize = 0;
                while (column < columns) : (column += 1) {
                    u[row] += embedding.weight.data[row * columns + column] * v[column];
                }
            }
            var u_norm_squared: f64 = 0.0;
            for (u) |value| u_norm_squared += @as(f64, value) * @as(f64, value);
            const u_norm = @sqrt(u_norm_squared);
            if (!std.math.isFinite(u_norm) or u_norm <= 1e-12) return TrainerError.InvalidSpectralState;
            const inverse_u_norm: f32 = @floatCast(1.0 / u_norm);
            for (u) |*value| value.* *= inverse_u_norm;
        }

        var sigma: f64 = 0.0;
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            var product: f64 = 0.0;
            var column: usize = 0;
            while (column < columns) : (column += 1) {
                product += @as(f64, embedding.weight.data[row * columns + column]) * @as(f64, v[column]);
            }
            sigma += product * @as(f64, u[row]);
        }
        if (!std.math.isFinite(sigma) or sigma <= 0.0) return TrainerError.InvalidSpectralState;
        if (sigma > 1.0) {
            const scale: f32 = @floatCast(1.0 / sigma);
            for (embedding.weight.data) |*weight| weight.* *= scale;
        }
    }

    pub fn enableEmbeddingAccelerator(self: *DistributedTrainerFuthark) !void {
        if (self.embedding_accel != null) return;
        self.embedding_accel = try EmbeddingAccelerator.init(
            &self.accelerator.ctx,
            self.vocab_size,
            self.model_dim,
            self.config.embedding_seed,
        );
    }

    pub fn buildKnowledgeGraph(self: *DistributedTrainerFuthark, text: []const u8) !void {
        if (text.len == 0) return TrainerError.EmptyKnowledgeGraphInput;

        _ = try self.crev_pipeline.processTextStream(text);
        const text_bytes = std.mem.sliceAsBytes(text);
        _ = try self.nsir_graph.encodeInformation(text_bytes);

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(self.knowledge_graph_nonce[0..]);
        hasher.update(text);
        var digest: [32]u8 = undefined;
        hasher.final(&digest);

        var node_id_buffer: [67]u8 = undefined;
        node_id_buffer[0] = 'k';
        node_id_buffer[1] = 'g';
        node_id_buffer[2] = '_';
        const hexadecimal = "0123456789abcdef";
        for (digest, 0..) |byte, index| {
            node_id_buffer[3 + index * 2] = hexadecimal[byte >> 4];
            node_id_buffer[4 + index * 2] = hexadecimal[byte & 0x0f];
        }
        const node_id = node_id_buffer[0..];

        const tree_id = try self.ensureKnowledgeFndsTree();
        const index_id = try self.ensureKnowledgeFndsIndex();
        _ = try self.fnds_manager.insertIntoTree(tree_id, node_id, text_bytes, 0);

        var location = try PatternLocation.init(
            self.allocator,
            tree_id,
            0,
            node_id,
            0,
            text_bytes.len,
            1.0,
        );
        var transferred = false;
        defer if (!transferred) location.deinit();
        try self.fnds_manager.addPatternToIndex(index_id, text_bytes, location);
        transferred = true;

        try self.r_gpu.distributeGraph(self.nsir_graph);
        try self.signal_engine.propagateStep();
    }
};