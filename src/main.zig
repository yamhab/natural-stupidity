const std = @import("std");
const Allocator = std.mem.Allocator;
const Init = std.process.Init;
const Io = std.Io;
const Random = std.Random;
const log = std.log;
const math = std.math;
const mem = std.mem;

const Network = struct {
    allocator: Allocator,
    layers: []Layer,

    fn init(allocator: Allocator, topology: []const usize) !@This() {
        var layers = try allocator.alloc(Layer, topology.len - 1);
        for (0..topology.len - 1) |i| {
            layers[i] = try Layer.init(allocator, topology[i], topology[i + 1]);
        }
        return @This(){ .allocator = allocator, .layers = layers };
    }

    fn deinit(self: *@This()) void {
        for (self.layers) |*layer|
            layer.deinit();
        self.allocator.free(self.layers);
    }

    fn mutate(self: *@This(), random: Random) void {
        for (self.layers) |*layer|
            layer.mutate(random);
    }

    fn infer(self: *@This(), inputs: []const f64) []f64 {
        self.layers[0].forward(inputs);
        for (1..self.layers.len) |i|
            self.layers[i].forward(self.layers[i - 1].outputs);
        return self.layers[self.layers.len - 1].outputs;
    }
};

const Layer = struct {
    const MUTATION_RATE: f64 = 0.2;
    const MUTATION_STRENGTH: f64 = 0.2;
    const WEIGHT_RANGE: f64 = 3.0;
    const BIAS_RANGE: f64 = 3.0;

    allocator: Allocator,
    weights: []f64,
    biases: []f64,
    outputs: []f64,

    fn init(allocator: Allocator, inputs: usize, outputs: usize) !@This() {
        return @This(){
            .allocator = allocator,
            .weights = try allocator.alloc(f64, inputs * outputs),
            .biases = try allocator.alloc(f64, outputs),
            .outputs = try allocator.alloc(f64, outputs),
        };
    }

    fn deinit(self: *@This()) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
        self.allocator.free(self.outputs);
    }

    fn mutate(self: *@This(), random: Random) void {
        mutateSlice(
            random,
            self.weights,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            WEIGHT_RANGE,
        );
        mutateSlice(
            random,
            self.biases,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            BIAS_RANGE,
        );
    }

    fn forward(self: *@This(), inputs: []const f64) void {
        for (self.outputs, 0..) |*output, i| {
            output.* = self.biases[i];
            for (inputs, 0..) |input, j|
                output.* += input * self.weights[i * inputs.len + j];
            output.* = sigmoid(output.*);
        }
    }
};

fn mutateSlice(
    random: Random,
    slice: []f64,
    rate: f64,
    strength: f64,
    range: f64,
) void {
    for (slice) |*element| {
        const chance = random.float(f64);
        if (chance < rate) {
            const tweak = random.floatNorm(f64) * strength;
            element.* = math.clamp(element.* + tweak, -range, range);
        }
    }
}

fn relu(n: f64) f64 {
    return @max(0.0, n);
}

fn sigmoid(n: f64) f64 {
    return 1.0 / (1.0 + math.exp(-n));
}

fn tanh(n: f64) f64 {
    return math.tanh(n);
}

pub fn main(init: Init) !void {
    const TOPOLOGY = [_]usize{ 2, 3, 1 };
    const XOR_INPUTS = [_][TOPOLOGY[0]]f64{
        [_]f64{ 0.0, 0.0 },
        [_]f64{ 1.0, 0.0 },
        [_]f64{ 0.0, 1.0 },
        [_]f64{ 1.0, 1.0 },
    };

    var network = try Network.init(init.gpa, &TOPOLOGY);
    defer network.deinit();
    var prng = Random.DefaultPrng.init(randomSeed(init.io));

    for (XOR_INPUTS) |inputs| {
        network.mutate(prng.random());
        log.info("{any}", .{network.infer(&inputs)});
    }
}

fn randomSeed(io: Io) u64 {
    var seed: u64 = undefined;
    Io.random(io, mem.asBytes(&seed));
    return seed;
}
