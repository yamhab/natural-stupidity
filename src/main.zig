const std = @import("std");
const Allocator = std.mem.Allocator;
const Init = std.process.Init;
const Io = std.Io;
const Random = std.Random;
const math = std.math;
const mem = std.mem;

const Population = struct {
    const ELITE_COUNT: usize = 50;

    allocator: Allocator,
    networks: []Network,

    fn init(allocator: Allocator, topologies: []const []const usize) !@This() {
        const networks = try allocator.alloc(Network, topologies.len);
        for (networks, 0..) |*network, i|
            network.* = try Network.init(allocator, topologies[i]);
        return @This(){ .allocator = allocator, .networks = networks };
    }

    fn deinit(self: *@This()) void {
        for (self.networks) |*network|
            network.deinit();
        self.allocator.free(self.networks);
    }

    fn mutate(self: *@This(), random: Random) void {
        for (self.networks[ELITE_COUNT..]) |*network|
            network.mutate(random);
    }

    fn evaluate(self: *@This(), inputs_set: []const []const f64) void {
        for (self.networks) |*network| {
            network.fitness = @as(f64, @floatFromInt(inputs_set.len));
            for (inputs_set) |inputs| {
                const correct = @as(u1, @trunc(inputs[0])) ^ @as(u1, @trunc(inputs[1]));
                const prediction = network.infer(inputs)[0];
                const err = @abs(correct - prediction);
                network.fitness -= err;
            }
        }
    }

    fn select(self: *@This()) !void {
        mem.sortUnstable(Network, self.networks, {}, comptime rankNetworks);
        for (self.networks[ELITE_COUNT..], ELITE_COUNT..) |*network, i| {
            network.deinit();
            network.* = try self.networks[i % ELITE_COUNT].clone(self.allocator);
        }
    }
};

fn rankNetworks(_: void, a: Network, b: Network) bool {
    return a.fitness > b.fitness;
}

const Network = struct {
    allocator: Allocator,
    layers: []Layer,
    fitness: f64,

    fn init(allocator: Allocator, topology: []const usize) !@This() {
        var layers = try allocator.alloc(Layer, topology.len - 1);
        for (0..layers.len) |i|
            layers[i] = try Layer.init(allocator, topology[i], topology[i + 1]);
        return @This(){ .allocator = allocator, .layers = layers, .fitness = 0.0 };
    }

    fn clone(self: *@This(), allocator: Allocator) !@This() {
        var network = @This(){
            .allocator = allocator,
            .layers = try allocator.alloc(Layer, self.layers.len),
            .fitness = self.fitness,
        };
        for (0..network.layers.len) |i|
            network.layers[i] = try self.layers[i].clone(allocator);
        return network;
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
    const MUTATION_RATE: f64 = 0.5;
    const MUTATION_STRENGTH: f64 = 0.5;
    const WEIGHT_RANGE: f64 = 3.0;
    const BIAS_RANGE: f64 = 3.0;

    allocator: Allocator,
    weights: []f64,
    biases: []f64,
    outputs: []f64,

    fn init(allocator: Allocator, input_count: usize, output_count: usize) !@This() {
        return @This(){
            .allocator = allocator,
            .weights = try allocator.alloc(f64, input_count * output_count),
            .biases = try allocator.alloc(f64, output_count),
            .outputs = try allocator.alloc(f64, output_count),
        };
    }

    fn clone(self: *@This(), allocator: Allocator) !@This() {
        const layer = @This(){
            .allocator = allocator,
            .weights = try allocator.alloc(f64, self.weights.len),
            .biases = try allocator.alloc(f64, self.biases.len),
            .outputs = try allocator.alloc(f64, self.outputs.len),
        };
        @memcpy(layer.weights, self.weights);
        @memcpy(layer.biases, self.biases);
        @memcpy(layer.outputs, self.outputs);
        return layer;
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
            output.* = relu(output.*);
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
    const TOPOLOGY = [_]usize{ 2, 3, 2, 1 };
    const POPULATION_COUNT = 1000;
    const TRIALS: usize = 1000;
    const XOR_INPUTS_SET = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
    };

    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(init.io, &buffer);
    var stdout = &writer.interface;

    var prng = Random.DefaultPrng.init(randomSeed(init.io));
    var population = try Population.init(init.gpa, &[_][]const usize{&TOPOLOGY} ** POPULATION_COUNT);
    defer population.deinit();

    for (0..TRIALS) |i| {
        population.mutate(prng.random());
        population.evaluate(&XOR_INPUTS_SET);
        try population.select();

        try stdout.print(
            "Generation #{}'s highest fitness score: {}%\n",
            .{ i + 1, population.networks[0].fitness / 4.0 * 100.0 },
        );
        try writer.flush();
    }

    try stdout.print("\nFittest specimen demo:\n", .{});
    for (XOR_INPUTS_SET) |inputs| {
        const outputs = population.networks[0].infer(inputs);
        try stdout.print("{} ^ {} = {}\n", .{ inputs[0], inputs[1], outputs[0] });
    }
    try writer.flush();
}

fn randomSeed(io: Io) u64 {
    var seed: u64 = undefined;
    Io.random(io, mem.asBytes(&seed));
    return seed;
}
