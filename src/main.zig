const std = @import("std");
const Allocator = std.mem.Allocator;
const Init = std.process.Init;
const Io = std.Io;
const Random = std.Random;
const math = std.math;
const mem = std.mem;

const POPULATION_COUNT = 1000;
const TOPOLOGY = [_]usize{ 2, 3, 2, 1 };
const TOPOLOGIES = [_][]const usize{&TOPOLOGY} ** POPULATION_COUNT;
const ELITE_COUNT: usize = 100;

const MUTATION_RATE: f64 = 0.5;
const MUTATION_STRENGTH: f64 = 0.5;
const MUTATION_RANGE: f64 = 3.0;

const TRIALS: usize = 1000;
const XOR_INPUTS_SET = [_][]const f64{
    &[_]f64{ 0.0, 0.0 },
    &[_]f64{ 1.0, 0.0 },
    &[_]f64{ 0.0, 1.0 },
    &[_]f64{ 1.0, 1.0 },
};

const Population = struct {
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

    fn mutate(
        self: *@This(),
        random: Random,
        elite: usize,
        rate: f64,
        strength: f64,
        range: f64,
    ) void {
        for (self.networks[elite..]) |*network|
            network.mutate(random, rate, strength, range);
    }

    fn evaluate(
        self: *@This(),
        inputs_set: []const []const f64,
        fitnessFn: fn ([]const f64, []const f64) f64,
    ) void {
        for (self.networks) |*network| {
            network.fitness = 0.0;
            for (inputs_set) |inputs|
                network.fitness += fitnessFn(inputs, network.infer(inputs));
        }
    }

    fn select(self: *@This(), elite: usize) !void {
        mem.sortUnstable(Network, self.networks, {}, struct {
            fn lessThanFn(_: void, a: Network, b: Network) bool {
                return a.fitness > b.fitness;
            }
        }.lessThanFn);

        for (self.networks[elite..], elite..) |*network, i| {
            network.deinit();
            network.* = try self.networks[i % elite].clone(self.allocator);
        }
    }
};

const Network = struct {
    allocator: Allocator,
    buffer: []f64,
    layers: []Layer,
    fitness: f64,

    fn init(allocator: Allocator, topology: []const usize) !@This() {
        var size: usize = 0;
        for (0..topology.len - 1) |i|
            size += topology[i + 1] * (topology[i] + 2);
        const buffer = try allocator.alloc(f64, size);

        const layers = try allocator.alloc(Layer, topology.len - 1);
        var offset: usize = 0;
        for (0..layers.len) |i| {
            const input_count = topology[i];
            const output_count = topology[i + 1];

            const weights = buffer[offset .. offset + input_count * output_count];
            offset += input_count * output_count;
            const biases = buffer[offset .. offset + output_count];
            offset += output_count;
            const outputs = buffer[offset .. offset + output_count];
            offset += output_count;

            layers[i] = Layer.init(weights, biases, outputs);
        }

        return @This(){ .allocator = allocator, .buffer = buffer, .layers = layers, .fitness = 0.0 };
    }

    fn clone(self: *@This(), allocator: Allocator) !@This() {
        var network = @This(){
            .allocator = allocator,
            .buffer = try allocator.alloc(f64, self.buffer.len),
            .layers = try allocator.alloc(Layer, self.layers.len),
            .fitness = self.fitness,
        };

        @memcpy(network.buffer, self.buffer);
        var offset: usize = 0;
        for (0..network.layers.len) |i| {
            const weights = network.buffer[0..self.layers[i].weights.len];
            offset += self.layers[i].weights.len;
            const biases = network.buffer[offset .. offset + self.layers[i].biases.len];
            offset += self.layers[i].biases.len;
            const outputs = network.buffer[offset .. offset + self.layers[i].outputs.len];
            offset += self.layers[i].outputs.len;

            network.layers[i] = Layer.init(weights, biases, outputs);
        }

        return network;
    }

    fn deinit(self: *@This()) void {
        self.allocator.free(self.layers);
        self.allocator.free(self.buffer);
    }

    fn mutate(self: *@This(), random: Random, rate: f64, strength: f64, range: f64) void {
        for (self.buffer) |*element| {
            const chance = random.float(f64);
            if (chance < rate) {
                const tweak = random.floatNorm(f64) * strength;
                element.* = math.clamp(element.* + tweak, -range, range);
            }
        }
    }

    fn infer(self: *@This(), inputs: []const f64) []const f64 {
        self.layers[0].forward(inputs);
        for (1..self.layers.len) |i|
            self.layers[i].forward(self.layers[i - 1].outputs);
        return self.layers[self.layers.len - 1].outputs;
    }
};

const Layer = struct {
    weights: []f64,
    biases: []f64,
    outputs: []f64,

    fn init(weights: []f64, biases: []f64, outputs: []f64) @This() {
        return @This(){ .weights = weights, .biases = biases, .outputs = outputs };
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
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &writer.interface;
    var prng = Random.DefaultPrng.init(randomSeed(init.io));

    var population = try Population.init(init.gpa, &TOPOLOGIES);
    defer population.deinit();
    population.mutate(prng.random(), 0, 1.0, 67.0, 69_420.0);
    try evolution(prng.random(), stdout, &population);

    try stdout.print("\nFittest specimen demo:\n", .{});
    for (XOR_INPUTS_SET) |inputs| {
        const outputs = population.networks[0].infer(inputs);
        try stdout.print("{} ^ {} = {}\n", .{ inputs[0], inputs[1], outputs[0] });
    }
    try stdout.flush();
}

fn randomSeed(io: Io) u64 {
    var seed: u64 = undefined;
    Io.random(io, mem.asBytes(&seed));
    return seed;
}

fn evolution(random: Random, stdout: *Io.Writer, population: *Population) !void {
    for (0..TRIALS) |i| {
        population.mutate(random, ELITE_COUNT, MUTATION_RATE, MUTATION_STRENGTH, MUTATION_RANGE);
        population.evaluate(&XOR_INPUTS_SET, struct {
            fn fitnessFn(inputs: []const f64, outputs: []const f64) f64 {
                const correct = @as(u1, @trunc(inputs[0])) ^ @as(u1, @trunc(inputs[1]));
                return 1.0 - @abs(correct - outputs[0]);
            }
        }.fitnessFn);
        try population.select(ELITE_COUNT);

        try stdout.print(
            "Generation #{}'s highest fitness score: {}%\n",
            .{ i + 1, population.networks[0].fitness / 4.0 * 100.0 },
        );
        try stdout.flush();
    }
}
