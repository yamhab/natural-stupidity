const std = @import("std");
const Allocator = std.mem.Allocator;
const Init = std.process.Init;
const Io = std.Io;
const Random = std.Random;
const math = std.math;
const mem = std.mem;

const POPULATION_COUNT = 500;
const TOPOLOGY = [_]usize{ 9, 72, 36, 9 };
const TOPOLOGIES = [_][]const usize{&TOPOLOGY} ** POPULATION_COUNT;
const ELITE_COUNT: usize = 50;

const MUTATION_RATE: f64 = 0.5;
const MUTATION_STRENGTH: f64 = 0.5;
const MUTATION_RANGE: f64 = 5.0;

const TRIALS: usize = 500;

const XOR_INPUTS_SET = [_][]const f64{
    &[_]f64{ 0.0, 0.0 },
    &[_]f64{ 1.0, 0.0 },
    &[_]f64{ 0.0, 1.0 },
    &[_]f64{ 1.0, 1.0 },
};

const GAMES: usize = 100;

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
        for (self.networks) |*network| network.deinit();
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
        stdout: *Io.Writer,
        random: Random,
        fitnessFn: fn (*Io.Writer, Random, []Network) anyerror!void,
    ) !void {
        try fitnessFn(stdout, random, self.networks);
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

            layers[i] = Layer.init(weights, biases, outputs, relu);
        }
        layers[layers.len - 1].activationFn = sigmoid;

        return @This(){ .allocator = allocator, .buffer = buffer, .layers = layers, .fitness = 0.0 };
    }

    fn clone(self: *const @This(), allocator: Allocator) !@This() {
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

            network.layers[i] = Layer.init(weights, biases, outputs, relu);
        }
        network.layers[network.layers.len - 1].activationFn = sigmoid;

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
    activationFn: *const fn (f64) f64,

    fn init(weights: []f64, biases: []f64, outputs: []f64, activationFn: fn (f64) f64) @This() {
        return @This(){
            .weights = weights,
            .biases = biases,
            .outputs = outputs,
            .activationFn = activationFn,
        };
    }

    fn forward(self: *@This(), inputs: []const f64) void {
        for (self.outputs, 0..) |*output, i| {
            output.* = self.biases[i];
            for (inputs, 0..) |input, j|
                output.* += input * self.weights[i * inputs.len + j];
            output.* = self.activationFn(output.*);
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
    var write_buffer: [16384]u8 = undefined;
    var writer = std.Io.File.stdout().writer(init.io, &write_buffer);
    const stdout = &writer.interface;

    var read_buffer: [16384]u8 = undefined;
    var reader = std.Io.File.stdin().reader(init.io, &read_buffer);
    const stdin = &reader.interface;

    var prng = Random.DefaultPrng.init(randomSeed(init.io));

    var population = try Population.init(init.gpa, &TOPOLOGIES);
    defer population.deinit();
    population.mutate(prng.random(), 0, 1.0, 67.0, 69_420.0);
    try evolution(stdout, prng.random(), &population);

    try stdout.print("\nFittest agent(s) demo:\n", .{});
    population.networks[0].fitness = 0.0;
    for (0..1000) |j| {
        try stdout.print("\nFittest agent's game #{}\n", .{j + 1});
        try playGameRandom(stdout, prng.random(), &population.networks[0]);
    }
    try stdout.print("\nFittest agent's fitness: {}\n", .{population.networks[0].fitness});

    try stdout.flush();

    try stdout.print(
        \\
        \\Population size: {} agents ({}% elite, {} games each)
        \\Network topology: {any}
        \\Mutation rate, strength, range: {}%, {}, {}
        \\
    , .{
        POPULATION_COUNT,
        @as(f64, @floatFromInt(ELITE_COUNT)) / @as(f64, @floatFromInt(POPULATION_COUNT)) * 100.0,
        GAMES,
        TOPOLOGY,
        MUTATION_RATE * 100.0,
        MUTATION_STRENGTH,
        MUTATION_RANGE,
    });
    try stdout.flush();

    while (true) try playGamePlayer(stdin, stdout, &population.networks[0]);
}

fn randomSeed(io: Io) u64 {
    var seed: u64 = undefined;
    Io.random(io, mem.asBytes(&seed));
    return seed;
}

fn evolution(stdout: *Io.Writer, random: Random, population: *Population) !void {
    for (0..TRIALS) |i| {
        population.mutate(random, ELITE_COUNT, MUTATION_RATE, MUTATION_STRENGTH, MUTATION_RANGE);
        try population.evaluate(stdout, random, ticTacToeFitnessFn);
        try population.select(ELITE_COUNT);

        var sum: f64 = 0.0;
        for (population.networks) |network| sum += network.fitness;
        try stdout.print("Generation #{} total fitness: {}\n", .{ i + 1, sum });
        try stdout.flush();
    }
}

fn xorFitnessFn(_: *Io.Writer, networks: []Network) void {
    for (networks) |*network| {
        network.fitness = 4.0;
        for (XOR_INPUTS_SET) |inputs| {
            const outputs = network.infer(inputs);
            const correct = @as(u1, @trunc(inputs[0])) ^ @as(u1, @trunc(inputs[1]));
            network.fitness -= @abs(correct - outputs[0]);
        }
        network.fitness /= 4.0;
    }
}

fn ticTacToeFitnessFn(stdout: *Io.Writer, random: Random, networks: []Network) !void {
    for (networks, 0..) |*network, i| {
        network.fitness = 0.0;
        for (0..GAMES) |j| {
            try stdout.print("\nAgent #{}'s game #{}\n", .{ i + 1, j + 1 });
            try playGameRandom(stdout, random, network);
        }
        try stdout.print("\nAgent #{}'s fitness: {}\n", .{ i + 1, network.fitness });
    }
}

fn playGameRandom(stdout: *Io.Writer, random: Random, network: *Network) !void {
    var board = [_]f64{0.0} ** 9;

    while (true) {
        var outputs: [9]f64 = undefined;
        @memcpy(&outputs, network.infer(&board));

        var max: usize = undefined;
        for (0..9) |_| {
            max = mem.findMax(f64, &outputs);
            if (board[max] != 0.0) {
                outputs[max] = -69_420.0;
            } else {
                break;
            }
        }
        board[max] = 1.0;
        try printBoard(stdout, &board);

        var player = status(&board);
        if (player == 1.0) {
            try stdout.print("\nWin!\n", .{});
            network.fitness += 1.0;
            break;
        } else if (player == 0.0) {
            try stdout.print("\nDraw!\n", .{});
            network.fitness += 0.25;
            break;
        } else if (player == -1.0) {
            try stdout.print("\nLoss!\n", .{});
            break;
        }

        while (true) {
            const move = random.uintLessThan(usize, 9);
            if (board[move] == 0.0) {
                board[move] = -1.0;
                break;
            }
        }
        try printBoard(stdout, &board);

        player = status(&board);
        if (player == 1.0) {
            try stdout.print("\nWin!\n", .{});
            network.fitness += 1.0;
            break;
        } else if (player == 0.0) {
            try stdout.print("\nDraw!\n", .{});
            network.fitness += 0.25;
            break;
        } else if (player == -1.0) {
            try stdout.print("\nLoss!\n", .{});
            break;
        }
    }
}

fn printBoard(stdout: *Io.Writer, board: []const f64) !void {
    try stdout.print(
        \\
        \\ {s} | {s} | {s}
        \\-----------
        \\ {s} | {s} | {s}
        \\-----------
        \\ {s} | {s} | {s}
        \\
    , .{
        if (board[0] == 1.0) "X" else if (board[0] == -1.0) "O" else " ",
        if (board[1] == 1.0) "X" else if (board[1] == -1.0) "O" else " ",
        if (board[2] == 1.0) "X" else if (board[2] == -1.0) "O" else " ",
        if (board[3] == 1.0) "X" else if (board[3] == -1.0) "O" else " ",
        if (board[4] == 1.0) "X" else if (board[4] == -1.0) "O" else " ",
        if (board[5] == 1.0) "X" else if (board[5] == -1.0) "O" else " ",
        if (board[6] == 1.0) "X" else if (board[6] == -1.0) "O" else " ",
        if (board[7] == 1.0) "X" else if (board[7] == -1.0) "O" else " ",
        if (board[8] == 1.0) "X" else if (board[8] == -1.0) "O" else " ",
    });
}

fn status(board: []const f64) ?f64 {
    for ([_]f64{ -1.0, 1.0 }) |player| {
        if (board[0] == player and board[1] == player and board[2] == player or
            board[3] == player and board[4] == player and board[5] == player or
            board[6] == player and board[7] == player and board[8] == player or
            board[0] == player and board[3] == player and board[6] == player or
            board[1] == player and board[4] == player and board[7] == player or
            board[2] == player and board[5] == player and board[8] == player or
            board[0] == player and board[4] == player and board[8] == player or
            board[2] == player and board[4] == player and board[6] == player)
        {
            return player;
        }
    }

    if (mem.findScalar(f64, board, 0.0) == null) return 0.0;
    return null;
}

fn playGamePlayer(stdin: *Io.Reader, stdout: *Io.Writer, network: *Network) !void {
    try stdout.print("\nYou are O!\n", .{});
    try stdout.flush();

    var board = [_]f64{0.0} ** 9;
    while (true) {
        var outputs: [9]f64 = undefined;
        @memcpy(&outputs, network.infer(&board));

        var max: usize = undefined;
        for (0..9) |_| {
            max = mem.findMax(f64, &outputs);
            if (board[max] != 0.0) {
                outputs[max] = -69_420.0;
            } else {
                break;
            }
        }
        board[max] = 1.0;
        try printBoard(stdout, &board);
        try stdout.flush();

        var player = status(&board);
        if (player == -1.0) {
            try stdout.print("\nWin!\n", .{});
            try stdout.flush();
            break;
        } else if (player == 0.0) {
            try stdout.print("\nDraw!\n", .{});
            try stdout.flush();
            break;
        } else if (player == 1.0) {
            try stdout.print("\nLoss!\n", .{});
            try stdout.flush();
            break;
        }

        while (true) {
            try stdout.print("\nMove? ", .{});
            try stdout.flush();
            const input = try stdin.takeDelimiter('\n') orelse continue;
            const move = std.fmt.parseUnsigned(usize, input, 10) catch continue;
            if (move == 0 or move > 9) continue;
            if (board[move - 1] == 0.0) {
                board[move - 1] = -1.0;
                break;
            }
        }
        try printBoard(stdout, &board);
        try stdout.flush();

        player = status(&board);
        if (player == -1.0) {
            try stdout.print("\nWin!\n", .{});
            try stdout.flush();
            break;
        } else if (player == 0.0) {
            try stdout.print("\nDraw!\n", .{});
            try stdout.flush();
            break;
        } else if (player == 1.0) {
            try stdout.print("\nLoss!\n", .{});
            try stdout.flush();
            break;
        }
    }
}
