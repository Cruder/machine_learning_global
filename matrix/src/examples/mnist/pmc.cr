require "../../matrix.cr"

# stdout = IO::Memory.new
# Process.run("ls ./data", shell: true, output: stdout)
# puts stdout

# images = File.open("./data/train-images-idx3-ubyte")
# labels = File.open("./data/train-labels-idx1-ubyte")

# loader = Mnist::Loader.new(images, labels)
# data = loader.call

# images = data.map do |datum|
#   datum.image.map do |pixel|
#     clamp_value_rgb(pixel)
#   end
# end

# labels = data.map do |datum|
#   Array.new(10) { |i| i == datum.label ? 1.0 : -1.0 }
# end

# pp images[0]
# pp labels[0]

# print_mnist_ascii(images[0])

# model = Neuratron::DeepModel.new(28 * 28, 16, 10)

# model.train_classification(images, labels, 5, 0.01)

# model.save("./save/deep_mnist.json")


class DeepModelData
  JSON.mapping(
    layer_count: Int32,
    d: Array(Int32),
    w: Array(Array(Array(Int32)))
  )
end

json = File.read("./save/linear_model_mnist.json")
data = DeepModelData.from_json(json)

raw_w = data.w.map do |i|
  i.map do |j|
    j.to_unsafe
  end.to_unsafe
end.to_unsafe

pp raw_w

pp data.w
# raw_model = LibNeuratron::DeepModelData.new(size_input: data.input, size_output: data.output, inputs: data.weights.to_unsafe)
