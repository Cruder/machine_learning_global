require "../../matrix.cr"

stdout = IO::Memory.new
Process.run("ls ./data", shell: true, output: stdout)
puts stdout

images = File.open("./data/train-images-idx3-ubyte")
labels = File.open("./data/train-labels-idx1-ubyte")

loader = Mnist::Loader.new(images, labels)
data = loader.call

images = data.map do |datum|
  datum.image.map do |pixel|
    clamp_value_rgb(pixel)
  end
end

labels = data.map do |datum|
  Array.new(10) { |i| i == datum.label ? 1.0 : -1.0 }
end

# pp images[0]
# pp labels[0]

# print_mnist_ascii(images[0])

model = Neuratron::LinearModel.new(28 * 28, 10)

model.train(
  images,
  labels,
  Neuratron::LinearModel::Classification.new(alpha: 0.01, iteration: 5)
)


model.save("./save/linear_model_mnist.json")





# class LinearModelData
#   JSON.mapping(
#     input: Int32,
#     output: Int32,
#     weights: Array(Float64)
#   )
# end

# json = File.read("./save/linear_model_mnist.json")
# data = LinearModelData.from_json(json)
# pp data
# raw_model = LibNeuratron::LinearModel.new(size_input: data.input, size_output: data.output, inputs: data.weights.to_unsafe)

# model = Neuratron::LinearModel.new(pointerof(raw_model))

# pp model


# images_file = File.open("./data/t10k-images-idx3-ubyte")
# labels_file = File.open("./data/t10k-labels-idx1-ubyte")

# loader = Mnist::Loader.new(images_file, labels_file)
# data = loader.call

# images = data.map do |datum|
#   datum.image.map do |pixel|
#     clamp_value_rgb(pixel)
#   end
# end

# labels = data.map do |datum|
#   Array.new(10) { |i| i == datum.label ? 1.0 : -1.0 }
# end

# accuracy = images.zip(labels)[0..1].map do |image, label|
#   prediction = model.predict(image, kind: Neuratron::LinearModel::Classification.new)

#   predicted_number = max_indice(prediction)
#   real_label = max_indice(label)

#   # print_mnist_ascii(image)
#   puts "Label #{max_indice(label)} #{label}"
#   puts "Predictions #{max_indice(prediction)} #{prediction}"

#   # input = gets.not_nil!

#   predicted_number == real_label
# end

# puts "For #{accuracy.size} images"
# puts "  #{accuracy.select { |i| i }.size}"
# puts "Accuracy #{accuracy.select { |o| o }.size / accuracy.size * 100} %"
