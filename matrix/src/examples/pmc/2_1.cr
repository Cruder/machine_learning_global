require "../../matrix"

inputs = [
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 5.0],
  [6.0, 6.0],
]
expected_outputs = [
  [1.0],
  [1.0],
  [-1.0],
  [-1.0],
]

input_to_classify = [
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 5.0],
  [6.0, 6.0],
  [1.3, 1.0],
  [5.5, 5.5],
  [1.5, 2.0],
  [7.7, 6.0],
  [0.98, 0.83],
  [7.42, 1.13],
  [7.32, 1.81],
  [1.0, 3.0],
  [4.86, 0.11],
  [10.5, 6.69],
  [6.8, 3.15],
  [1.42, 5.47],
]

model = Neuratron::DeepModel.new(2, 1)
model.train_classification(inputs, expected_outputs, 10000, 0.01)

predictions = input_to_classify.map do |input|
  puts "Predict for #{input}"
  results = model.predict(input)
  puts "Prediction #{results}"
  results
end

positions = input_to_classify.zip(predictions).map do |data|
  { data[0][0], data[0][1], data[1][0] }
end

pp positions

# math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
fns = [
  AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
  AquaPlot::Function.new("0", title: "0"),
  # AquaPlot::Function.new(math_formulat, title: math_formulat),
]

pp fns[-1].style = "pm3d"

plt = AquaPlot::Plot3D.new fns
plt.set_key("left box")
plt.show
plt.set_view(100, 80, 1)
plt.show
    plt.close
