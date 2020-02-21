require "../../matrix"

inputs = [
  [0.98, 0.83],
  [7.42, 1.13],
  [2.56, 0.67],
  [9.5, 4.41],
  [0.98, 1.45],
  [7.32, 1.81],
  [3.53, 1.33],
  [8.82, 3.89],
  [1.0, 3.0],
  [7.32, 2.97],
  [4.86, 0.11],
  [7.32, 2.97],
  [4.86, 0.11],
  [9.7, 5.19],
  [0.78, 4.15],
  [7.22, 3.49],
  [6.3, 0.31],
  [9.5, 5.97],
  [2.18, 2.59],
  [6.92, 4.17],
  [7.22, 0.37],
  [9.0, 7.0],
  [1.66, 1.09],
  [6.4, 4.55],
  [6.2, 1.47],
  [10.5, 6.69],
  [2.96, 0.13],
  [5.44, 5.17],
  [5.2, 2.15],
  [10.38, 5.81],
  [2.46, 1.83],
  [5.12, 5.51],
  [6.8, 3.15],
  [10.74, 4.61],
  [1.66, 4.51],
  [3.96, 6.23],
  [7.67, 2.43],
  [11.0, 4.0],
  [2.7, 3.31],
  [7.0, 6.0],
  [8.18, 2.67],
  [3.52, 2.19],
  [7.3, 5.35],
  [7.92, 4.25],
  [4.2, 0.85],
  [6.3, 5.93],
  [7.9, 4.63],
  [5.48, 6.41],
  [8.22, 5.85],
  [4.86, 1.98],
  [5.78, 3.91],
  [8.16, 7.01],
  [4.28, 2.65],
  [4.6, 4.77],
  [8.44, 5.41],
  [3.9, 3.19],
  [4.26, 5.23],
  [9.0, 3.0],
  [3.42, 3.92],
  [3.38, 6.21],
  [0.5, 5.47],
  [8.5, 1.33],
  [1.42, 5.47],
  [4.36, 4.05],
  [9.22, 1.95],
  [9.74, 1.29],
]
expected_outputs = [
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, 1.0, -1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, -1.0, 1.0],
  [1.0, -1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
  [-1.0, 1.0, -1.0],
]

input_to_classify = [
  [0.8, 0.7],
  [0.4, 0.4],
  [0.4, 1.4],
  [0.5, 1.1],
  [0.4, 2.8],
  [0.9, 2.8],
  [0.5, 3.8],
  [0.5, 3.2],
  [0.0, 4.8],
  [0.2, 4.7],
  [0.7, 5.0],
  [0.8, 5.9],
  [0.4, 6.8],
  [0.2, 6.9],
  [1.3, 0.4],
  [1.4, 0.9],
  [1.1, 1.8],
  [1.5, 1.6],
  [1.4, 2.2],
  [1.0, 2.0],
  [1.7, 3.4],
  [1.8, 3.6],
  [1.9, 4.5],
  [1.8, 4.6],
  [1.4, 5.3],
  [1.5, 5.0],
  [1.4, 6.5],
  [1.5, 6.3],
  [2.7, 0.8],
  [2.1, 0.3],
  [2.6, 1.9],
  [2.0, 1.1],
  [2.0, 2.1],
  [2.7, 2.3],
  [2.2, 3.0],
  [2.9, 3.7],
  [2.4, 4.4],
  [2.7, 4.4],
  [2.5, 5.8],
  [2.5, 5.9],
  [2.8, 6.3],
  [2.9, 6.7],
  [3.3, 0.3],
  [3.5, 0.7],
  [3.2, 1.2],
  [3.0, 1.3],
  [3.9, 2.4],
  [3.2, 2.2],
  [3.4, 3.0],
  [3.4, 3.3],
  [3.6, 4.9],
  [3.8, 4.6],
  [3.3, 5.9],
  [3.0, 5.0],
  [3.8, 6.9],
  [3.6, 6.4],
  [4.1, 0.1],
  [4.2, 0.2],
  [4.7, 1.9],
  [4.0, 1.1],
  [4.1, 2.9],
  [4.5, 2.4],
  [4.7, 3.8],
  [4.6, 3.4],
  [4.3, 4.6],
  [4.8, 4.1],
  [4.4, 5.4],
  [4.9, 5.2],
  [4.8, 6.0],
  [4.9, 6.3],
  [5.5, 0.6],
  [5.2, 0.6],
  [5.6, 1.8],
  [5.7, 1.4],
  [5.0, 2.9],
  [5.5, 2.8],
  [5.1, 3.1],
  [5.3, 3.6],
  [5.4, 4.6],
  [5.5, 4.3],
  [5.8, 5.4],
  [5.2, 5.3],
  [5.2, 6.9],
  [5.9, 6.2],
  [6.9, 0.0],
  [6.3, 0.8],
  [6.6, 1.4],
  [6.6, 1.9],
  [6.9, 2.5],
  [6.4, 2.5],
  [6.6, 3.7],
  [6.8, 3.3],
  [6.2, 4.7],
  [6.8, 4.8],
  [6.3, 5.2],
  [6.0, 5.1],
  [6.4, 6.3],
  [6.1, 6.7],
  [7.7, 0.5],
  [7.1, 0.3],
  [7.9, 1.9],
  [7.4, 1.8],
  [7.8, 2.9],
  [7.9, 2.0],
  [7.6, 3.8],
  [7.7, 3.9],
  [7.3, 4.8],
  [7.7, 4.6],
  [7.4, 5.9],
  [7.8, 5.6],
  [7.6, 6.8],
  [7.1, 6.1],
  [8.7, 0.9],
  [8.8, 0.1],
  [8.6, 1.9],
  [8.8, 1.6],
  [8.2, 2.2],
  [8.8, 2.7],
  [8.0, 3.8],
  [8.3, 3.6],
  [8.0, 4.2],
  [8.8, 4.6],
  [8.6, 5.4],
  [8.6, 5.4],
  [8.9, 6.7],
  [8.5, 6.2],
  [9.0, 0.1],
  [9.7, 0.4],
  [9.8, 1.5],
  [9.4, 1.8],
  [9.5, 2.7],
  [9.1, 2.5],
  [9.6, 3.9],
  [9.4, 3.7],
  [9.0, 4.3],
  [9.2, 4.5],
  [9.9, 5.6],
  [9.6, 5.8],
  [9.7, 6.6],
  [9.3, 6.0]
]

model = Neuratron::DeepModel.new(2, 3, 3)
model.train_classification(inputs, expected_outputs, 10000, 0.03)
predictions = input_to_classify.map do |input|
  puts "Predict for #{input}"
  results = model.predict(input)
  puts "Prediction #{results}"
  results
end
points = Array(Array(Tuple(Float64, Float64))).new(3) { Array(Tuple(Float64, Float64)).new }
positions = input_to_classify.zip(predictions).map do |data|
  pp data[1]
  points[max_indice(data[1])] << { data[0][0], data[0][1] }
end
aqua_points = Array(AquaPlot::Scatter).new
points.each do |point|
  aqua_points << AquaPlot::Scatter.from_points(point).tap(&.set_title("Points"))
end
#pp aqua_points[-1].style = "pm3d"

plt = AquaPlot::Plot.new aqua_points
plt.set_key("off box")
plt.show
# plt.set_view(100, 80, 1)
# plt.show
plt.close