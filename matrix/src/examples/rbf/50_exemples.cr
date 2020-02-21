require "../../matrix"

inputs = [
  [1.0, 2.0],
  [1.5, 2.5],
  [3.0, 4.0],
  [3.2, 4.5],
  [3.7, 4.4],
  [3.5, 4.5],
  [4.7, 6.1],
  [5.4, 5.4],
  [5.2, 5.2],
  [4.3, 4.3],

  [5.5, 5.5],
  [5.0, 5.5],
  [6.0, 6.0],
  [-0.23, -0.04],
  [-0.39, 0.89],
  [-0.30, 1.80],
  [-0.03, 2.58],
  [-0.35, 4.16],
  [0.24, 4.64],
  [-0.24, 6.07],

  [0.14, 7.33],
  [1.00, 0.18],
  [1.40, 0.71],
  [1.12, 1.52],
  [0.68, 3.08],
  [0.78, 4.32],
  [0.73, 5.00],
  [1.49, 6.02],
  [0.59, 7.14],
  [1.80, 0.31],

  [2.13, 0.54],
  [1.80, 1.67],
  [1.75, 3.21],
  [2.14, 4.05],
  [1.90, 4.52],
  [2.02, 6.33],
  [1.54, 7.11],
  [3.21, 0.04],
  [2.93, 0.98],
  [2.78, 1.58],

  [2.76, 2.66],
  [2.90, 3.83],
  [3.36, 4.65],
  [2.69, 5.85],
  [2.73, 7.47],
  [3.69, 0.07],
  [3.63, 1.28],
  [3.65, 1.60],
  [3.62, 3.40],
  [3.62, 3.70],

  [4.16, 5.11],
  [3.74, 6.04],
  [4.47, 7.01],
  [4.68, 0.23],
  [5.22, 0.52],
  [4.86, 1.69],
  [5.27, 3.02],
  [4.66, 3.98],
  [4.86, 5.07],
  [5.33, 6.38],

  [5.08, 6.74],
  [5.78, 0.16],
  [5.88, 1.34],
  [5.71, 2.01],
  [6.34, 2.51],
  [5.84, 4.46],
  [6.13, 4.89],
  [6.02, 6.20],
  [5.57, 6.52],
  [6.79, -0.15],

  [6.63, 1.00],
  [7.24, 2.45],
  [6.53, 2.66],
  [7.02, 4.16],
  [6.56, 4.92],
  [6.70, 6.27],
  [6.95, 6.86],
]
expected_outputs = [
  [2.0],
  [2.5],
  [3.0],
  [3.5],
  [3.2],
  [3.5],
  [2.1],
  [1.8],
  [1.5],
  [1.3],

  [1.2],
  [1.1],
  [40.0],
  [-4.82],
  [-5.87],
  [-6.08],
  [-9.59],
  [-8.18],
  [-12.07],
  [-12.58],

  [1.17],
  [1.80],
  [-0.37],
  [-4.31],
  [-5.79],
  [-6.99],
  [-5.80],
  [-10.70],
  [3.83],
  [4.67],

  [1.79],
  [-0.67],
  [-0.57],
  [-2.11],
  [-4.40],
  [-7.28],
  [9.23],
  [6.84],
  [5.39],
  [3.72],

  [2.48],
  [2.86],
  [-1.33],
  [-3.62],
  [10.90],
  [8.85],
  [8.47],
  [5.64],
  [5.20],
  [5.01],

  [2.13],
  [3.25],
  [14.16],
  [15.66],
  [12.61],
  [12.08],
  [8.48],
  [7.56],
  [7.28],
  [5.82],

  [18.18],
  [16.79],
  [15.15],
  [16.65],
  [11.96],
  [12.33],
  [9.99],
  [7.88],
  [22.24],
  [19.95],

  [19.93],
  [17.11],
  [16.59],
  [13.81],
  [12.27],
  [12.29],
  [12.28],
]

inputs_to_classify = [
  [-3.1, -3.1],
  [-3.7, -3.7],
  [-3.8, -2.2],
  [-3.1, -2.6],
  [-3.3, -2.0],
  [-3.7, -1.9],
  [-3.2, -0.30],
  [-3.9, -1.0],
  [-3.1, 0.6],
  [-3.1, 0.7],
  [-3.2, 1.3],
  [-4.0, 1.1],
  [-3.4, 2.2],
  [-3.1, 2.8],
  [-3.9, 3.4],
  [-3.6, 3.1],
  [-3.9, 4.1],
  [-3.6, 4.6],
  [-2.7, -4.0],
  [-2.9, -4.0],
  [-2.3, -2.4],
  [-2.4, -2.4],
  [-2.7, -1.8],
  [-2.5, -1.1],
  [-2.6, -0.09],
  [-2.5, -0.8],
  [-2.1, 0.8],
  [-2.2, 0.6],
  [-2.3, 1.6],
  [-2.5, 1.7],
  [-2.5, 2.1],
  [-2.8, 2.8],
  [-2.8, 3.4],
  [-3.0, 3.0],
  [-2.6, 4.7],
  [-2.7, 4.9],
  [-1.2, -3.1],
  [-1.4, -3.3],
  [-1.7, -2.2],
  [-1.2, -2.1],
  [-1.5, -2.0],
  [-1.9, -1.8],
  [-1.2, -0.6],
  [-1.6, -0.19],
  [-1.6, 0.4],
  [-1.4, 0.9],
  [-1.6, 1.6],
  [-2.0, 1.7],
  [-1.7, 2.7],
  [-1.1, 2.2],
  [-2.0, 3.2],
  [-1.1, 3.2],
  [-1.3, 4.5],
  [-1.6, 4.3],
  [-0.7, -3.7],
  [-0.8, -3.8],
  [-0.4, -2.8],
  [-0.8, -2.2],
  [-0.4, -1.8],
  [-0.30, -1.9],
  [-0.30, -0.4],
  [-0.5, -0.5],
  [-0.30, 0.3],
  [-0.19, 0.2],
  [-0.5, 1.9],
  [-0.19, 1.4],
  [-0.8, 2.4],
  [-0.09, 2.6],
  [-0.9, 3.2],
  [-0.9, 3.8],
  [-0.6, 4.9],
  [-0.30, 4.4],
  [0.0, -3.9],
  [0.5, -3.9],
  [0.8, -2.4],
  [0.5, -2.6],
  [0.5, -1.9],
  [0.0, -1.7],
  [0.5, -0.5],
  [0.9, -0.9],
  [0.9, 0.7],
  [0.2, 0.4],
  [0.0, 1.4],
  [0.9, 1.1],
  [0.3, 2.6],
  [0.3, 2.1],
  [0.8, 3.0],
  [0.4, 3.2],
  [0.0, 4.0],
  [0.5, 4.9],
  [1.6, -3.1],
  [1.6, -3.4],
  [1.8, -2.8],
  [1.2, -2.1],
  [1.9, -1.3],
  [1.0, -1.8],
  [1.8, -1.0],
  [1.9, -0.30],
  [1.3, 0.7],
  [1.0, 0.8],
  [1.0, 1.2],
  [1.9, 1.2],
  [1.0, 2.3],
  [1.6, 2.7],
  [1.9, 3.8],
  [1.7, 3.8],
  [1.2, 4.6],
  [1.1, 4.2],
  [2.9, -3.2],
  [2.7, -3.1],
  [2.2, -2.4],
  [2.4, -2.7],
  [2.6, -1.7],
  [2.2, -1.2],
  [2.8, -0.8],
  [2.5, -0.19],
  [2.5, 0.4],
  [2.3, 0.2],
  [2.5, 1.1],
  [2.1, 1.4],
  [2.5, 2.9],
  [2.2, 2.9],
  [2.3, 3.7],
  [2.0, 3.8],
  [2.6, 4.5],
  [2.6, 4.8],
  [3.8, -3.9],
  [3.9, -3.5],
  [3.7, -2.2],
  [3.1, -2.7],
  [3.1, -1.2],
  [3.6, -1.7],
  [3.2, -0.7],
  [3.2, -1.0],
  [3.7, 0.0],
  [3.3, 0.8],
  [3.8, 1.1],
  [3.2, 1.9],
  [3.7, 2.6],
  [3.9, 2.3],
  [3.9, 3.2],
  [3.0, 3.3],
  [3.2, 4.1],
  [3.5, 4.4],
  [4.9, -3.2],
  [4.8, -4.0],
  [4.1, -2.2],
  [4.1, -2.9],
  [4.8, -1.7],
  [4.1, -1.5],
  [4.1, -0.19],
  [4.5, -0.9],
  [4.1, 0.2],
  [4.3, 0.8],
  [4.1, 1.6],
  [4.9, 1.2],
  [4.8, 2.8],
  [4.1, 2.4],
  [4.0, 3.1],
  [4.0, 3.6],
  [4.6, 4.9],
  [4.5, 4.8]
]

model = Neuratron::RBFModel.new(inputs, 1, 0.1)
model.train(expected_outputs, 1)

predictions = model.predict(inputs_to_classify).as(Array(Float64))
pp predictions
positions = inputs_to_classify.zip(predictions).map do |data|
    pp "heyaa"
    pp data
  #{ data[0][0], data[0][1],  data[1][0] }
end

pp "positions", positions

#math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
#fns = [
#  AquaPlot::Function.new("0", title: "0"),
#  AquaPlot::Function.new(math_formulat, title: math_formulat),
#  AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
#]

#pp fns[1].style = "pm3d"

#plt = AquaPlot::Plot3D.new fns
#plt.set_key("left box")
#plt.show
#plt.set_view(100, 80, 1)
#plt.show
#plt.close
