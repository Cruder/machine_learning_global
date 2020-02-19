require "./spec_helper"

describe Neuratron::LinearModel do
  it "linear classification" do
    inputs = [
      [-4.0, -4.0],
      [-4.0, -3.0],
      [-4.0, -2.0],
      [-4.0, -1.0],
      [-4.0, 0.0],
      [-4.0, 1.0],
      [-4.0, 2.0],
      [-4.0, 3.0],
      [-4.0, 4.0],
      [-3.0, -4.0],
      [-3.0, -3.0],
      [-3.0, -2.0],
      [-3.0, -1.0],
      [-3.0, 0.0],
      [-3.0, 1.0],
      [-3.0, 2.0],
      [-3.0, 3.0],
      [-3.0, 4.0],
      [-2.0, -4.0],
      [-2.0, -3.0],
      [-2.0, -2.0],
      [-2.0, -1.0],
      [-2.0, 0.0],
      [-2.0, 1.0],
      [-2.0, 2.0],
      [-2.0, 3.0],
      [-2.0, 4.0],
      [-1.0, -4.0],
      [-1.0, -3.0],
      [-1.0, -2.0],
      [-1.0, -1.0],
      [-1.0, 0.0],
      [-1.0, 1.0],
      [-1.0, 2.0],
      [-1.0, 3.0],
      [-1.0, 4.0],
      [0.0, -4.0],
      [0.0, -3.0],
      [0.0, -2.0],
      [0.0, -1.0],
      [0.0, 0.0],
      [0.0, 1.0],
      [0.0, 2.0],
      [0.0, 3.0],
      [0.0, 4.0],
      [1.0, -4.0],
      [1.0, -3.0],
      [1.0, -2.0],
      [1.0, -1.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [1.0, 2.0],
      [1.0, 3.0],
      [1.0, 4.0],
      [2.0, -4.0],
      [2.0, -3.0],
      [2.0, -2.0],
      [2.0, -1.0],
      [2.0, 0.0],
      [2.0, 1.0],
      [2.0, 2.0],
      [2.0, 3.0],
      [2.0, 4.0],
      [3.0, -4.0],
      [3.0, -3.0],
      [3.0, -2.0],
      [3.0, -1.0],
      [3.0, 0.0],
      [3.0, 1.0],
      [3.0, 2.0],
      [3.0, 3.0],
      [3.0, 4.0],
      [4.0, -4.0],
      [4.0, -3.0],
      [4.0, -2.0],
      [4.0, -1.0],
      [4.0, 0.0],
      [4.0, 1.0],
      [4.0, 2.0],
      [4.0, 3.0],
      [4.0, 4.0]
    ]
    expected_outputs = [
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [1.0],
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0],
      [-1.0]
    ]


    model = Neuratron::LinearModel.new(2, 1)
    model.train(inputs.flatten, expected_outputs.flatten)

    inputs_to_classify = [
      [-3.6, -3.9],
      [-3.2, -3.6],
      [-4.0, -3.9],
      [-3.4, -2.3],
      [-3.1, -2.2],
      [-3.5, -2.5],
      [-3.5, -2.0],
      [-3.8, -1.3],
      [-3.7, -1.9],
      [-4.0, -0.09],
      [-3.6, -0.9],
      [-3.2, -0.09],
      [-3.3, 0.3],
      [-3.8, 0.1],
      [-3.8, 0.1],
      [-3.3, 1.0],
      [-4.0, 1.8],
      [-3.9, 1.7],
      [-3.6, 2.9],
      [-3.5, 2.4],
      [-4.0, 2.3],
      [-3.7, 3.7],
      [-3.2, 3.0],
      [-3.4, 3.1],
      [-3.9, 4.8],
      [-3.6, 4.9],
      [-3.7, 4.4],
      [-2.6, -3.8],
      [-2.4, -3.1],
      [-2.2, -3.8],
      [-3.0, -2.6],
      [-2.7, -2.8],
      [-2.8, -2.5],
      [-3.0, -2.0],
      [-2.9, -1.6],
      [-3.0, -1.5],
      [-3.0, -0.19],
      [-2.5, -1.0],
      [-2.4, -0.09],
      [-2.9, 0.2],
      [-2.5, 0.8],
      [-2.9, 0.5],
      [-2.1, 1.5],
      [-2.9, 1.1],
      [-2.4, 1.3],
      [-2.7, 2.8],
      [-2.2, 2.5],
      [-2.5, 2.2],
      [-2.9, 3.1],
      [-2.4, 3.2],
      [-2.8, 3.6],
      [-2.4, 4.5],
      [-2.3, 4.9],
      [-2.6, 4.2],
      [-1.5, -3.9],
      [-1.7, -3.8],
      [-2.0, -3.3],
      [-1.8, -3.0],
      [-1.7, -3.0],
      [-1.5, -3.0],
      [-1.2, -1.6],
      [-1.8, -2.0],
      [-1.1, -1.6],
      [-1.6, -0.30],
      [-1.9, -0.09],
      [-1.4, -0.7],
      [-1.5, 0.4],
      [-1.1, 0.9],
      [-2.0, 0.2],
      [-1.7, 1.7],
      [-1.1, 1.4],
      [-1.7, 1.2],
      [-1.2, 2.0],
      [-1.8, 2.2],
      [-1.2, 2.3],
      [-2.0, 3.5],
      [-2.0, 3.4],
      [-1.7, 3.5],
      [-1.8, 4.2],
      [-1.4, 4.7],
      [-1.7, 4.3],
      [-0.19, -3.6],
      [-0.30, -3.6],
      [-0.19, -3.6],
      [-0.09, -2.1],
      [-1.0, -3.0],
      [-0.30, -2.4],
      [-0.9, -2.0],
      [-0.5, -1.2],
      [-0.9, -1.6],
      [-0.09, -1.0],
      [-0.4, -1.0],
      [-0.5, -0.4],
      [-0.7, 0.8],
      [-0.5, 0.0],
      [-0.6, 0.4],
      [-0.6, 1.6],
      [-0.19, 1.0],
      [-0.09, 1.5],
      [-0.5, 2.6],
      [-0.09, 2.5],
      [-0.19, 2.9],
      [-0.19, 3.6],
      [-0.9, 3.4],
      [-0.7, 3.1],
      [-0.19, 4.2],
      [-0.9, 4.2],
      [-0.7, 4.6],
      [0.4, -3.3],
      [0.3, -3.9],
      [0.5, -3.7],
      [0.0, -2.2],
      [0.6, -2.7],
      [0.9, -2.7],
      [0.1, -1.5],
      [0.9, -1.4],
      [0.7, -1.1],
      [0.9, -0.5],
      [0.6, -0.4],
      [0.0, -0.8],
      [0.4, 0.0],
      [0.8, 0.3],
      [0.8, 0.9],
      [0.1, 1.2],
      [0.4, 1.4],
      [0.3, 1.6],
      [0.3, 2.0],
      [0.8, 2.7],
      [0.9, 2.1],
      [0.1, 3.2],
      [0.6, 3.0],
      [0.8, 3.4],
      [0.6, 4.1],
      [0.9, 4.4],
      [0.0, 4.4],
      [1.8, -3.2],
      [1.8, -4.0],
      [1.9, -3.2],
      [1.2, -2.4],
      [1.4, -2.4],
      [1.7, -2.4],
      [1.8, -1.6],
      [1.9, -1.7],
      [1.4, -1.3],
      [1.4, -0.5],
      [1.1, -0.4],
      [1.1, -0.19],
      [1.4, 0.6],
      [1.6, 0.4],
      [1.6, 0.5],
      [1.0, 1.6],
      [1.6, 1.8],
      [1.6, 1.5],
      [1.4, 2.8],
      [1.8, 2.9],
      [1.8, 2.8],
      [1.6, 3.7],
      [1.2, 3.1],
      [1.2, 3.5],
      [1.5, 4.8],
      [1.6, 4.1],
      [1.2, 4.8],
      [2.5, -4.0],
      [2.7, -3.1],
      [2.0, -3.4],
      [2.1, -2.8],
      [2.7, -2.7],
      [2.6, -2.1],
      [2.1, -1.5],
      [2.2, -1.9],
      [2.9, -1.5],
      [2.6, -0.09],
      [2.9, -0.30],
      [2.5, -0.30],
      [2.1, 0.0],
      [2.1, 0.7],
      [2.0, 0.2],
      [2.3, 1.5],
      [2.5, 1.5],
      [2.7, 1.4],
      [2.5, 2.8],
      [2.4, 2.2],
      [2.6, 2.2],
      [2.5, 3.8],
      [2.2, 3.2],
      [2.2, 3.8],
      [2.9, 4.1],
      [2.1, 4.8],
      [2.2, 4.3],
      [3.9, -3.7],
      [3.5, -3.7],
      [3.3, -3.9],
      [3.5, -2.7],
      [3.5, -2.5],
      [3.5, -2.7],
      [3.2, -1.9],
      [3.8, -1.4],
      [3.1, -1.5],
      [3.0, -1.0],
      [3.1, -0.5],
      [3.0, -1.0],
      [3.8, 0.8],
      [3.8, 0.2],
      [3.6, 0.2],
      [3.0, 1.3],
      [3.8, 1.8],
      [3.7, 1.3],
      [3.5, 2.0],
      [3.9, 2.6],
      [3.1, 2.1],
      [3.2, 3.7],
      [3.6, 3.3],
      [3.3, 3.5],
      [3.3, 4.1],
      [3.1, 4.7],
      [3.4, 4.8],
      [4.0, -3.9],
      [4.7, -3.4],
      [4.4, -3.2],
      [4.6, -2.9],
      [4.7, -2.3],
      [4.6, -2.5],
      [4.4, -1.3],
      [4.4, -1.2],
      [4.9, -1.3],
      [4.4, -0.5],
      [4.7, -0.5],
      [4.0, -1.0],
      [4.8, 0.7],
      [4.7, 0.5],
      [4.3, 0.9],
      [4.0, 1.5],
      [4.3, 1.1],
      [4.1, 1.9],
      [4.0, 2.4],
      [4.8, 2.7],
      [4.9, 2.5],
      [4.0, 3.5],
      [4.1, 3.3],
      [4.4, 3.5],
      [4.5, 4.2],
      [4.8, 4.6],
      [4.8, 4.8]
    ]


    predictions = inputs_to_classify.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input, Neuratron::LinearModel::Classification.new)
      puts "Prediction #{results}"
      results
    end

    points1 = Array(Tuple(Float64, Float64)).new
    points2 = Array(Tuple(Float64, Float64)).new

    positions = inputs_to_classify.zip(predictions).map do |data|
        point = { data[0][0], data[0][1] }
        data[1] < 0 ? points1 << point : points2 << point
    end

    # pp "positions", positions

    math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
    fns = [
      AquaPlot::Scatter.from_points(points1).tap(&.set_title("Points")),
      AquaPlot::Scatter.from_points(points2).tap(&.set_title("Points")),
      AquaPlot::Function.new("0", title: "0"),
    ]

    pp fns[-1].style = "pm3d"

    plt = AquaPlot::Plot.new fns
    plt.set_key("off box")
    plt.show
    # plt.set_view(100, 80, 1)
    # plt.show
    plt.close
  end
end
