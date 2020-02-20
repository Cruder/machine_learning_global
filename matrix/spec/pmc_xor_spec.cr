require "./spec_helper"

describe Neuratron::LinearModel do
  it "with 4 exemples" do
    inputs = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]

    expected_outputs = [
        [1.0],
        [-1.0],
        [1.0],
        [-1.0],
    ]

    inputs_to_classify = [
      [-1.0, -1.0],
      [-0.4, -0.19],
      [-0.5, -1.0],
      [-0.6, 0.0],
      [-0.9, 0.3],
      [-0.19, 0.8],
      [-0.19, 1.1],
      [-1.0, 1.6],
      [-0.30, 1.6],
      [-0.4, 2.9],
      [-0.30, 2.2],
      [-0.9, 2.4],
      [0.0, -0.9],
      [0.1, -1.0],
      [0.9, -0.8],
      [0.9, 0.7],
      [0.0, 0.1],
      [0.4, 0.2],
      [0.1, 1.9],
      [0.9, 1.3],
      [0.3, 1.2],
      [0.5, 2.1],
      [0.6, 2.2],
      [0.6, 2.7],
      [1.1, -0.5],
      [1.1, -1.0],
      [1.0, -0.4],
      [1.5, 0.1],
      [1.2, 0.3],
      [1.1, 0.4],
      [1.7, 1.8],
      [1.2, 1.5],
      [1.4, 1.4],
      [1.3, 2.3],
      [1.4, 2.6],
      [1.9, 2.6],
      [2.5, -0.19],
      [2.0, -1.0],
      [2.0, -0.4],
      [2.3, 0.4],
      [2.9, 0.1],
      [2.1, 0.9],
      [2.6, 1.6],
      [2.3, 1.6],
      [2.6, 1.1],
      [2.1, 2.3],
      [2.3, 2.2],
      [2.5, 2.2]
    ]

    model = Neuratron::DeepModel.new(2, 3, 1)
    model.train_classification(inputs, expected_outputs, 10000, 0.3)

    predictions = inputs_to_classify.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input)
      puts "Prediction #{results}"
      results
    end

    points1 = Array(Tuple(Float64, Float64)).new
    points2 = Array(Tuple(Float64, Float64)).new

    inputs_to_classify.zip(predictions).map do |data|
        point = { data[0][0], data[0][1] }
        data[1][0] < 0 ? points1 << point : points2 << point
    end

    pp "points 2", points2

    fns = [
      AquaPlot::Scatter.from_points(points1).tap(&.set_title("Points")),
      AquaPlot::Scatter.from_points(points2).tap(&.set_title("Points")),
      AquaPlot::Function.new("0", title: "0"),
      # AquaPlot::Function.new(math_formulat, title: math_formulat),
    ]

    pp fns[-1].style = "pm3d"

    plt = AquaPlot::Plot.new fns
    plt.set_key("off box")
    plt.show
    plt.close
  end
end
