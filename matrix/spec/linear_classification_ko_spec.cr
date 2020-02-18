require "./spec_helper"

def project(input)
    x = input[0] - 150
    y = (input[0] - 150)**2
    [x, y]
end

describe Neuratron::LinearModel do
  it "linear classification" do
    inputs = [
        [120.0],
        [140.0],
        [160.0],
        [170.0],
    ]
    projected_inputs = inputs.map { |o| project o }
    expected_outputs = [
        [-1.0],
        [1.0],
        [1.0],
        [-1.0],
    ]

    model = Neuratron::LinearModel.new(2, 1)
    model.train(projected_inputs.flatten, expected_outputs.flatten)

    inputs_to_classify = [
        [111.0],
        [115.0],
        [125.0],
        [128.0],
        [131.0],
        [133.0],
        [136.0],
        [138.0],
        [141.0],
        [143.0],
        [147.0],
        [153.0],
        [157.0],
        [162.0],
        [164.0],
        [169.0],
        [174.0],
        [175.0],
        [179.0],
        [182.0],
    ]
    projected_inputs_to_classify = inputs_to_classify.map { |o| project o }

    pp inputs_to_classify.class
    pp projected_inputs_to_classify.class

    predictions = projected_inputs_to_classify.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input, Neuratron::LinearModel::Classification.new)
      puts "Prediction #{results}"
      results
    end

    points1 = Array(Tuple(Float64, Float64)).new
    points2 = Array(Tuple(Float64, Float64)).new

    positions = inputs_to_classify.zip(predictions).map do |data|
        point = { data[0][0], 0.0 }
        data[1] < 0 ? points1 << point : points2 << point
    end

    # pp "positions", positions

    math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
    fns = [
      AquaPlot::Scatter.from_points(points1).tap(&.set_title("Points")),
      AquaPlot::Scatter.from_points(points2).tap(&.set_title("Points")),
      AquaPlot::Function.new("0", title: "0"),
      # AquaPlot::Function.new(math_formulat, title: math_formulat),
    ]

    pp fns[-1].style = "pm3d"

    plt = AquaPlot::Plot.new fns
    plt.set_key("left box")
    plt.show
    # plt.set_view(100, 80, 1)
    # plt.show
    plt.close
  end
end