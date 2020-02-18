require "./spec_helper"

describe Neuratron::LinearModel do
  it "with 50 exemples" do
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
    ]

    model = Neuratron::LinearModel.new(2, 1)
    model.train(inputs.flatten, expected_outputs.flatten)

    predictions = inputs.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input)
      puts "Prediction #{results}"
      results
    end

    positions = inputs.zip(predictions).map do |data|
      { data[0][0], data[0][1],  data[1] }
    end

    pp "positions", positions

    math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
    fns = [
      AquaPlot::Function.new("0", title: "0"),
      AquaPlot::Function.new(math_formulat, title: math_formulat),
      AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
    ]

    pp fns[1].style = "pm3d"

    plt = AquaPlot::Plot3D.new fns
    plt.set_key("left box")
    plt.show
    plt.set_view(100, 80, 1)
    plt.show
    plt.close
  end

  it "when inputs are not distincts" do

  end
end
