require "./spec_helper"

describe Neuratron::LinearModel do
  it "with 4 exemples" do
    inputs = [
      [0.0, 0.0],
      [2.0, 2.0],
      [1.0, 1.0],
    ]
    expected_outputs = [
      [0.0],
      [0.0],
      [0.0],
    ]

    inputs_to_classify = [
      [-3.1, -3.1],
      [-3.7, -3.7],
      [-3.8, -2.2],
      [-3.1, -2.6],
      [-3.3, -2.0],
      [-3.2, -0.30],
      [-3.9, -1.0],
      [-3.1, 0.6],
      [-2.1, 0.8],
      [-2.3, 1.6],
      [-2.5, 2.1],
      [-1.2, -3.1],
      [-1.5, -2.0],
      [-1.6, 0.4],
      [-1.6, 1.6],
      [-2.0, 3.2],
      [-0.7, -3.7],
      [-0.30, -0.4],
      [-0.19, 0.2],
      [-0.19, 1.4],
      [-0.9, 3.8],
      [0.5, -3.9],
      [0.0, -1.7],
      [0.2, 0.4],
      [0.3, 2.6],
      [0.0, 4.0],
      [1.8, -2.8],
      [1.8, -1.0],
      [1.0, 1.2],
      [1.0, 2.3],
      [1.7, 3.8],
      [2.7, -3.1],
      [2.2, -1.2],
      [2.5, 0.4],
      [2.5, 2.9],
      [2.6, 4.5],
      [3.9, -3.5],
      [3.6, -1.7],
      [3.3, 0.8],
      [3.7, 2.6],
      [3.2, 4.1],
      [4.1, -2.2],
      [4.1, -0.19],
      [4.1, 1.6],
      [4.1, 2.4],
      [4.5, 4.8]
    ]

    model = Neuratron::LinearModel.new(2, 1)
    if model.train(inputs.flatten, expected_outputs.flatten)
      predictions = inputs_to_classify.map do |input|
        model.predict(input)
      end

      positions = inputs_to_classify.zip(predictions).map do |data|
        { data[0][0], data[0][1],  data[1] }
      end

      math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
      fns = [
        AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
        AquaPlot::Function.new("0", title: "0"),
        AquaPlot::Function.new(math_formulat, title: math_formulat),
      ]

      pp fns[-1].style = "pm3d"

      plt = AquaPlot::Plot3D.new fns
      plt.set_key("left box")
      plt.show
      plt.set_view(100, 80, 1)
      plt.show
      plt.close
    else
      pp "Impossible de calculer un plan"
    end
  end
end
